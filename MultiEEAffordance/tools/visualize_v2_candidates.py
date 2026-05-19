#!/usr/bin/env python3
"""Create human-review visualizations for v2 3D candidate regions.

This script is intentionally separate from render_candidate_overlays_v2.py.
The overlay renderer is optimized for VLM candidate selection, while this file
is optimized for a human reviewer who needs to inspect where each candidate id
actually lies on the object.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from path_utils import relative_to_dataset, resolve_portable_path


PALETTE = [
    (255, 72, 72),
    (50, 220, 120),
    (255, 210, 64),
    (120, 160, 255),
    (255, 112, 210),
    (70, 230, 235),
    (255, 145, 80),
    (190, 125, 255),
    (170, 230, 90),
    (255, 255, 255),
    (255, 120, 120),
    (120, 255, 180),
    (255, 230, 130),
    (160, 190, 255),
    (255, 160, 230),
    (120, 245, 245),
    (255, 180, 130),
    (215, 170, 255),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize v2 candidates for human review.")
    parser.add_argument("--dataset-root", default=".", help="Dataset root directory.")
    parser.add_argument(
        "--pilot-csv",
        default="processed/metadata/vlm_pilot_samples_v0_1.csv",
        help="Pilot CSV relative to dataset root.",
    )
    parser.add_argument(
        "--candidate-root",
        default="processed/vlm_candidate_v2/3d_candidates",
        help="Candidate root relative to dataset root.",
    )
    parser.add_argument(
        "--renders-root",
        default="processed/vlm_semantic_part/renders",
        help="Preferred render root relative to dataset root.",
    )
    parser.add_argument(
        "--fallback-renders-root",
        default="processed/vlm_pilot/renders",
        help="Fallback render root relative to dataset root.",
    )
    parser.add_argument(
        "--rule-root",
        default="processed/vlm_candidate_v2/rule_filter",
        help="Optional rule-filter root used when --selected-candidates is empty.",
    )
    parser.add_argument(
        "--output-root",
        default="processed/vlm_candidate_v2/review_visualizations",
        help="Output root relative to dataset root.",
    )
    parser.add_argument("--pilot-id", required=True, help="Pilot id to visualize.")
    parser.add_argument(
        "--selected-candidates",
        default="",
        help="Comma-separated candidate ids to emphasize. If empty, accepted candidates from rule_filter are used.",
    )
    parser.add_argument("--views", default="", help="Optional comma-separated view names. Default: all views.")
    parser.add_argument("--point-radius", type=int, default=6, help="Candidate overlay point radius.")
    parser.add_argument("--alpha", type=float, default=0.95, help="Candidate overlay opacity.")
    parser.add_argument("--crop-padding", type=int, default=72, help="Padding around selected candidate crop.")
    parser.add_argument("--grid-cols", type=int, default=3, help="Columns for per-candidate contact sheet.")
    parser.add_argument("--thumb-size", type=int, default=360, help="Thumbnail size for each candidate tile.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def resolve_path(root: Path, value: str | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Pilot CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_text(path: Path, text: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_id_list(value: str) -> list[str]:
    out: list[str] = []
    for item in str(value or "").split(","):
        cid = item.strip().upper()
        if cid and cid not in out:
            out.append(cid)
    return out


def render_manifest_path(root: Path, args: argparse.Namespace, sample_id: str) -> Path:
    preferred = resolve_path(root, args.renders_root) / sample_id / "view_manifest.json"
    if preferred.exists():
        return preferred
    fallback = resolve_path(root, args.fallback_renders_root) / sample_id / "view_manifest.json"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No view_manifest.json found for sample: {sample_id}")


def image_path_from_view(root: Path, entry: dict[str, Any], base_dir: Path) -> Path:
    for key in ("dense_render_path", "render_path", "selector_path", "silhouette_path"):
        value = entry.get(key)
        if value:
            path = resolve_portable_path(root, value, base_dir)
            if path.exists():
                return path
    raise FileNotFoundError(f"No render image path found in view entry: {entry}")


def candidate_pixel_mask(index_map: np.ndarray, candidate_mask: np.ndarray) -> np.ndarray:
    valid = index_map >= 0
    safe = np.maximum(index_map, 0).astype(np.int64)
    safe = np.minimum(safe, candidate_mask.shape[0] - 1)
    return valid & candidate_mask[safe].astype(bool)


def bbox_from_mask(mask_2d: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask_2d)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def union_bbox(bboxes: list[list[int] | None], padding: int, w: int, h: int) -> list[int] | None:
    xs: list[int] = []
    ys: list[int] = []
    for box in bboxes:
        if box is None:
            continue
        xs.extend([box[0], box[2]])
        ys.extend([box[1], box[3]])
    if not xs:
        return None
    pad = max(0, int(padding))
    return [max(0, min(xs) - pad), max(0, min(ys) - pad), min(w - 1, max(xs) + pad), min(h - 1, max(ys) + pad)]


def blend_disc(base: np.ndarray, y: int, x: int, color: tuple[int, int, int], radius: int, alpha: float) -> None:
    h, w = base.shape[:2]
    r = max(0, int(radius))
    r2 = r * r
    color_arr = np.asarray(color, dtype=np.float32)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > r2:
                continue
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w:
                base[yy, xx] = (base[yy, xx].astype(np.float32) * (1.0 - alpha) + color_arr * alpha).astype(np.uint8)


def draw_candidate_points(
    image: Image.Image,
    masks_2d: list[np.ndarray],
    colors: list[tuple[int, int, int]],
    point_radius: int,
    alpha: float,
    dim: bool = True,
) -> Image.Image:
    base = np.asarray(image.convert("RGB")).copy()
    if dim:
        base = (base.astype(np.float32) * 0.42).astype(np.uint8)
    for mask_2d, color in zip(masks_2d, colors):
        ys, xs = np.where(mask_2d)
        for y, x in zip(ys.tolist(), xs.tolist()):
            blend_disc(base, y, x, color, point_radius, alpha)
    return Image.fromarray(base)


def draw_bbox_and_label(image: Image.Image, bbox: list[int] | None, label: str, color: tuple[int, int, int]) -> None:
    if bbox is None:
        return
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    text_w = len(label) * 6 + 8
    y0 = max(0, y1 - 18)
    draw.rectangle([x1, y0, x1 + text_w, y0 + 16], fill=(8, 10, 14), outline=color)
    draw.text((x1 + 4, y0 + 3), label, fill=color, font=font)


def add_caption(image: Image.Image, title: str, subtitle: str = "") -> Image.Image:
    font = ImageFont.load_default()
    h = 42 if subtitle else 26
    out = Image.new("RGB", (image.width, image.height + h), (8, 10, 14))
    out.paste(image, (0, h))
    draw = ImageDraw.Draw(out)
    draw.text((10, 7), title, fill=(245, 247, 252), font=font)
    if subtitle:
        draw.text((10, 24), subtitle, fill=(180, 190, 205), font=font)
    return out


def resize_to_square(image: Image.Image, size: int) -> Image.Image:
    resample = getattr(Image, "Resampling", Image).BICUBIC
    scale = min(size / max(1, image.width), size / max(1, image.height))
    resized = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), resample)
    canvas = Image.new("RGB", (size, size), (14, 16, 20))
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return canvas


def make_selected_context(
    raw: Image.Image,
    selected_overlay: Image.Image,
    selected_bbox: list[int] | None,
    crop_padding: int,
    selected_ids: list[str],
) -> Image.Image:
    w, h = raw.size
    if selected_bbox is None:
        selected_bbox = [0, 0, w - 1, h - 1]
    crop_box = union_bbox([selected_bbox], crop_padding, w, h) or [0, 0, w - 1, h - 1]
    crop = selected_overlay.crop((crop_box[0], crop_box[1], crop_box[2] + 1, crop_box[3] + 1))
    crop = resize_to_square(crop, h)
    panel = Image.new("RGB", (w * 2 + h, h), (14, 16, 20))
    panel.paste(raw, (0, 0))
    panel.paste(selected_overlay, (w, 0))
    panel.paste(crop, (w * 2, 0))
    draw = ImageDraw.Draw(panel)
    draw.line([(w, 0), (w, h)], fill=(88, 96, 112), width=1)
    draw.line([(w * 2, 0), (w * 2, h)], fill=(88, 96, 112), width=1)
    return add_caption(
        panel,
        f"Selected candidate review: {','.join(selected_ids) if selected_ids else '(none)'}",
        "Left: raw render | Middle: selected candidates only | Right: zoomed selected region",
    )


def make_candidate_grid(
    raw: Image.Image,
    candidate_items: list[dict[str, Any]],
    masks_2d: dict[str, np.ndarray],
    bboxes: dict[str, list[int] | None],
    selected_ids: set[str],
    point_radius: int,
    alpha: float,
    cols: int,
    thumb_size: int,
) -> Image.Image:
    cols = max(1, int(cols))
    rows = int(np.ceil(len(candidate_items) / cols))
    tile_w = int(thumb_size)
    tile_h = int(thumb_size) + 58
    sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), (8, 10, 14))
    font = ImageFont.load_default()
    for idx, item in enumerate(candidate_items):
        cid = str(item["candidate_id"])
        color = PALETTE[idx % len(PALETTE)]
        overlay = draw_candidate_points(raw, [masks_2d[cid]], [color], point_radius, alpha, dim=True)
        draw_bbox_and_label(overlay, bboxes[cid], cid, color)
        overlay = resize_to_square(overlay, tile_w)
        x = (idx % cols) * tile_w
        y = (idx // cols) * tile_h
        sheet.paste(overlay, (x, y + 58))
        draw = ImageDraw.Draw(sheet)
        status = "SELECTED" if cid in selected_ids else "candidate"
        fill = (255, 230, 130) if cid in selected_ids else (235, 238, 245)
        title = f"{cid} [{status}] {item.get('candidate_name', '')}"
        subtitle = f"{item.get('candidate_family', '')} | points={item.get('point_count', '')}"
        draw.rectangle([x, y, x + tile_w - 1, y + 57], fill=(8, 10, 14), outline=(88, 96, 112))
        draw.text((x + 8, y + 8), title[:58], fill=fill, font=font)
        draw.text((x + 8, y + 28), subtitle[:58], fill=(180, 190, 205), font=font)
    return sheet


def selected_from_rule(root: Path, args: argparse.Namespace) -> list[str]:
    rule_path = resolve_path(root, args.rule_root) / args.pilot_id / "rule_filter.json"
    if not rule_path.exists():
        return []
    data = read_json(rule_path)
    return [str(item).strip().upper() for item in data.get("accepted_candidates", []) if str(item).strip()]


def relative_path(from_dir: Path, target: Path) -> str:
    try:
        return target.relative_to(from_dir).as_posix()
    except ValueError:
        return target.as_posix()


def build_html(manifest: dict[str, Any], output_dir: Path, selected_ids: list[str]) -> str:
    title = f"v2 candidate review - {manifest['pilot_id']}"
    rows = []
    for view in manifest["views"]:
        rows.append(
            f"""
<section>
  <h2>{html.escape(view['view'])}</h2>
  <p>已选候选：<code>{html.escape(','.join(selected_ids) or '(none)')}</code></p>
  <h3>已选候选定位</h3>
  <img src="{html.escape(relative_path(output_dir, output_dir / view['selected_context_path']))}" />
  <h3>每个候选单独显示</h3>
  <img src="{html.escape(relative_path(output_dir, output_dir / view['candidate_grid_path']))}" />
</section>
"""
        )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)}</title>
  <style>
    body {{ margin: 24px; background: #0b0f15; color: #e8edf5; font-family: Arial, "Microsoft YaHei", sans-serif; }}
    h1, h2, h3 {{ font-weight: 650; }}
    section {{ margin: 28px 0 48px; padding-bottom: 24px; border-bottom: 1px solid #313948; }}
    img {{ display: block; max-width: 100%; margin: 12px 0 24px; border: 1px solid #313948; background: #090d12; }}
    code {{ color: #ffd166; }}
    .meta {{ color: #b8c2d0; line-height: 1.6; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <div class="meta">
    sample_id: <code>{html.escape(manifest['sample_id'])}</code><br/>
    object/task/executor: <code>{html.escape(manifest['object_category'])}</code> /
    <code>{html.escape(manifest['task'])}</code> /
    <code>{html.escape(manifest['executor'])}</code><br/>
    说明：这里的图用于人工审查候选区域，不是 ground truth。
  </div>
  {''.join(rows)}
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    rows = read_csv(resolve_path(root, args.pilot_csv))
    row = next((item for item in rows if item.get("pilot_id") == args.pilot_id), None)
    if row is None:
        raise ValueError(f"Pilot id not found: {args.pilot_id}")

    sample_id = row["sample_id"]
    candidate_manifest_path = resolve_path(root, args.candidate_root) / args.pilot_id / "candidate_manifest.json"
    candidate_manifest = read_json(candidate_manifest_path)
    candidate_npz_path = resolve_portable_path(root, candidate_manifest["candidate_npz"], candidate_manifest_path.parent)
    data = np.load(candidate_npz_path, allow_pickle=True)
    candidate_masks = data["candidate_masks"].astype(np.uint8)
    candidates = candidate_manifest["candidates"]
    candidate_by_id = {str(item["candidate_id"]).upper(): item for item in candidates}

    selected_ids = parse_id_list(args.selected_candidates) or selected_from_rule(root, args)
    selected_ids = [cid for cid in selected_ids if cid in candidate_by_id]

    view_manifest_path = render_manifest_path(root, args, sample_id)
    view_manifest = read_json(view_manifest_path)
    wanted_views = set(parse_id_list(args.views))
    output_dir = resolve_path(root, args.output_root) / args.pilot_id
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_views: list[dict[str, Any]] = []
    candidate_ids = [str(item["candidate_id"]).upper() for item in candidates]
    candidate_index = {cid: idx for idx, cid in enumerate(candidate_ids)}

    for entry in view_manifest.get("views", []):
        view = str(entry["view"])
        if wanted_views and view.upper() not in wanted_views and view not in wanted_views:
            continue
        image_path = image_path_from_view(root, entry, view_manifest_path.parent)
        index_path = resolve_portable_path(root, entry["point_index_path"], view_manifest_path.parent)
        raw = Image.open(image_path).convert("RGB")
        index_map = np.load(index_path)

        masks_2d: dict[str, np.ndarray] = {}
        bboxes: dict[str, list[int] | None] = {}
        for cid in candidate_ids:
            mask_2d = candidate_pixel_mask(index_map, candidate_masks[candidate_index[cid]])
            masks_2d[cid] = mask_2d
            bboxes[cid] = bbox_from_mask(mask_2d)

        selected_masks = [masks_2d[cid] for cid in selected_ids]
        selected_colors = [PALETTE[candidate_index[cid] % len(PALETTE)] for cid in selected_ids]
        selected_overlay = draw_candidate_points(
            raw,
            selected_masks,
            selected_colors,
            args.point_radius,
            max(0.0, min(1.0, float(args.alpha))),
            dim=True,
        )
        selected_bbox = union_bbox([bboxes[cid] for cid in selected_ids], 0, raw.width, raw.height)
        for cid in selected_ids:
            draw_bbox_and_label(selected_overlay, bboxes[cid], cid, PALETTE[candidate_index[cid] % len(PALETTE)])

        context = make_selected_context(raw, selected_overlay, selected_bbox, args.crop_padding, selected_ids)
        context_path = output_dir / f"{view}_selected_context.png"
        if context_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists. Use --overwrite: {context_path}")
        context.save(context_path)

        grid = make_candidate_grid(
            raw=raw,
            candidate_items=candidates,
            masks_2d=masks_2d,
            bboxes=bboxes,
            selected_ids=set(selected_ids),
            point_radius=max(1, int(args.point_radius)),
            alpha=max(0.0, min(1.0, float(args.alpha))),
            cols=args.grid_cols,
            thumb_size=args.thumb_size,
        )
        grid_path = output_dir / f"{view}_candidate_grid.png"
        if grid_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists. Use --overwrite: {grid_path}")
        grid.save(grid_path)

        manifest_views.append(
            {
                "view": view,
                "source_image_path": relative_to_dataset(root, image_path),
                "point_index_path": relative_to_dataset(root, index_path),
                "selected_context_path": context_path.name,
                "candidate_grid_path": grid_path.name,
                "selected_candidate_bboxes": {cid: bboxes[cid] for cid in selected_ids},
            }
        )

    if not manifest_views:
        raise ValueError("No views selected.")

    output_manifest = {
        "version": "v2",
        "pipeline": "v2_candidate_human_review_visualization",
        "pilot_id": args.pilot_id,
        "sample_id": sample_id,
        "object_category": row.get("object_category", candidate_manifest.get("object_category", "")),
        "task": row.get("task", candidate_manifest.get("task", "")),
        "executor": row.get("executor", candidate_manifest.get("executor", "")),
        "candidate_manifest": relative_to_dataset(root, candidate_manifest_path),
        "selected_candidates": selected_ids,
        "views": manifest_views,
        "notes": "Images are for human inspection only; v2 candidate labels require manual review before checked use.",
    }
    write_json(output_dir / "review_visualization_manifest.json", output_manifest, args.overwrite)
    html_text = build_html(output_manifest, output_dir, selected_ids)
    html_path = output_dir / "index.html"
    write_text(html_path, html_text, args.overwrite)

    print(
        json.dumps(
            {
                "pilot_id": args.pilot_id,
                "selected_candidates": selected_ids,
                "views": len(manifest_views),
                "output_dir": relative_to_dataset(root, output_dir),
                "index_html": relative_to_dataset(root, html_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
