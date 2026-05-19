#!/usr/bin/env python3
"""Render VLM-readable overlays for general v2 3D candidates."""

from __future__ import annotations

import argparse
import csv
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render v2 candidate overlay images.")
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
        "--output-root",
        default="processed/vlm_candidate_v2/candidate_overlays",
        help="Output root relative to dataset root.",
    )
    parser.add_argument("--pilot-id", default=None, help="Render only one pilot row.")
    parser.add_argument("--limit", type=int, default=None, help="Limit selected pilot rows.")
    parser.add_argument("--max-candidates", type=int, default=10, help="Maximum candidates shown to VLM.")
    parser.add_argument("--point-radius", type=int, default=4, help="Overlay point radius.")
    parser.add_argument("--alpha", type=float, default=0.78, help="Overlay opacity.")
    parser.add_argument("--selector-panel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--crop-padding", type=int, default=80, help="Padding around selected candidate crop.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
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


def blend_disc(base: np.ndarray, y: int, x: int, color: tuple[int, int, int], radius: int, alpha: float) -> None:
    radius = max(0, int(radius))
    h, w = base.shape[:2]
    r2 = radius * radius
    color_arr = np.asarray(color, dtype=np.float32)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > r2:
                continue
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w:
                base[yy, xx] = (base[yy, xx].astype(np.float32) * (1.0 - alpha) + color_arr * alpha).astype(np.uint8)


def overlay_candidates(
    image: Image.Image,
    index_map: np.ndarray,
    candidate_masks: np.ndarray,
    candidate_ids: list[str],
    point_radius: int,
    alpha: float,
) -> tuple[Image.Image, dict[str, int], dict[str, list[int] | None]]:
    base = np.asarray(image.convert("RGB")).copy()
    positive_pixels: dict[str, int] = {}
    bboxes: dict[str, list[int] | None] = {}
    for idx in range(min(candidate_masks.shape[0], len(candidate_ids)) - 1, -1, -1):
        candidate_id = candidate_ids[idx]
        color = PALETTE[idx % len(PALETTE)]
        mask_2d = candidate_pixel_mask(index_map, candidate_masks[idx])
        ys, xs = np.where(mask_2d)
        positive_pixels[candidate_id] = int(len(xs))
        if len(xs):
            bboxes[candidate_id] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        else:
            bboxes[candidate_id] = None
        for y, x in zip(ys.tolist(), xs.tolist()):
            blend_disc(base, y, x, color, point_radius, alpha)
    return Image.fromarray(base), positive_pixels, bboxes


def draw_legend(image: Image.Image, candidates: list[dict[str, Any]]) -> None:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    x0, y0 = 12, 12
    row_h = 18
    width = min(image.width - 24, 560)
    height = 10 + row_h * len(candidates)
    draw.rectangle([x0 - 6, y0 - 6, x0 + width, y0 + height], fill=(8, 10, 14), outline=(88, 96, 112))
    for idx, item in enumerate(candidates):
        y = y0 + idx * row_h
        color = PALETTE[idx % len(PALETTE)]
        draw.rectangle([x0, y + 2, x0 + 12, y + 14], fill=color)
        label = f"{item['candidate_id']}: {item['candidate_name']} ({item['candidate_family']})"
        draw.text((x0 + 18, y), label[:84], fill=(235, 238, 245), font=font)


def crop_box_from_bboxes(bboxes: dict[str, list[int] | None], padding: int, w: int, h: int) -> list[int] | None:
    xs: list[int] = []
    ys: list[int] = []
    for bbox in bboxes.values():
        if bbox is None:
            continue
        xs.extend([bbox[0], bbox[2]])
        ys.extend([bbox[1], bbox[3]])
    if not xs:
        return None
    pad = max(0, int(padding))
    return [max(0, min(xs) - pad), max(0, min(ys) - pad), min(w - 1, max(xs) + pad), min(h - 1, max(ys) + pad)]


def make_selector_panel(overlay: Image.Image, crop_box: list[int] | None) -> Image.Image:
    w, h = overlay.size
    if crop_box is None:
        crop_box = [0, 0, w - 1, h - 1]
    crop = overlay.crop((crop_box[0], crop_box[1], crop_box[2] + 1, crop_box[3] + 1))
    target = h
    scale = min(target / max(1, crop.width), target / max(1, crop.height))
    new_size = (max(1, int(crop.width * scale)), max(1, int(crop.height * scale)))
    resample = getattr(Image, "Resampling", Image).BICUBIC
    crop = crop.resize(new_size, resample)
    canvas = Image.new("RGB", (target, target), (14, 16, 20))
    canvas.paste(crop, ((target - crop.width) // 2, (target - crop.height) // 2))
    panel = Image.new("RGB", (w + target, h), (14, 16, 20))
    panel.paste(overlay, (0, 0))
    panel.paste(canvas, (w, 0))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    draw.text((12, h - 22), "Full view with candidates", fill=(235, 238, 245), font=font)
    draw.text((w + 12, h - 22), "Zoomed candidate region", fill=(235, 238, 245), font=font)
    draw.line([(w, 0), (w, h)], fill=(88, 96, 112), width=1)
    return panel


def render_for_row(root: Path, args: argparse.Namespace, row: dict[str, str]) -> dict[str, Any]:
    pilot_id = row["pilot_id"]
    sample_id = row["sample_id"]
    candidate_manifest_path = resolve_path(root, args.candidate_root) / pilot_id / "candidate_manifest.json"
    candidate_manifest = read_json(candidate_manifest_path)
    candidate_npz_path = resolve_portable_path(root, candidate_manifest["candidate_npz"], candidate_manifest_path.parent)
    data = np.load(candidate_npz_path, allow_pickle=True)
    candidate_masks = data["candidate_masks"].astype(np.uint8)
    candidates = candidate_manifest["candidates"][: max(1, int(args.max_candidates))]
    candidate_masks = candidate_masks[: len(candidates)]
    candidate_ids = [str(item["candidate_id"]) for item in candidates]

    view_manifest_path = render_manifest_path(root, args, sample_id)
    view_manifest = read_json(view_manifest_path)
    output_dir = resolve_path(root, args.output_root) / pilot_id
    output_dir.mkdir(parents=True, exist_ok=True)
    views: list[dict[str, Any]] = []

    for entry in view_manifest.get("views", []):
        view = str(entry["view"])
        image_path = image_path_from_view(root, entry, view_manifest_path.parent)
        index_path = resolve_portable_path(root, entry["point_index_path"], view_manifest_path.parent)
        if not index_path.exists():
            raise FileNotFoundError(f"Point index map not found: {index_path}")
        image = Image.open(image_path).convert("RGB")
        index_map = np.load(index_path)
        overlay, positive_pixels, bboxes = overlay_candidates(
            image=image,
            index_map=index_map,
            candidate_masks=candidate_masks,
            candidate_ids=candidate_ids,
            point_radius=args.point_radius,
            alpha=max(0.0, min(1.0, float(args.alpha))),
        )
        draw_legend(overlay, candidates)
        overlay_path = output_dir / f"{view}_overlay.png"
        if overlay_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists. Use --overwrite: {overlay_path}")
        overlay.save(overlay_path)
        selector_path = overlay_path
        crop_box = None
        if args.selector_panel:
            crop_box = crop_box_from_bboxes(bboxes, args.crop_padding, image.width, image.height)
            selector = make_selector_panel(overlay, crop_box)
            selector_path = output_dir / f"{view}_selector.png"
            if selector_path.exists() and not args.overwrite:
                raise FileExistsError(f"Output exists. Use --overwrite: {selector_path}")
            selector.save(selector_path)
        views.append(
            {
                "view": view,
                "source_image_path": relative_to_dataset(root, image_path),
                "point_index_path": relative_to_dataset(root, index_path),
                "overlay_path": relative_to_dataset(root, overlay_path),
                "selector_path": relative_to_dataset(root, selector_path),
                "selector_crop_bbox": crop_box,
                "positive_pixels": positive_pixels,
                "candidate_bboxes": bboxes,
            }
        )

    manifest = {
        "version": "v2",
        "pipeline": "vlm_guided_candidate_selection",
        "pilot_id": pilot_id,
        "sample_id": sample_id,
        "object_category": row.get("object_category", ""),
        "task": row.get("task", ""),
        "executor": row.get("executor", ""),
        "candidate_manifest": relative_to_dataset(root, candidate_manifest_path),
        "candidate_ids": candidate_ids,
        "candidates": candidates,
        "views": views,
        "notes": "Candidate overlays are for VLM selection and human inspection only.",
    }
    write_json(output_dir / "overlay_manifest.json", manifest, args.overwrite)
    return {"pilot_id": pilot_id, "sample_id": sample_id, "views": len(views)}


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    rows = read_csv(resolve_path(root, args.pilot_csv))
    if args.pilot_id:
        rows = [row for row in rows if row.get("pilot_id") == args.pilot_id]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No pilot rows selected.")
    outputs = [render_for_row(root, args, row) for row in rows]
    print(json.dumps({"rendered_rows": len(outputs), "rows": outputs}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
