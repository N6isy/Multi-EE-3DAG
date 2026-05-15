#!/usr/bin/env python3
"""Render colored 2D overlays for geometry hook candidates.

The overlay images are meant for VLM candidate selection. Qwen3-VL should select
candidate labels such as A/B/C from these images instead of producing fragile
pixel coordinates directly.
"""

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


COLORS = {
    "A": (255, 72, 72),
    "B": (48, 220, 120),
    "C": (255, 210, 64),
    "D": (120, 160, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render hook candidate overlay images.")
    parser.add_argument("--dataset-root", default=".", help="Dataset root directory.")
    parser.add_argument(
        "--pilot-csv",
        default="processed/metadata/vlm_pilot_samples_v0_1.csv",
        help="Pilot CSV relative to dataset root.",
    )
    parser.add_argument(
        "--renders-root",
        default="processed/vlm_pilot/renders",
        help="Render root relative to dataset root.",
    )
    parser.add_argument(
        "--candidate-root",
        default="processed/vlm_pilot/hook_candidates",
        help="Candidate root relative to dataset root.",
    )
    parser.add_argument(
        "--output-root",
        default="processed/vlm_pilot/hook_candidate_overlays",
        help="Output overlay root relative to dataset root.",
    )
    parser.add_argument("--pilot-id", default=None, help="Render only one pilot row.")
    parser.add_argument("--limit", type=int, default=None, help="Limit selected hook pilot rows.")
    parser.add_argument("--executor", default="hook", help="Executor to process. Default: hook.")
    parser.add_argument("--point-radius", type=int, default=3, help="Overlay dot radius in pixels.")
    parser.add_argument("--alpha", type=float, default=0.82, help="Overlay color opacity.")
    parser.add_argument(
        "--selector-panel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write a VLM selector panel with full overlay plus a zoomed candidate crop.",
    )
    parser.add_argument(
        "--zoom-candidate-id",
        default="A",
        help="Candidate id used to center the zoom crop. Default: A.",
    )
    parser.add_argument("--crop-padding", type=int, default=96, help="Pixel padding around the zoom candidate crop.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing overlay outputs.")
    return parser.parse_args()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
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


def blend_pixel(base: np.ndarray, y: int, x: int, color: tuple[int, int, int], alpha: float) -> None:
    if 0 <= y < base.shape[0] and 0 <= x < base.shape[1]:
        base[y, x] = (base[y, x].astype(np.float32) * (1.0 - alpha) + np.asarray(color) * alpha).astype(np.uint8)


def draw_candidate_points(
    base: np.ndarray,
    index_map: np.ndarray,
    candidate_mask: np.ndarray,
    color: tuple[int, int, int],
    radius: int,
    alpha: float,
) -> int:
    valid = index_map >= 0
    selected = valid & candidate_mask[np.maximum(index_map, 0).astype(np.int64)].astype(bool)
    ys, xs = np.where(selected)
    radius = max(0, int(radius))
    radius2 = radius * radius
    for y, x in zip(ys.tolist(), xs.tolist()):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius2:
                    blend_pixel(base, y + dy, x + dx, color, alpha)
    return int(len(ys))


def candidate_pixel_mask(index_map: np.ndarray, candidate_mask: np.ndarray) -> np.ndarray:
    valid = index_map >= 0
    safe_index = np.maximum(index_map, 0).astype(np.int64)
    return valid & candidate_mask[safe_index].astype(bool)


def draw_legend(image: Image.Image, candidate_ids: list[str], candidate_names: list[str]) -> None:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    x0, y0 = 12, 12
    row_h = 18
    width = 340
    height = 12 + row_h * len(candidate_ids)
    draw.rectangle([x0 - 6, y0 - 6, x0 + width, y0 + height], fill=(8, 10, 14), outline=(88, 96, 112))
    for idx, candidate_id in enumerate(candidate_ids):
        y = y0 + idx * row_h
        color = COLORS.get(candidate_id, (255, 255, 255))
        draw.rectangle([x0, y + 2, x0 + 12, y + 14], fill=color)
        label = f"{candidate_id}: {candidate_names[idx]}"
        draw.text((x0 + 18, y), label, fill=(235, 238, 245), font=font)


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    font = ImageFont.load_default()
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle([bbox[0] - 5, bbox[1] - 4, bbox[2] + 5, bbox[3] + 4], fill=(8, 10, 14), outline=(88, 96, 112))
    draw.text((x, y), text, fill=(235, 238, 245), font=font)


def crop_bbox_from_mask(mask: np.ndarray, padding: int) -> list[int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    pad = max(0, int(padding))
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(w - 1, int(xs.max()) + pad)
    y2 = min(h - 1, int(ys.max()) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def make_selector_panel(
    overlay: Image.Image,
    crop_source: Image.Image,
    index_map: np.ndarray,
    candidate_ids: list[str],
    candidate_masks: np.ndarray,
    zoom_candidate_id: str,
    crop_padding: int,
) -> tuple[Image.Image, list[int] | None]:
    """Create a side-by-side image: full object plus zoomed candidate region."""
    w, h = overlay.size
    zoom_id = str(zoom_candidate_id).strip().upper()
    if zoom_id in candidate_ids:
        zoom_idx = candidate_ids.index(zoom_id)
        zoom_mask = candidate_pixel_mask(index_map, candidate_masks[zoom_idx])
    else:
        zoom_mask = np.zeros(index_map.shape, dtype=bool)

    if not np.any(zoom_mask):
        union = np.zeros(index_map.shape, dtype=bool)
        for idx in range(candidate_masks.shape[0]):
            union |= candidate_pixel_mask(index_map, candidate_masks[idx])
        zoom_mask = union

    crop_box = crop_bbox_from_mask(zoom_mask, crop_padding)
    if crop_box is None:
        crop_box = [0, 0, w - 1, h - 1]

    crop = crop_source.crop((crop_box[0], crop_box[1], crop_box[2] + 1, crop_box[3] + 1))
    # Fit the crop into a square canvas while preserving aspect ratio.
    target = h
    scale = min(target / max(1, crop.width), target / max(1, crop.height))
    new_size = (max(1, int(crop.width * scale)), max(1, int(crop.height * scale)))
    resample = getattr(Image, "Resampling", Image).BICUBIC
    crop = crop.resize(new_size, resample)
    crop_canvas = Image.new("RGB", (target, target), (14, 16, 20))
    offset = ((target - crop.width) // 2, (target - crop.height) // 2)
    crop_canvas.paste(crop, offset)

    panel = Image.new("RGB", (w + target, h), (14, 16, 20))
    panel.paste(overlay, (0, 0))
    panel.paste(crop_canvas, (w, 0))
    draw = ImageDraw.Draw(panel)
    draw_label(draw, (12, h - 28), "Full object view")
    draw_label(draw, (w + 12, h - 28), f"Zoomed candidate region around {zoom_id}")
    draw.rectangle([w, 0, w + target - 1, h - 1], outline=(88, 96, 112), width=1)
    return panel, crop_box


def render_for_row(root: Path, args: argparse.Namespace, row: dict[str, str]) -> dict[str, Any]:
    pilot_id = row["pilot_id"]
    sample_id = row["sample_id"]
    manifest_path = resolve_path(root, args.renders_root) / sample_id / "view_manifest.json"
    render_manifest = read_json(manifest_path)
    candidate_manifest_path = resolve_path(root, args.candidate_root) / pilot_id / "candidate_manifest.json"
    candidate_manifest = read_json(candidate_manifest_path)
    candidate_npz_path = resolve_portable_path(root, candidate_manifest["candidate_npz"], candidate_manifest_path.parent)
    if not candidate_npz_path.exists():
        raise FileNotFoundError(f"Candidate NPZ not found: {candidate_npz_path}")

    data = np.load(candidate_npz_path, allow_pickle=True)
    candidate_ids = [str(item) for item in data["candidate_ids"].tolist()]
    candidate_names = [str(item) for item in data["candidate_names"].tolist()]
    candidate_masks = data["candidate_masks"].astype(np.uint8)
    output_dir = resolve_path(root, args.output_root) / pilot_id
    output_dir.mkdir(parents=True, exist_ok=True)

    view_outputs: list[dict[str, Any]] = []
    for entry in render_manifest.get("views", []):
        view = entry["view"]
        image_path = resolve_portable_path(root, entry["render_path"], manifest_path.parent)
        index_path = resolve_portable_path(root, entry["point_index_path"], manifest_path.parent)
        if not image_path.exists():
            raise FileNotFoundError(f"Render image not found: {image_path}")
        if not index_path.exists():
            raise FileNotFoundError(f"Point-index map not found: {index_path}")
        image = Image.open(image_path).convert("RGB")
        base = np.asarray(image).copy()
        index_map = np.load(index_path)
        positive_pixels: dict[str, int] = {}

        # Draw broad candidates first and stricter ones last so A remains visible.
        draw_order = list(range(len(candidate_ids) - 1, -1, -1))
        for idx in draw_order:
            candidate_id = candidate_ids[idx]
            positive_pixels[candidate_id] = draw_candidate_points(
                base=base,
                index_map=index_map,
                candidate_mask=candidate_masks[idx],
                color=COLORS.get(candidate_id, (255, 255, 255)),
                radius=args.point_radius,
                alpha=max(0.0, min(1.0, float(args.alpha))),
            )
        overlay = Image.fromarray(base)
        draw_legend(overlay, candidate_ids, candidate_names)
        overlay_path = output_dir / f"{view}_overlay.png"
        if overlay_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists. Use --overwrite: {overlay_path}")
        overlay.save(overlay_path)
        selector_path = overlay_path
        crop_box = None
        if args.selector_panel:
            selector, crop_box = make_selector_panel(
                overlay=overlay,
                crop_source=Image.fromarray(base),
                index_map=index_map,
                candidate_ids=candidate_ids,
                candidate_masks=candidate_masks,
                zoom_candidate_id=args.zoom_candidate_id,
                crop_padding=args.crop_padding,
            )
            selector_path = output_dir / f"{view}_selector.png"
            if selector_path.exists() and not args.overwrite:
                raise FileExistsError(f"Output exists. Use --overwrite: {selector_path}")
            selector.save(selector_path)
        view_outputs.append(
            {
                "view": view,
                "overlay_path": relative_to_dataset(root, overlay_path),
                "selector_path": relative_to_dataset(root, selector_path),
                "selector_crop_bbox": crop_box,
                "positive_pixels": positive_pixels,
            }
        )

    overlay_manifest = {
        "pilot_id": pilot_id,
        "sample_id": sample_id,
        "object_category": row.get("object_category", ""),
        "task": row.get("task", ""),
        "executor": row.get("executor", args.executor),
        "candidate_manifest": relative_to_dataset(root, candidate_manifest_path),
        "candidate_ids": candidate_ids,
        "candidate_names": candidate_names,
        "views": view_outputs,
        "notes": "Overlay colors are VLM-readable candidate labels. They are not ground truth.",
    }
    write_json(output_dir / "overlay_manifest.json", overlay_manifest, args.overwrite)
    return overlay_manifest


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    rows = read_csv(resolve_path(root, args.pilot_csv))
    rows = [row for row in rows if row.get("executor") == args.executor]
    if args.pilot_id:
        rows = [row for row in rows if row.get("pilot_id") == args.pilot_id]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No hook pilot rows selected.")

    rendered = [render_for_row(root, args, row) for row in rows]
    print(
        json.dumps(
            {
                "rendered_rows": len(rendered),
                "rows": [
                    {
                        "pilot_id": item["pilot_id"],
                        "sample_id": item["sample_id"],
                        "views": len(item["views"]),
                    }
                    for item in rendered
                ],
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
