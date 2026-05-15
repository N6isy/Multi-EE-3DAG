#!/usr/bin/env python3
"""Render VLM-friendly multi-view images while preserving point-index maps.

This script creates two synchronized render products:
  1. Dense visual images for VLM / open-vocabulary grounding models.
  2. Sparse point-index maps for projecting 2D masks back to real 3D points.

The dense visual images are allowed to be easier for models to read, but the
final 3D supervision must still be produced through the point-index maps.
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

from path_utils import relative_to_dataset
from render_multiview import (
    DEFAULT_VIEWS,
    depth_to_rgb,
    find_sample,
    load_points,
    normalize_points,
    project_view,
    read_jsonl,
    write_png_rgb,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render VLM-friendly views for semantic-part pipeline.")
    parser.add_argument("--dataset-root", default=".", help="Dataset root directory.")
    parser.add_argument(
        "--pilot-csv",
        default="processed/metadata/vlm_pilot_samples_v0_1.csv",
        help="Pilot CSV relative to dataset root.",
    )
    parser.add_argument(
        "--samples",
        default="processed/metadata/samples_checked_v0_1.jsonl",
        help="Checked samples JSONL relative to dataset root.",
    )
    parser.add_argument(
        "--output-root",
        default="processed/vlm_semantic_part/renders",
        help="Output render root relative to dataset root.",
    )
    parser.add_argument("--pilot-id", default=None, help="Render only one pilot id.")
    parser.add_argument("--limit", type=int, default=None, help="Limit pilot rows before deduplication.")
    parser.add_argument("--views", default=",".join(DEFAULT_VIEWS), help="Comma-separated view names.")
    parser.add_argument("--image-size", type=int, default=768, help="Square render size.")
    parser.add_argument("--index-point-size", type=int, default=1, help="Point radius for point-index map.")
    parser.add_argument("--visual-point-size", type=int, default=5, help="Point radius for dense visual render.")
    parser.add_argument("--silhouette-radius", type=int, default=5, help="Dilation radius for silhouette image.")
    parser.add_argument("--crop-padding", type=int, default=96, help="Padding for upper-structure zoom crop.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing render outputs.")
    return parser.parse_args()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Pilot CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, data: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def dilate_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    if radius <= 0:
        return mask.astype(bool)
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    radius2 = radius * radius
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius2:
                continue
            src_y0 = max(0, -dy)
            src_y1 = min(h, h - dy)
            src_x0 = max(0, -dx)
            src_x1 = min(w, w - dx)
            dst_y0 = max(0, dy)
            dst_y1 = min(h, h + dy)
            dst_x0 = max(0, dx)
            dst_x1 = min(w, w + dx)
            out[dst_y0:dst_y1, dst_x0:dst_x1] |= mask[src_y0:src_y1, src_x0:src_x1]
    return out


def silhouette_rgb(index_map: np.ndarray, radius: int) -> np.ndarray:
    foreground = dilate_bool(index_map >= 0, radius)
    rgb = np.zeros((*index_map.shape, 3), dtype=np.uint8)
    rgb[..., :] = np.array([14, 16, 20], dtype=np.uint8)
    rgb[foreground] = np.array([232, 236, 243], dtype=np.uint8)
    return rgb


def upper_structure_bbox(index_map: np.ndarray, padding: int) -> list[int] | None:
    valid = index_map >= 0
    if not np.any(valid):
        return None
    row_counts = valid.sum(axis=1)
    threshold = max(6, int(row_counts.max() * 0.18))
    dense_rows = np.where(row_counts >= threshold)[0]
    if dense_rows.size == 0:
        return None
    body_top = int(dense_rows.min())
    candidate = valid[: max(1, body_top), :]
    if not np.any(candidate):
        # Fall back to the top 20 percent if no detached upper structure exists.
        candidate = valid[: max(1, int(index_map.shape[0] * 0.20)), :]
    ys, xs = np.where(candidate)
    if len(xs) == 0:
        return None
    pad = max(0, int(padding))
    h, w = index_map.shape
    return [
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(w - 1, int(xs.max()) + pad),
        min(h - 1, int(ys.max()) + pad),
    ]


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    font = ImageFont.load_default()
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle([bbox[0] - 5, bbox[1] - 4, bbox[2] + 5, bbox[3] + 4], fill=(8, 10, 14), outline=(88, 96, 112))
    draw.text((x, y), text, fill=(235, 238, 245), font=font)


def make_selector_panel(dense_rgb: np.ndarray, sil_rgb: np.ndarray, crop_bbox: list[int] | None) -> Image.Image:
    dense = Image.fromarray(dense_rgb).convert("RGB")
    sil = Image.fromarray(sil_rgb).convert("RGB")
    w, h = dense.size
    if crop_bbox is None:
        crop_bbox = [0, 0, w - 1, h - 1]
    crop = dense.crop((crop_bbox[0], crop_bbox[1], crop_bbox[2] + 1, crop_bbox[3] + 1))
    target = h
    scale = min(target / max(1, crop.width), target / max(1, crop.height))
    resample = getattr(Image, "Resampling", Image).BICUBIC
    crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))), resample)
    crop_canvas = Image.new("RGB", (target, target), (14, 16, 20))
    crop_canvas.paste(crop, ((target - crop.width) // 2, (target - crop.height) // 2))

    panel = Image.new("RGB", (w * 3, h), (14, 16, 20))
    panel.paste(dense, (0, 0))
    panel.paste(sil, (w, 0))
    panel.paste(crop_canvas, (w * 2, 0))
    draw = ImageDraw.Draw(panel)
    draw.rectangle([w, 0, w * 2 - 1, h - 1], outline=(88, 96, 112), width=1)
    draw.rectangle([w * 2, 0, w * 3 - 1, h - 1], outline=(88, 96, 112), width=1)
    draw_label(draw, (12, h - 28), "Dense point render")
    draw_label(draw, (w + 12, h - 28), "Silhouette")
    draw_label(draw, (w * 2 + 12, h - 28), "Upper/target zoom")
    return panel


def render_one_sample(
    root: Path,
    samples: list[dict[str, Any]],
    sample_id: str,
    output_root: Path,
    views: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    sample = find_sample(samples, sample_id)
    points_path = resolve_path(root, sample["point_cloud_path"])
    points_xyz = normalize_points(load_points(points_path))
    sample_dir = output_root / sample_id
    if sample_dir.exists() and any(sample_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory exists. Use --overwrite: {sample_dir}")
    sample_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "sample_id": sample_id,
        "object_id": sample.get("object_id", ""),
        "object_category": sample.get("object_category", ""),
        "task": sample.get("task", ""),
        "task_instruction": sample.get("task_instruction", ""),
        "point_cloud_path": sample.get("point_cloud_path", ""),
        "num_points": int(points_xyz.shape[0]),
        "image_size": int(args.image_size),
        "index_point_size": int(args.index_point_size),
        "visual_point_size": int(args.visual_point_size),
        "views": [],
    }

    for view in views:
        index_map, depth_map = project_view(points_xyz, view, args.image_size, max(0, args.index_point_size))
        dense_index_map, dense_depth_map = project_view(points_xyz, view, args.image_size, max(0, args.visual_point_size))
        dense_rgb = depth_to_rgb(dense_depth_map)
        sil_rgb = silhouette_rgb(dense_index_map, args.silhouette_radius)
        crop_bbox = upper_structure_bbox(index_map, args.crop_padding)
        selector = make_selector_panel(dense_rgb, sil_rgb, crop_bbox)

        point_index_path = sample_dir / f"{view}_point_index.npy"
        depth_path = sample_dir / f"{view}_depth.npy"
        dense_path = sample_dir / f"{view}_dense.png"
        silhouette_path = sample_dir / f"{view}_silhouette.png"
        selector_path = sample_dir / f"{view}_selector.png"
        np.save(point_index_path, index_map)
        np.save(depth_path, depth_map)
        write_png_rgb(dense_path, dense_rgb)
        write_png_rgb(silhouette_path, sil_rgb)
        selector.save(selector_path)

        manifest["views"].append(
            {
                "view": view,
                "point_index_path": relative_to_dataset(root, point_index_path),
                "depth_path": relative_to_dataset(root, depth_path),
                "dense_render_path": relative_to_dataset(root, dense_path),
                "silhouette_path": relative_to_dataset(root, silhouette_path),
                "selector_path": relative_to_dataset(root, selector_path),
                "zoom_crop_bbox": crop_bbox,
            }
        )

    manifest_path = sample_dir / "view_manifest.json"
    write_json(manifest_path, manifest, args.overwrite)
    return {"sample_id": sample_id, "manifest": relative_to_dataset(root, manifest_path), "views": views}


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    pilot_rows = read_csv(resolve_path(root, args.pilot_csv))
    if args.pilot_id:
        pilot_rows = [row for row in pilot_rows if row.get("pilot_id") == args.pilot_id]
    if args.limit is not None:
        pilot_rows = pilot_rows[: args.limit]
    if not pilot_rows:
        raise ValueError("No pilot rows selected.")

    selected: list[str] = []
    seen: set[str] = set()
    for row in pilot_rows:
        sample_id = row["sample_id"]
        if sample_id not in seen:
            seen.add(sample_id)
            selected.append(sample_id)

    views = [item.strip().lower() for item in args.views.split(",") if item.strip()]
    if not views:
        raise ValueError("At least one view is required.")

    samples = read_jsonl(resolve_path(root, args.samples))
    output_root = resolve_path(root, args.output_root)
    rendered = [render_one_sample(root, samples, sample_id, output_root, views, args) for sample_id in selected]
    print(json.dumps({"rendered_samples": len(rendered), "rows": rendered}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
