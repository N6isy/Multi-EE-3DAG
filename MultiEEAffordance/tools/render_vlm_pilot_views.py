#!/usr/bin/env python3
"""Render multi-view images for VLM pilot rows.

This is a small batch wrapper around render_multiview.py. It reads the pilot CSV,
deduplicates object-task sample ids, and writes render PNGs plus point-index maps
for every selected sample.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from path_utils import relative_to_dataset
from render_multiview import DEFAULT_VIEWS, find_sample, load_points, normalize_points, project_view, read_jsonl, save_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render VLM pilot sample views.")
    parser.add_argument("--dataset-root", default=".", help="Dataset root.")
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
        "--output-dir",
        default="processed/vlm_pilot/renders",
        help="Output render root relative to dataset root.",
    )
    parser.add_argument("--pilot-id", default=None, help="Render only one pilot id.")
    parser.add_argument("--limit", type=int, default=None, help="Limit pilot rows before deduplication.")
    parser.add_argument("--image-size", type=int, default=512, help="Square render size.")
    parser.add_argument("--point-size", type=int, default=4, help="Raster point radius.")
    parser.add_argument("--views", default=",".join(DEFAULT_VIEWS), help="Comma-separated view names.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing sample render dirs.")
    return parser.parse_args()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Pilot CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def render_sample(
    root: Path,
    samples: list[dict[str, Any]],
    sample_id: str,
    output_root: Path,
    views: list[str],
    image_size: int,
    point_size: int,
    overwrite: bool,
) -> dict[str, Any]:
    sample = find_sample(samples, sample_id)
    points_path = resolve_path(root, sample["point_cloud_path"])
    points_xyz = normalize_points(load_points(points_path))

    sample_dir = output_root / sample_id
    if sample_dir.exists() and any(sample_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory already exists. Use --overwrite: {sample_dir}")
    sample_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "sample_id": sample_id,
        "object_id": sample.get("object_id", ""),
        "object_category": sample.get("object_category", ""),
        "task": sample.get("task", ""),
        "task_instruction": sample.get("task_instruction", ""),
        "point_cloud_path": relative_to_dataset(root, points_path),
        "num_points": int(points_xyz.shape[0]),
        "image_size": int(image_size),
        "point_size": int(point_size),
        "views": [],
    }

    for view in views:
        index_map, depth_map = project_view(points_xyz, view, image_size, max(0, point_size))
        index_path = sample_dir / f"{view}_point_index.npy"
        depth_path = sample_dir / f"{view}_depth.npy"
        png_path = sample_dir / f"{view}_render.png"
        np.save(index_path, index_map)
        np.save(depth_path, depth_map)
        save_png(depth_map, png_path)
        manifest["views"].append(
            {
                "view": view,
                "point_index_path": relative_to_dataset(root, index_path),
                "depth_path": relative_to_dataset(root, depth_path),
                "render_path": relative_to_dataset(root, png_path),
            }
        )

    manifest_path = sample_dir / "view_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return {"sample_id": sample_id, "manifest": str(manifest_path), "views": views}


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

    selected: list[str] = []
    seen: set[str] = set()
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id not in seen:
            seen.add(sample_id)
            selected.append(sample_id)

    samples = read_jsonl(resolve_path(root, args.samples))
    output_root = resolve_path(root, args.output_dir)
    views = [item.strip().lower() for item in args.views.split(",") if item.strip()]
    if not views:
        raise ValueError("At least one view is required.")

    rendered = [
        render_sample(root, samples, sample_id, output_root, views, args.image_size, args.point_size, args.overwrite)
        for sample_id in selected
    ]
    print(json.dumps({"rendered_samples": len(rendered), "rows": rendered}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
