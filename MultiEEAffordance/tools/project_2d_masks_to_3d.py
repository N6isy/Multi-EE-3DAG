#!/usr/bin/env python3
"""Project 2D view masks back to point-level vote scores.

Expected VLM mask layout:
  mask_dir/
    front.npy
    back.npy
    ...

Each mask can be [H,W] for one executor, or [H,W,4] for all executors.
PNG support is optional and requires imageio or PIL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


EXECUTOR_ORDER = ["gripper", "suction", "hook", "dexterous_hand"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project 2D masks back to 3D point votes.")
    parser.add_argument("--view-manifest", required=True, help="Path to view_manifest.json from render_multiview.py.")
    parser.add_argument("--mask-dir", required=True, help="Directory containing VLM 2D masks by view name.")
    parser.add_argument("--output", required=True, help="Output .npz path for per-point votes.")
    parser.add_argument(
        "--executor",
        choices=EXECUTOR_ORDER,
        default=None,
        help="Executor represented by 2D single-channel masks. Omit if masks are [H,W,4].",
    )
    parser.add_argument("--positive-threshold", type=float, default=0.5, help="2D mask threshold.")
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"View manifest not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_mask(path_stem: Path) -> np.ndarray:
    npy_path = path_stem.with_suffix(".npy")
    if npy_path.exists():
        return np.load(npy_path)

    png_path = path_stem.with_suffix(".png")
    if not png_path.exists():
        raise FileNotFoundError(f"Missing 2D mask for view: {npy_path} or {png_path}")

    try:
        import imageio.v3 as iio

        image = iio.imread(png_path)
    except Exception:
        try:
            from PIL import Image

            image = np.asarray(Image.open(png_path))
        except Exception as exc:
            raise RuntimeError("PNG mask loading requires imageio or PIL. Prefer .npy masks.") from exc
    if image.ndim == 3:
        image = image[..., 0]
    return image.astype(np.float32) / 255.0


def ensure_mask_channels(mask: np.ndarray, executor: str | None) -> np.ndarray:
    if mask.ndim == 2:
        if executor is None:
            raise ValueError("Single-channel 2D masks require --executor.")
        out = np.zeros((*mask.shape, len(EXECUTOR_ORDER)), dtype=np.float32)
        out[..., EXECUTOR_ORDER.index(executor)] = mask
        return out
    if mask.ndim == 3 and mask.shape[2] == len(EXECUTOR_ORDER):
        return mask.astype(np.float32)
    raise ValueError(f"Expected mask shape [H,W] or [H,W,4], got {mask.shape}")


def infer_num_points(manifest: dict[str, Any], views: list[dict[str, Any]]) -> int:
    if manifest.get("num_points"):
        return int(manifest["num_points"])
    max_index = -1
    for view in views:
        index_map = np.load(view["point_index_path"])
        if index_map.size:
            max_index = max(max_index, int(index_map.max()))
    if max_index < 0:
        raise ValueError("No visible point index found in view maps.")
    return max_index + 1


def main() -> int:
    args = parse_args()
    manifest = load_manifest(Path(args.view_manifest))
    views = manifest.get("views", [])
    if not views:
        raise ValueError("Manifest has no views.")

    num_points = infer_num_points(manifest, views)
    votes = np.zeros((num_points, len(EXECUTOR_ORDER)), dtype=np.float32)
    visible = np.zeros((num_points,), dtype=np.float32)

    mask_dir = Path(args.mask_dir)
    for view in views:
        name = view["view"]
        index_map = np.load(view["point_index_path"])
        mask = ensure_mask_channels(load_mask(mask_dir / name), args.executor)
        if mask.shape[:2] != index_map.shape:
            raise ValueError(f"Mask shape {mask.shape[:2]} does not match index map {index_map.shape}: {name}")

        valid = index_map >= 0
        point_ids = index_map[valid]
        visible_counts = np.bincount(point_ids, minlength=num_points).astype(np.float32)
        visible += visible_counts

        positive = (mask[valid] >= args.positive_threshold).astype(np.float32)
        for channel in range(len(EXECUTOR_ORDER)):
            votes[:, channel] += np.bincount(point_ids, weights=positive[:, channel], minlength=num_points)

    scores = np.divide(votes, np.maximum(visible[:, None], 1.0), out=np.zeros_like(votes), where=visible[:, None] > 0)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        sample_id=manifest.get("sample_id", ""),
        executor_order=np.array(EXECUTOR_ORDER),
        votes=votes,
        visible=visible,
        scores=scores,
    )
    print(json.dumps({"sample_id": manifest.get("sample_id", ""), "output": str(output_path), "num_points": num_points}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
