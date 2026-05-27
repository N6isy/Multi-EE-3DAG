#!/usr/bin/env python3
"""Render simple orthographic multi-view images from a point cloud sample.

This script is intentionally lightweight. It does not call any VLM. It only
creates view images and point-index maps so a later 2D mask can be projected
back to 3D.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from path_utils import relative_to_dataset


# DEFAULT_VIEWS = [
#     "yaw000_elev20",
#     "yaw045_elev20",
#     "yaw090_elev20",
#     "yaw135_elev20",
#     "yaw180_elev20",
#     "yaw225_elev20",
#     "yaw270_elev20",
#     "yaw315_elev20",
# ]

DEFAULT_VIEWS = [
    "yaw000_elev20",
    "yaw090_elev20",
    "yaw180_elev20",
    "yaw270_elev20",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render multi-view point cloud images.")
    parser.add_argument("--dataset-root", default=".", help="Dataset root directory.")
    parser.add_argument(
        "--samples",
        default="processed/metadata/samples_checked_v0_1.jsonl",
        help="Samples JSONL path relative to dataset root.",
    )
    parser.add_argument("--sample-id", required=True, help="Sample id to render.")
    parser.add_argument(
        "--output-dir",
        default="processed/vlm_pilot/renders",
        help="Output directory relative to dataset root.",
    )
    parser.add_argument("--image-size", type=int, default=768, help="Square render size in pixels.")
    parser.add_argument(
        "--views",
        default=",".join(DEFAULT_VIEWS),
        help="Comma-separated view names. Supports front/back/left/right/top/bottom/iso and yawDDD_elevDD.",
    )
    parser.add_argument("--point-size", type=int, default=1, help="Point-index raster radius in pixels.")
    parser.add_argument(
        "--visual-point-size",
        type=int,
        default=4,
        help="Visual render raster radius in pixels. This can be larger than --point-size for VLM readability.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing sample render dir.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Samples file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def find_sample(samples: list[dict[str, Any]], sample_id: str) -> dict[str, Any]:
    for sample in samples:
        if sample.get("sample_id", sample.get("object_id")) == sample_id:
            return sample
    raise KeyError(f"Sample id not found: {sample_id}")


def load_points(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {path}")
    points = np.load(path)
    if points.ndim != 2 or points.shape[1] not in (3, 6):
        raise ValueError(f"Expected point cloud shape [N,3] or [N,6], got {points.shape}: {path}")
    if points.shape[0] == 0:
        raise ValueError(f"Point cloud is empty: {path}")
    return points[:, :3].astype(np.float32)


def rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def view_rotation(name: str) -> np.ndarray:
    name = name.lower()
    match = re.fullmatch(r"yaw(-?\d+)_elev(-?\d+)", name)
    if match:
        yaw = math.radians(float(match.group(1)))
        elev = math.radians(float(match.group(2)))
        return rot_x(-elev) @ rot_y(yaw)
    if name == "front":
        return np.eye(3, dtype=np.float32)
    if name == "back":
        return rot_y(math.pi)
    if name == "left":
        return rot_y(-math.pi / 2)
    if name == "right":
        return rot_y(math.pi / 2)
    if name == "top":
        return rot_x(-math.pi / 2)
    if name == "bottom":
        return rot_x(math.pi / 2)
    if name == "iso":
        return rot_z(math.radians(35)) @ rot_x(math.radians(-35)) @ rot_y(math.radians(35))
    raise ValueError(f"Unsupported view name: {name}")


def normalize_points(points_xyz: np.ndarray) -> np.ndarray:
    center = (points_xyz.min(axis=0) + points_xyz.max(axis=0)) / 2.0
    shifted = points_xyz - center
    scale = np.linalg.norm(shifted, axis=1).max()
    if scale <= 0:
        raise ValueError("Cannot normalize degenerate point cloud.")
    return shifted / scale


def splat_pixel(
    index_map: np.ndarray,
    depth_map: np.ndarray,
    point_index: int,
    px: int,
    py: int,
    depth: float,
    radius: int,
) -> None:
    h, w = index_map.shape
    y0, y1 = max(0, py - radius), min(h - 1, py + radius)
    x0, x1 = max(0, px - radius), min(w - 1, px + radius)
    radius2 = radius * radius
    for yy in range(y0, y1 + 1):
        for xx in range(x0, x1 + 1):
            if radius > 0 and (yy - py) * (yy - py) + (xx - px) * (xx - px) > radius2:
                continue
            if depth > depth_map[yy, xx]:
                depth_map[yy, xx] = depth
                index_map[yy, xx] = point_index


def project_view(points_xyz: np.ndarray, view: str, image_size: int, point_radius: int) -> tuple[np.ndarray, np.ndarray]:
    rotated = points_xyz @ view_rotation(view).T
    xy = rotated[:, :2]
    depth = rotated[:, 2]

    padding = 0.08
    xy_min = xy.min(axis=0)
    xy_max = xy.max(axis=0)
    span = np.maximum(xy_max - xy_min, 1e-6)
    xy_norm = (xy - xy_min) / span
    xy_norm = xy_norm * (1 - 2 * padding) + padding

    px = np.clip((xy_norm[:, 0] * (image_size - 1)).round().astype(np.int32), 0, image_size - 1)
    py = np.clip(((1.0 - xy_norm[:, 1]) * (image_size - 1)).round().astype(np.int32), 0, image_size - 1)

    index_map = np.full((image_size, image_size), -1, dtype=np.int64)
    depth_map = np.full((image_size, image_size), -np.inf, dtype=np.float32)
    order = np.argsort(depth)
    for idx in order:
        splat_pixel(index_map, depth_map, int(idx), int(px[idx]), int(py[idx]), float(depth[idx]), point_radius)
    depth_map[index_map < 0] = np.nan
    return index_map, depth_map


def write_png_rgb(output_path: Path, rgb: np.ndarray) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGB image, got {rgb.shape} {rgb.dtype}")

    height, width, _ = rgb.shape

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=6))
        + chunk(b"IEND", b"")
    )
    output_path.write_bytes(payload)


def depth_to_rgb(depth_map: np.ndarray) -> np.ndarray:
    finite = np.isfinite(depth_map)
    rgb = np.zeros((*depth_map.shape, 3), dtype=np.uint8)
    rgb[..., :] = np.array([14, 16, 20], dtype=np.uint8)
    if not np.any(finite):
        return rgb
    values = depth_map[finite]
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-6:
        scaled = np.full(values.shape, 180, dtype=np.uint8)
    else:
        scaled = ((values - lo) / (hi - lo) * 205 + 50).astype(np.uint8)
    rgb_finite = np.zeros((values.shape[0], 3), dtype=np.uint8)
    rgb_finite[:, 0] = scaled // 3
    rgb_finite[:, 1] = scaled
    rgb_finite[:, 2] = 255 - scaled // 4
    rgb[finite] = rgb_finite
    return rgb


def save_png(depth_map: np.ndarray, output_path: Path) -> bool:
    # Keep PNG size exactly equal to depth_map/index_map size. This is important
    # because VLM 2D masks are projected back by pixel coordinates.
    write_png_rgb(output_path, depth_to_rgb(depth_map))
    return True


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    samples_path = resolve_path(root, args.samples)
    output_root = resolve_path(root, args.output_dir)
    views = [item.strip().lower() for item in args.views.split(",") if item.strip()]
    if not views:
        raise ValueError("At least one view is required.")

    samples = read_jsonl(samples_path)
    sample = find_sample(samples, args.sample_id)
    points_path = resolve_path(root, sample["point_cloud_path"])
    points_xyz = normalize_points(load_points(points_path))

    sample_dir = output_root / args.sample_id
    if sample_dir.exists() and any(sample_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory already exists. Use --overwrite: {sample_dir}")
    sample_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "sample_id": args.sample_id,
        "object_id": sample.get("object_id", ""),
        "object_category": sample.get("object_category", ""),
        "task": sample.get("task", ""),
        "task_instruction": sample.get("task_instruction", ""),
        "point_cloud_path": relative_to_dataset(root, points_path),
        "num_points": int(points_xyz.shape[0]),
        "image_size": args.image_size,
        "point_size": int(args.point_size),
        "visual_point_size": int(args.visual_point_size),
        "views": [],
    }

    for view in views:
        index_map, depth_map = project_view(points_xyz, view, args.image_size, max(0, args.point_size))
        _, visual_depth_map = project_view(points_xyz, view, args.image_size, max(0, args.visual_point_size))
        index_path = sample_dir / f"{view}_point_index.npy"
        depth_path = sample_dir / f"{view}_depth.npy"
        visual_depth_path = sample_dir / f"{view}_visual_depth.npy"
        png_path = sample_dir / f"{view}_render.png"
        np.save(index_path, index_map)
        np.save(depth_path, depth_map)
        np.save(visual_depth_path, visual_depth_map)
        png_written = save_png(visual_depth_map, png_path)
        manifest["views"].append(
            {
                "view": view,
                "point_index_path": relative_to_dataset(root, index_path),
                "depth_path": relative_to_dataset(root, depth_path),
                "visual_depth_path": relative_to_dataset(root, visual_depth_path),
                "render_path": relative_to_dataset(root, png_path) if png_written else "",
            }
        )

    manifest_path = sample_dir / "view_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps({"sample_id": args.sample_id, "manifest": str(manifest_path), "views": views}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
