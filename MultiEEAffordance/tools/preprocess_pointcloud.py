#!/usr/bin/env python3
"""Convert raw object point clouds to the dataset points.npy format.

The output is a NumPy array with shape [N, 3] or [N, 6]:
  [x, y, z] or [x, y, z, nx, ny, nz]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess raw point clouds into Multi-EE points.npy files.")
    parser.add_argument("--input", required=True, help="Input point cloud file: .npy, .npz, .xyz, .pts, .txt, or .csv.")
    parser.add_argument("--output", required=True, help="Output .npy path.")
    parser.add_argument("--npz-key", help="Array key to read when --input is .npz.")
    parser.add_argument(
        "--xyz-columns",
        default="0,1,2",
        help="Comma-separated x,y,z column indices for tabular inputs. Default: 0,1,2.",
    )
    parser.add_argument(
        "--normal-columns",
        help="Optional comma-separated nx,ny,nz column indices. If omitted, columns 3,4,5 are used when present.",
    )
    parser.add_argument("--delimiter", help="Delimiter for .txt/.csv/.xyz/.pts. Defaults to comma for .csv, whitespace otherwise.")
    parser.add_argument("--skip-rows", type=int, default=0, help="Rows to skip for text-like inputs.")
    parser.add_argument("--sample-size", type=int, help="Randomly sample this many points after filtering.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used for sampling.")
    parser.add_argument(
        "--normalize",
        default="none",
        choices=["none", "unit_sphere", "unit_bbox"],
        help="Optional coordinate normalization.",
    )
    parser.add_argument(
        "--estimate-normals",
        action="store_true",
        help="Estimate normals with open3d when input normals are unavailable.",
    )
    parser.add_argument("--normal-radius", type=float, default=0.05, help="open3d radius for normal estimation.")
    parser.add_argument("--normal-max-nn", type=int, default=30, help="open3d max_nn for normal estimation.")
    parser.add_argument("--write-info", help="Optional JSON path for preprocessing statistics.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing output file.")
    return parser.parse_args()


def error(message: str) -> None:
    raise ValueError(message)


def parse_columns(value: str, expected: int, name: str) -> list[int]:
    try:
        columns = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be comma-separated integer indices, got {value!r}") from exc
    if len(columns) != expected:
        error(f"{name} must contain exactly {expected} indices, got {columns}")
    if any(column < 0 for column in columns):
        error(f"{name} cannot contain negative indices: {columns}")
    return columns


def load_raw_array(path: Path, npz_key: str | None, delimiter: str | None, skip_rows: int) -> np.ndarray:
    if not path.exists():
        error(f"Input file does not exist: {path}")
    suffix = path.suffix.lower()

    if suffix == ".npy":
        array = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        loaded = np.load(path, allow_pickle=False)
        if npz_key:
            if npz_key not in loaded:
                error(f"NPZ key {npz_key!r} not found. Available keys: {list(loaded.files)}")
            array = loaded[npz_key]
        else:
            if len(loaded.files) != 1:
                error(f"--npz-key is required because {path} contains keys {list(loaded.files)}")
            array = loaded[loaded.files[0]]
    elif suffix in {".xyz", ".pts", ".txt", ".csv"}:
        effective_delimiter = delimiter
        if effective_delimiter is None and suffix == ".csv":
            effective_delimiter = ","
        array = np.loadtxt(path, delimiter=effective_delimiter, skiprows=skip_rows)
    else:
        error(f"Unsupported input extension: {path.suffix}")

    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        error(f"Input array must be 2D, got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        error(f"Input array must be numeric, got dtype {array.dtype}")
    return array.astype(np.float32, copy=False)


def select_columns(array: np.ndarray, columns: Iterable[int], name: str) -> np.ndarray:
    columns = list(columns)
    if max(columns) >= array.shape[1]:
        error(f"{name} columns {columns} exceed input width {array.shape[1]}")
    return array[:, columns]


def choose_normals(array: np.ndarray, normal_columns: str | None, xyz_columns: list[int]) -> np.ndarray | None:
    if normal_columns:
        return select_columns(array, parse_columns(normal_columns, 3, "--normal-columns"), "normal")
    default_normal_columns = [3, 4, 5]
    if array.shape[1] >= 6 and xyz_columns == [0, 1, 2]:
        return select_columns(array, default_normal_columns, "normal")
    return None


def remove_invalid_rows(xyz: np.ndarray, normals: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None, int]:
    valid = np.all(np.isfinite(xyz), axis=1)
    if normals is not None:
        valid = valid & np.all(np.isfinite(normals), axis=1)
    removed = int((~valid).sum())
    xyz = xyz[valid]
    if normals is not None:
        normals = normals[valid]
    if xyz.shape[0] == 0:
        error("No valid points remain after removing NaN/Inf rows")
    return xyz, normals, removed


def normalize_xyz(xyz: np.ndarray, mode: str) -> tuple[np.ndarray, dict[str, object]]:
    info: dict[str, object] = {"mode": mode}
    if mode == "none":
        return xyz, info

    if mode == "unit_sphere":
        center = xyz.mean(axis=0)
        shifted = xyz - center
        scale = float(np.linalg.norm(shifted, axis=1).max())
        if scale <= 1e-12:
            error("Cannot unit_sphere normalize a degenerate point cloud")
        info.update({"center": center.tolist(), "scale": scale})
        return shifted / scale, info

    if mode == "unit_bbox":
        min_xyz = xyz.min(axis=0)
        max_xyz = xyz.max(axis=0)
        center = (min_xyz + max_xyz) / 2.0
        scale = float((max_xyz - min_xyz).max())
        if scale <= 1e-12:
            error("Cannot unit_bbox normalize a degenerate point cloud")
        info.update({"bbox_min": min_xyz.tolist(), "bbox_max": max_xyz.tolist(), "center": center.tolist(), "scale": scale})
        return (xyz - center) / scale, info

    error(f"Unknown normalization mode: {mode}")


def sample_points(
    xyz: np.ndarray,
    normals: np.ndarray | None,
    sample_size: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    if sample_size is None:
        return xyz, normals, xyz.shape[0]
    if sample_size <= 0:
        error("--sample-size must be positive")
    before = xyz.shape[0]
    if before <= sample_size:
        return xyz, normals, before
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(before, size=sample_size, replace=False))
    xyz = xyz[indices]
    if normals is not None:
        normals = normals[indices]
    return xyz, normals, before


def normalize_normals(normals: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-8
    normalized = np.zeros_like(normals, dtype=np.float32)
    normalized[valid] = normals[valid] / lengths[valid, None]
    return normalized


def estimate_normals_open3d(xyz: np.ndarray, radius: float, max_nn: int) -> np.ndarray:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ValueError("open3d is required for --estimate-normals but is not installed") from exc

    if radius <= 0:
        error("--normal-radius must be positive")
    if max_nn <= 0:
        error("--normal-max-nn must be positive")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    pcd.normalize_normals()
    return np.asarray(pcd.normals, dtype=np.float32)


def save_info(path: Path, info: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> int:
    args = parse_args()
    try:
        input_path = Path(args.input)
        output_path = Path(args.output)
        if output_path.exists() and not args.overwrite:
            error(f"Output already exists. Use --overwrite to replace it: {output_path}")

        raw = load_raw_array(input_path, args.npz_key, args.delimiter, args.skip_rows)
        xyz_columns = parse_columns(args.xyz_columns, 3, "--xyz-columns")
        xyz = select_columns(raw, xyz_columns, "xyz")
        normals = choose_normals(raw, args.normal_columns, xyz_columns)
        xyz, normals, removed_invalid = remove_invalid_rows(xyz, normals)
        xyz, normals, points_before_sampling = sample_points(xyz, normals, args.sample_size, args.seed)
        xyz, normalization_info = normalize_xyz(xyz, args.normalize)

        if normals is None and args.estimate_normals:
            normals = estimate_normals_open3d(xyz, args.normal_radius, args.normal_max_nn)
        if normals is not None:
            normals = normalize_normals(normals)
            output = np.concatenate([xyz, normals], axis=1).astype(np.float32)
        else:
            output = xyz.astype(np.float32)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, output)

        info = {
            "input": str(input_path),
            "output": str(output_path),
            "input_shape": list(raw.shape),
            "output_shape": list(output.shape),
            "removed_invalid_points": removed_invalid,
            "points_before_sampling": points_before_sampling,
            "normal_channels": output.shape[1] == 6,
            "normal_estimated": bool(args.estimate_normals and normals is not None),
            "normalization": normalization_info,
        }
        if args.write_info:
            save_info(Path(args.write_info), info)
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
