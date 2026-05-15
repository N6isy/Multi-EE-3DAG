#!/usr/bin/env python3
"""Project semantic-part 2D masks back to real 3D point indices."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from path_utils import relative_to_dataset, resolve_portable_path


EXECUTOR_ORDER = ["gripper", "suction", "hook", "dexterous_hand"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project grounded 2D masks to 3D point votes.")
    parser.add_argument("--dataset-root", default=".", help="Dataset root directory.")
    parser.add_argument(
        "--pilot-csv",
        default="processed/metadata/vlm_pilot_samples_v0_1.csv",
        help="Pilot CSV relative to dataset root.",
    )
    parser.add_argument(
        "--renders-root",
        default="processed/vlm_semantic_part/renders",
        help="VLM-friendly render root relative to dataset root.",
    )
    parser.add_argument(
        "--grounded-root",
        default="processed/vlm_semantic_part/grounded_2d",
        help="2D grounded mask root relative to dataset root.",
    )
    parser.add_argument(
        "--output-root",
        default="processed/vlm_semantic_part/projected_3d",
        help="Projected 3D vote root relative to dataset root.",
    )
    parser.add_argument("--pilot-id", default=None, help="Project only one pilot row.")
    parser.add_argument("--limit", type=int, default=None, help="Limit selected pilot rows.")
    parser.add_argument("--nearest-radius", type=int, default=4, help="Map dense mask pixels to nearby true point pixels.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--allow-missing", action="store_true", help="Skip missing 2D masks.")
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


def selected_rows(root: Path, args: argparse.Namespace) -> list[dict[str, str]]:
    rows = read_csv(resolve_path(root, args.pilot_csv))
    if args.pilot_id:
        rows = [row for row in rows if row.get("pilot_id") == args.pilot_id]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No pilot rows selected.")
    return rows


def project_one(root: Path, args: argparse.Namespace, row: dict[str, str]) -> dict[str, Any]:
    pilot_id = row["pilot_id"]
    sample_id = row["sample_id"]
    executor = row["executor"]
    channel = EXECUTOR_ORDER.index(executor)
    manifest_path = resolve_path(root, args.renders_root) / sample_id / "view_manifest.json"
    manifest = read_json(manifest_path)
    num_points = int(manifest["num_points"])
    votes = np.zeros((num_points, len(EXECUTOR_ORDER)), dtype=np.float32)
    visible = np.zeros((num_points,), dtype=np.float32)
    per_view: list[dict[str, Any]] = []
    missing: list[str] = []

    for entry in manifest.get("views", []):
        view = entry["view"]
        index_path = resolve_portable_path(root, entry["point_index_path"], manifest_path.parent)
        if not index_path.exists():
            raise FileNotFoundError(f"Point-index map not found: {index_path}")
        index_map = np.load(index_path)
        if index_map.shape[0] != index_map.shape[1]:
            raise ValueError(f"Expected square index map, got {index_map.shape}: {index_path}")
        mask_path = resolve_path(root, args.grounded_root) / pilot_id / f"{view}_mask.npy"
        if not mask_path.exists():
            missing.append(str(mask_path))
            if args.allow_missing:
                continue
            raise FileNotFoundError(f"Missing 2D mask: {mask_path}")
        mask_2d = np.load(mask_path) > 0
        if mask_2d.shape != index_map.shape:
            raise ValueError(f"2D mask shape {mask_2d.shape} does not match index map {index_map.shape}: {mask_path}")

        valid = index_map >= 0
        visible_ids = index_map[valid].astype(np.int64)
        visible += np.bincount(visible_ids, minlength=num_points).astype(np.float32)

        projected_mask = dilate_bool(mask_2d, args.nearest_radius)
        selected = valid & projected_mask
        positive_ids = index_map[selected].astype(np.int64)
        if positive_ids.size:
            votes[:, channel] += np.bincount(positive_ids, minlength=num_points).astype(np.float32)
        per_view.append(
            {
                "view": view,
                "mask_path": relative_to_dataset(root, mask_path),
                "positive_pixels": int(mask_2d.sum()),
                "projected_points": int(np.unique(positive_ids).size) if positive_ids.size else 0,
            }
        )

    scores = np.divide(votes, np.maximum(visible[:, None], 1.0), out=np.zeros_like(votes), where=visible[:, None] > 0)
    output_npz = resolve_path(root, args.output_root) / f"{pilot_id}_votes.npz"
    if output_npz.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {output_npz}")
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        pilot_id=pilot_id,
        sample_id=sample_id,
        executor=executor,
        executor_order=np.asarray(EXECUTOR_ORDER, dtype=object),
        votes=votes,
        visible=visible,
        scores=scores,
    )
    summary = {
        "pilot_id": pilot_id,
        "sample_id": sample_id,
        "executor": executor,
        "projected_npz": relative_to_dataset(root, output_npz),
        "nearest_radius": int(args.nearest_radius),
        "views": per_view,
        "missing": missing,
    }
    write_json(output_npz.with_suffix(".json"), summary, args.overwrite)
    return summary


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    rows = selected_rows(root, args)
    outputs = [project_one(root, args, row) for row in rows]
    print(json.dumps({"rows": len(outputs), "outputs": outputs}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
