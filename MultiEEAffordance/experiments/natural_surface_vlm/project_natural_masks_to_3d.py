#!/usr/bin/env python3
"""Project 2D masks from natural-like renders back to original 3D points."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

DATASET_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = DATASET_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from path_utils import relative_to_dataset, resolve_portable_path  # noqa: E402


EXECUTOR_ORDER = ["gripper", "suction", "hook", "dexterous_hand"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backproject 2D masks on natural-like views to 3D point masks.")
    parser.add_argument("--dataset-root", default=".", help="Dataset root directory.")
    parser.add_argument("--manifest", required=True, help="Natural render view_manifest.json path.")
    parser.add_argument(
        "--mask-root",
        required=True,
        help="Directory containing per-view 2D masks. Expected names: VIEW_mask.npy/png or VIEW.npy/png.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory relative to dataset root or absolute.")
    parser.add_argument("--executor", choices=EXECUTOR_ORDER, required=True, help="Executor channel to write.")
    parser.add_argument("--threshold", type=float, default=0.5, help="2D mask threshold. PNG threshold uses 0-1 scale.")
    parser.add_argument("--min-confidence", type=float, default=0.35, help="Minimum natural-render confidence for a pixel.")
    parser.add_argument("--min-view-votes", type=int, default=1, help="Minimum number of views voting for a 3D point.")
    parser.add_argument("--direct-only", action="store_true", help="Use only direct splat pixels, ignoring filled pixels.")
    parser.add_argument("--allow-missing-views", action="store_true", help="Skip views with no 2D mask file.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite outputs.")
    return parser.parse_args()


def resolve_path(root: Path, value: str | Path, base_dir: Path | None = None) -> Path:
    return resolve_portable_path(root, value, base_dir=base_dir)


def write_json(path: Path, data: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def find_mask(mask_root: Path, view: str) -> Path | None:
    candidates = [
        mask_root / f"{view}_mask.npy",
        mask_root / f"{view}.npy",
        mask_root / f"{view}_mask.png",
        mask_root / f"{view}.png",
        mask_root / f"{view}_mask.jpg",
        mask_root / f"{view}.jpg",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_2d_mask(path: Path, threshold: float) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        if arr.ndim == 3:
            arr = arr.max(axis=2)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D npy mask, got {arr.shape}: {path}")
        return arr.astype(np.float32) > float(threshold)

    image = Image.open(path).convert("L")
    arr = np.asarray(image).astype(np.float32) / 255.0
    return arr > float(threshold)


def load_manifest(root: Path, path_value: str) -> tuple[Path, dict[str, Any]]:
    manifest_path = resolve_path(root, path_value)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return manifest_path, json.loads(manifest_path.read_text(encoding="utf-8"))


def load_map(root: Path, entry: dict[str, Any], key: str, base_dir: Path) -> np.ndarray:
    path = resolve_path(root, entry[key], base_dir=base_dir)
    if not path.exists():
        raise FileNotFoundError(f"Required map not found: {path}")
    return np.load(path)


def project_view(
    root: Path,
    manifest_dir: Path,
    view_entry: dict[str, Any],
    mask_path: Path,
    threshold: float,
    min_confidence: float,
    direct_only: bool,
    n_points: int,
) -> dict[str, Any]:
    view = str(view_entry["view"])
    mask = load_2d_mask(mask_path, threshold)
    index_map = load_map(root, view_entry, "point_index_path", manifest_dir)
    confidence = load_map(root, view_entry, "confidence_path", manifest_dir)
    source = load_map(root, view_entry, "source_path", manifest_dir)
    if index_map.shape != mask.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match point_index shape {index_map.shape}: {mask_path}")
    if confidence.shape != mask.shape or source.shape != mask.shape:
        raise ValueError(f"confidence/source shape mismatch for view: {view}")

    valid = mask & (index_map >= 0) & (confidence >= min_confidence)
    if direct_only:
        valid &= source == 1
    ids = index_map[valid].astype(np.int64)
    if ids.size and (ids.min() < 0 or ids.max() >= n_points):
        raise ValueError(f"Point index out of range in view {view}: min={ids.min()}, max={ids.max()}, n={n_points}")

    pixel_votes = np.bincount(ids, minlength=n_points).astype(np.int32) if ids.size else np.zeros(n_points, dtype=np.int32)
    point_hits = pixel_votes > 0
    return {
        "view": view,
        "mask_path": mask_path,
        "positive_pixels": int(mask.sum()),
        "valid_projected_pixels": int(ids.size),
        "projected_points": int(point_hits.sum()),
        "pixel_votes": pixel_votes,
        "point_hits": point_hits,
    }


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    manifest_path, manifest = load_manifest(root, args.manifest)
    manifest_dir = manifest_path.parent
    mask_root = resolve_path(root, args.mask_root)
    output_dir = resolve_path(root, args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory exists. Use --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    n_points = int(manifest["num_points"])
    view_votes = np.zeros(n_points, dtype=np.int32)
    pixel_votes_total = np.zeros(n_points, dtype=np.int32)
    view_summaries: list[dict[str, Any]] = []

    for entry in manifest.get("views", []):
        view = str(entry["view"])
        mask_path = find_mask(mask_root, view)
        if mask_path is None:
            if args.allow_missing_views:
                view_summaries.append({"view": view, "status": "missing_mask"})
                continue
            raise FileNotFoundError(f"2D mask not found for view {view} in {mask_root}")
        result = project_view(
            root,
            manifest_dir,
            entry,
            mask_path,
            args.threshold,
            args.min_confidence,
            args.direct_only,
            n_points,
        )
        view_votes += result["point_hits"].astype(np.int32)
        pixel_votes_total += result["pixel_votes"].astype(np.int32)
        view_summaries.append(
            {
                "view": result["view"],
                "mask_path": relative_to_dataset(root, result["mask_path"]),
                "positive_pixels": result["positive_pixels"],
                "valid_projected_pixels": result["valid_projected_pixels"],
                "projected_points": result["projected_points"],
            }
        )

    point_mask = view_votes >= max(1, int(args.min_view_votes))
    channel = EXECUTOR_ORDER.index(args.executor)
    multi_channel_mask = np.zeros((n_points, len(EXECUTOR_ORDER)), dtype=np.uint8)
    multi_channel_mask[:, channel] = point_mask.astype(np.uint8)

    point_mask_path = output_dir / f"{manifest['sample_id']}_{args.executor}_point_mask.npy"
    point_view_votes_path = output_dir / f"{manifest['sample_id']}_{args.executor}_view_votes.npy"
    point_pixel_votes_path = output_dir / f"{manifest['sample_id']}_{args.executor}_pixel_votes.npy"
    multi_channel_path = output_dir / f"{manifest['sample_id']}_{args.executor}_multi_channel_mask.npy"
    summary_path = output_dir / "projection_summary.json"
    np.save(point_mask_path, point_mask.astype(np.uint8))
    np.save(point_view_votes_path, view_votes)
    np.save(point_pixel_votes_path, pixel_votes_total)
    np.save(multi_channel_path, multi_channel_mask)

    summary = {
        "version": "natural_surface_vlm_backprojection_v0",
        "sample_id": manifest["sample_id"],
        "executor": args.executor,
        "executor_order": EXECUTOR_ORDER,
        "manifest": relative_to_dataset(root, manifest_path),
        "mask_root": relative_to_dataset(root, mask_root),
        "num_points": n_points,
        "positive_points": int(point_mask.sum()),
        "min_confidence": float(args.min_confidence),
        "min_view_votes": int(args.min_view_votes),
        "direct_only": bool(args.direct_only),
        "point_mask_path": relative_to_dataset(root, point_mask_path),
        "point_view_votes_path": relative_to_dataset(root, point_view_votes_path),
        "point_pixel_votes_path": relative_to_dataset(root, point_pixel_votes_path),
        "multi_channel_mask_path": relative_to_dataset(root, multi_channel_path),
        "views": view_summaries,
        "notes": "Backprojected masks are candidate labels. They require rule checks and human review before GT use.",
    }
    write_json(summary_path, summary, args.overwrite)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
