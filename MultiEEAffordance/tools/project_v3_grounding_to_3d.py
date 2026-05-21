#!/usr/bin/env python3
"""Project v3 target/reject 2D grounding seeds back to 3D point votes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from path_utils import relative_to_dataset, resolve_portable_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project v3 target/reject 2D seeds to 3D votes.")
    parser.add_argument("--dataset-root", default=".", help="Dataset root directory.")
    parser.add_argument("--pilot-csv", default="processed/metadata/vlm_pilot_samples_v0_1.csv")
    parser.add_argument("--renders-root", default="processed/vlm_semantic_part/renders")
    parser.add_argument("--grounding-root", default="processed/vlm_candidate_v3/target_reject_grounding")
    parser.add_argument("--output-root", default="processed/vlm_candidate_v3/projected_3d")
    parser.add_argument("--pilot-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--box-shrink-ratio", type=float, default=0.0, help="Optional shrink for broad boxes, 0 disables.")
    parser.add_argument("--point-radius", type=int, default=4, help="Pixel radius around positive points.")
    parser.add_argument("--box-dilate-radius", type=int, default=2, help="Pixel dilation radius for box masks.")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
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


def selected_rows(root: Path, args: argparse.Namespace) -> list[dict[str, str]]:
    rows = read_csv(resolve_path(root, args.pilot_csv))
    if args.pilot_id:
        rows = [row for row in rows if row.get("pilot_id") == args.pilot_id]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No pilot rows selected.")
    return rows


def dilate_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    if radius <= 0:
        return mask.astype(bool)
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    r2 = radius * radius
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > r2:
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


def shrink_box(box: list[int], ratio: float) -> list[int]:
    ratio = max(0.0, min(0.45, float(ratio)))
    if ratio <= 0:
        return box
    x1, y1, x2, y2 = box
    dx = int(round((x2 - x1) * ratio))
    dy = int(round((y2 - y1) * ratio))
    if x2 - x1 > 2 * dx + 1 and y2 - y1 > 2 * dy + 1:
        return [x1 + dx, y1 + dy, x2 - dx, y2 - dy]
    return box


def region_to_pixel_mask(region: dict[str, Any], shape: tuple[int, int], args: argparse.Namespace) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    box = region.get("box")
    if isinstance(box, list) and len(box) == 4:
        x1, y1, x2, y2 = shrink_box([int(v) for v in box], args.box_shrink_ratio)
        x1, x2 = max(0, x1), min(w - 1, x2)
        y1, y2 = max(0, y1), min(h - 1, y2)
        if x2 >= x1 and y2 >= y1:
            mask[y1 : y2 + 1, x1 : x2 + 1] = True
    radius = max(0, int(args.point_radius))
    for point in region.get("positive_points", []) or []:
        if not isinstance(point, list) or len(point) != 2:
            continue
        x, y = int(point[0]), int(point[1])
        x0, x1 = max(0, x - radius), min(w - 1, x + radius)
        y0, y1 = max(0, y - radius), min(h - 1, y + radius)
        mask[y0 : y1 + 1, x0 : x1 + 1] = True
    return dilate_bool(mask, args.box_dilate_radius)


def add_region_votes(
    votes: np.ndarray,
    index_map: np.ndarray,
    region: dict[str, Any],
    args: argparse.Namespace,
) -> int:
    pixel_mask = region_to_pixel_mask(region, index_map.shape, args)
    selected = (index_map >= 0) & pixel_mask
    ids = index_map[selected].astype(np.int64)
    if ids.size:
        votes += np.bincount(ids, minlength=votes.shape[0]).astype(np.float32)
    return int(np.unique(ids).size) if ids.size else 0


def project_one(root: Path, args: argparse.Namespace, row: dict[str, str]) -> dict[str, Any]:
    pilot_id = row["pilot_id"]
    sample_id = row["sample_id"]
    render_manifest_path = resolve_path(root, args.renders_root) / sample_id / "view_manifest.json"
    render_manifest = read_json(render_manifest_path)
    num_points = int(render_manifest["num_points"])
    target_votes = np.zeros((num_points,), dtype=np.float32)
    reject_votes = np.zeros((num_points,), dtype=np.float32)
    visible = np.zeros((num_points,), dtype=np.float32)
    per_view: list[dict[str, Any]] = []
    missing: list[str] = []

    for entry in render_manifest.get("views", []):
        view = str(entry["view"])
        index_path = resolve_portable_path(root, entry["point_index_path"], render_manifest_path.parent)
        if not index_path.exists():
            raise FileNotFoundError(f"Point-index map not found: {index_path}")
        index_map = np.load(index_path)
        valid = index_map >= 0
        visible_ids = index_map[valid].astype(np.int64)
        if visible_ids.size:
            visible += np.bincount(visible_ids, minlength=num_points).astype(np.float32)
        grounding_path = resolve_path(root, args.grounding_root) / pilot_id / f"{view}_target_reject_grounding.json"
        if not grounding_path.exists():
            missing.append(str(grounding_path))
            if args.allow_missing:
                continue
            raise FileNotFoundError(f"Missing grounding JSON: {grounding_path}")
        grounding = read_json(grounding_path)
        target_points = 0
        reject_points = 0
        for region in grounding.get("target_regions", []):
            target_points += add_region_votes(target_votes, index_map, region, args)
        for region in grounding.get("reject_regions", []):
            reject_points += add_region_votes(reject_votes, index_map, region, args)
        per_view.append(
            {
                "view": view,
                "grounding_path": relative_to_dataset(root, grounding_path),
                "target_regions": len(grounding.get("target_regions", [])),
                "reject_regions": len(grounding.get("reject_regions", [])),
                "projected_target_points": int(target_points),
                "projected_reject_points": int(reject_points),
            }
        )

    target_scores = np.divide(target_votes, np.maximum(visible, 1.0), out=np.zeros_like(target_votes), where=visible > 0)
    reject_scores = np.divide(reject_votes, np.maximum(visible, 1.0), out=np.zeros_like(reject_votes), where=visible > 0)
    output_npz = resolve_path(root, args.output_root) / f"{pilot_id}_target_reject_votes.npz"
    if output_npz.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {output_npz}")
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        pilot_id=pilot_id,
        sample_id=sample_id,
        target_votes=target_votes,
        reject_votes=reject_votes,
        visible=visible,
        target_scores=target_scores,
        reject_scores=reject_scores,
    )
    summary = {
        "version": "v3",
        "pipeline": "target_reject_projection_to_3d",
        "pilot_id": pilot_id,
        "sample_id": sample_id,
        "executor": row.get("executor", ""),
        "projected_npz": relative_to_dataset(root, output_npz),
        "views": per_view,
        "missing": missing,
        "target_positive_points_any_vote": int((target_votes > 0).sum()),
        "reject_veto_points_any_vote": int((reject_votes > 0).sum()),
        "notes": "Target votes seed candidate growth; reject votes are hard-veto evidence.",
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
