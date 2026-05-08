#!/usr/bin/env python3
"""Build fused 3D candidate masks from VLM pilot 2D masks.

This is a batch wrapper for the pilot stage. It consumes per-view masks created
by run_openai_vlm_pilot.py, projects them back to 3D, fuses the target executor
channel, and writes a small samples JSONL that can be opened by serve_review_app.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from path_utils import resolve_portable_path


EXECUTOR_ORDER = ["gripper", "suction", "hook", "dexterous_hand"]
VIEW_ORDER = ["front", "back", "left", "right", "top", "iso"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build VLM pilot fused candidate masks.")
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
        "--renders-root",
        default="processed/vlm_pilot/renders",
        help="Render root relative to dataset root.",
    )
    parser.add_argument(
        "--vlm-mask-root",
        default="processed/vlm_pilot/vlm_2d_masks",
        help="2D VLM mask root relative to dataset root.",
    )
    parser.add_argument(
        "--projected-root",
        default="processed/vlm_pilot/projected",
        help="Projected vote output root relative to dataset root.",
    )
    parser.add_argument(
        "--fused-mask-root",
        default="processed/vlm_pilot/fused_masks",
        help="Fused candidate mask root relative to dataset root.",
    )
    parser.add_argument(
        "--output-samples",
        default="processed/metadata/vlm_pilot_candidate_samples_v0_1.jsonl",
        help="Output candidate samples JSONL relative to dataset root.",
    )
    parser.add_argument(
        "--output-split-dir",
        default="splits_vlm_pilot_candidates",
        help="Output split directory relative to dataset root.",
    )
    parser.add_argument(
        "--summary-json",
        default="processed/metadata/vlm_pilot_candidate_summary_v0_1.json",
        help="Output summary JSON relative to dataset root.",
    )
    parser.add_argument("--score-threshold", type=float, default=0.45, help="Vote score threshold.")
    parser.add_argument("--min-visible", type=int, default=1, help="Minimum visible pixels per point.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--allow-missing", action="store_true", help="Skip pilot rows whose VLM 2D masks are missing.")
    return parser.parse_args()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Samples JSONL not found: {path}")
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


def write_jsonl(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Pilot CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"View manifest not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_2d_mask(mask_dir: Path, view: str, shape: tuple[int, int]) -> np.ndarray:
    npy_path = mask_dir / f"{view}.npy"
    if not npy_path.exists():
        raise FileNotFoundError(f"Missing VLM 2D mask: {npy_path}")
    mask = np.load(npy_path)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask [H,W], got {mask.shape}: {npy_path}")
    if mask.shape != shape:
        raise ValueError(f"2D mask shape {mask.shape} does not match index map {shape}: {npy_path}")
    return (mask > 0).astype(np.uint8)


def project_masks(
    manifest: dict[str, Any],
    manifest_dir: Path,
    dataset_root: Path,
    mask_dir: Path,
    executor: str,
    output_npz: Path,
) -> dict[str, Any]:
    num_points = int(manifest["num_points"])
    votes = np.zeros((num_points, len(EXECUTOR_ORDER)), dtype=np.float32)
    visible = np.zeros((num_points,), dtype=np.float32)
    channel = EXECUTOR_ORDER.index(executor)
    per_view_positive: dict[str, int] = {}

    view_entries = {entry["view"]: entry for entry in manifest["views"]}
    for view in VIEW_ORDER:
        entry = view_entries[view]
        index_path = resolve_portable_path(dataset_root, entry["point_index_path"], manifest_dir)
        index_map = np.load(index_path)
        mask_2d = load_2d_mask(mask_dir, view, index_map.shape)
        valid = index_map >= 0
        point_ids = index_map[valid]
        visible += np.bincount(point_ids, minlength=num_points).astype(np.float32)
        positive = mask_2d[valid].astype(np.float32)
        votes[:, channel] += np.bincount(point_ids, weights=positive, minlength=num_points)
        per_view_positive[view] = int(mask_2d.sum())

    scores = np.divide(votes, np.maximum(visible[:, None], 1.0), out=np.zeros_like(votes), where=visible[:, None] > 0)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        sample_id=manifest.get("sample_id", ""),
        executor_order=np.array(EXECUTOR_ORDER),
        votes=votes,
        visible=visible,
        scores=scores,
    )
    return {"projected_npz": str(output_npz), "per_view_positive_pixels": per_view_positive}


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    pilot_rows = read_csv(resolve_path(root, args.pilot_csv))
    checked_samples = read_jsonl(resolve_path(root, args.samples))
    sample_by_id = {row["sample_id"]: row for row in checked_samples}

    renders_root = resolve_path(root, args.renders_root)
    vlm_mask_root = resolve_path(root, args.vlm_mask_root)
    projected_root = resolve_path(root, args.projected_root)
    fused_mask_root = resolve_path(root, args.fused_mask_root)
    output_samples_path = resolve_path(root, args.output_samples)
    split_dir = resolve_path(root, args.output_split_dir)
    summary_path = resolve_path(root, args.summary_json)

    masks_by_sample: dict[str, np.ndarray] = {}
    updates_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_rows: list[dict[str, str]] = []
    applied_rows: list[dict[str, Any]] = []

    for row in pilot_rows:
        sample_id = row["sample_id"]
        executor = row["executor"]
        pilot_id = row["pilot_id"]
        if sample_id not in sample_by_id:
            raise KeyError(f"Pilot sample is not in checked samples: {sample_id}")

        mask_dir = vlm_mask_root / sample_id / executor
        required = [mask_dir / f"{view}.npy" for view in VIEW_ORDER]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            missing_rows.append({"pilot_id": pilot_id, "sample_id": sample_id, "executor": executor, "missing": ";".join(missing)})
            if args.allow_missing:
                continue
            raise FileNotFoundError(f"Missing VLM masks for {pilot_id}: {missing[0]}")

        if sample_id not in masks_by_sample:
            checked_mask_path = resolve_path(root, sample_by_id[sample_id]["multi_channel_mask_path"])
            checked_mask = np.load(checked_mask_path)
            if checked_mask.ndim != 2 or checked_mask.shape[1] != len(EXECUTOR_ORDER):
                raise ValueError(f"Invalid checked mask shape {checked_mask.shape}: {checked_mask_path}")
            masks_by_sample[sample_id] = checked_mask.astype(np.uint8).copy()

        manifest_path = renders_root / sample_id / "view_manifest.json"
        manifest = load_manifest(manifest_path)
        projected_npz = projected_root / f"{pilot_id}_{executor}_votes.npz"
        projection_stats = project_masks(manifest, manifest_path.parent, root, mask_dir, executor, projected_npz)
        data = np.load(projected_npz, allow_pickle=True)
        scores = data["scores"]
        visible = data["visible"]
        channel = EXECUTOR_ORDER.index(executor)
        fused_channel = ((scores[:, channel] >= args.score_threshold) & (visible >= args.min_visible)).astype(np.uint8)
        if fused_channel.shape[0] != masks_by_sample[sample_id].shape[0]:
            raise ValueError(f"Fused channel length mismatch for {sample_id}")
        masks_by_sample[sample_id][:, channel] = fused_channel

        update = {
            "pilot_id": pilot_id,
            "executor": executor,
            "issue_type": row.get("issue_type", ""),
            "decision": row.get("decision", ""),
            "positive_points": int(fused_channel.sum()),
            **projection_stats,
        }
        updates_by_sample[sample_id].append(update)
        applied_rows.append({"sample_id": sample_id, **update})

    candidate_samples: list[dict[str, Any]] = []
    for sample_id, mask in masks_by_sample.items():
        output_mask_path = fused_mask_root / f"{sample_id}.npy"
        if output_mask_path.exists() and not args.overwrite:
            raise FileExistsError(f"Candidate mask exists. Use --overwrite: {output_mask_path}")
        output_mask_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_mask_path, mask.astype(np.uint8))

        sample = dict(sample_by_id[sample_id])
        sample["multi_channel_mask_path"] = str(output_mask_path.relative_to(root)).replace("\\", "/")
        sample["quality_flag"] = "weak"
        sample["label_source"] = {executor: "vlm_pilot_candidate" for executor in EXECUTOR_ORDER}
        sample["vlm_pilot_updates"] = updates_by_sample[sample_id]
        candidate_samples.append(sample)

    write_jsonl(output_samples_path, candidate_samples, args.overwrite)
    split_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ["train", "val", "test", "contrast_test"]:
        split_path = split_dir / f"{split_name}.txt"
        if split_path.exists() and not args.overwrite:
            raise FileExistsError(f"Split file exists. Use --overwrite: {split_path}")
        ids = [sample["sample_id"] for sample in candidate_samples] if split_name == "val" else []
        split_path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")

    summary = {
        "candidate_samples": len(candidate_samples),
        "pilot_rows_total": len(pilot_rows),
        "pilot_rows_applied": len(applied_rows),
        "pilot_rows_missing": len(missing_rows),
        "score_threshold": args.score_threshold,
        "min_visible": args.min_visible,
        "output_samples": str(output_samples_path.relative_to(root)),
        "fused_mask_root": str(fused_mask_root.relative_to(root)),
        "applied_rows": applied_rows,
        "missing_rows": missing_rows,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"Summary exists. Use --overwrite: {summary_path}")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
