#!/usr/bin/env python3
"""Fuse projected semantic-part votes into Multi-EE four-channel masks."""

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
    parser = argparse.ArgumentParser(description="Fuse semantic-part projected votes into candidate 3D masks.")
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
        "--projected-root",
        default="processed/vlm_semantic_part/projected_3d",
        help="Projected vote root relative to dataset root.",
    )
    parser.add_argument(
        "--part-plan-root",
        default="processed/vlm_semantic_part/part_plans",
        help="Part-plan root relative to dataset root.",
    )
    parser.add_argument(
        "--output-mask-root",
        default="processed/vlm_semantic_part/fused_masks",
        help="Output candidate mask root relative to dataset root.",
    )
    parser.add_argument(
        "--output-samples",
        default="processed/metadata/semantic_part_candidate_samples_v0_1.jsonl",
        help="Output candidate samples JSONL relative to dataset root.",
    )
    parser.add_argument(
        "--output-split-dir",
        default="splits_semantic_part_candidates",
        help="Output split directory relative to dataset root.",
    )
    parser.add_argument(
        "--summary-json",
        default="processed/metadata/semantic_part_candidate_summary_v0_1.json",
        help="Output summary JSON relative to dataset root.",
    )
    parser.add_argument("--pilot-id", default=None, help="Fuse only one pilot row.")
    parser.add_argument("--limit", type=int, default=None, help="Limit selected pilot rows.")
    parser.add_argument("--score-threshold", type=float, default=0.30, help="Minimum vote score for positive points.")
    parser.add_argument("--min-visible", type=int, default=1, help="Minimum visibility count for a point.")
    parser.add_argument("--allow-empty", action="store_true", help="Allow writing empty candidate channel.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
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


def write_json(path: Path, data: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
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


def build_for_row(root: Path, args: argparse.Namespace, row: dict[str, str], sample_by_id: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    pilot_id = row["pilot_id"]
    sample_id = row["sample_id"]
    executor = row["executor"]
    if sample_id not in sample_by_id:
        raise KeyError(f"Pilot sample is not in checked samples: {sample_id}")
    sample = dict(sample_by_id[sample_id])
    channel = EXECUTOR_ORDER.index(executor)

    projected_path = resolve_path(root, args.projected_root) / f"{pilot_id}_votes.npz"
    if not projected_path.exists():
        raise FileNotFoundError(f"Projected votes not found: {projected_path}")
    data = np.load(projected_path, allow_pickle=True)
    scores = data["scores"]
    visible = data["visible"]
    candidate_channel = ((scores[:, channel] >= args.score_threshold) & (visible >= args.min_visible)).astype(np.uint8)
    if int(candidate_channel.sum()) == 0 and not args.allow_empty:
        raise ValueError(
            f"Semantic-part fused channel is empty for {pilot_id}. "
            "Use --allow-empty if this is expected, or inspect grounding/projection outputs."
        )

    base_mask_path = resolve_portable_path(root, sample["multi_channel_mask_path"])
    if row.get("checked_mask_path"):
        base_mask_path = resolve_portable_path(root, row["checked_mask_path"])
    if not base_mask_path.exists():
        raise FileNotFoundError(f"Base mask not found: {base_mask_path}")
    base_mask = np.load(base_mask_path)
    if base_mask.ndim != 2 or base_mask.shape[1] != len(EXECUTOR_ORDER):
        raise ValueError(f"Expected base mask shape [N,4], got {base_mask.shape}: {base_mask_path}")
    if base_mask.shape[0] != candidate_channel.shape[0]:
        raise ValueError(f"Candidate length {candidate_channel.shape[0]} does not match base mask {base_mask.shape[0]}")

    output_mask = base_mask.astype(np.uint8).copy()
    output_mask[:, channel] = candidate_channel
    output_mask_path = resolve_path(root, args.output_mask_root) / f"{sample_id}_{pilot_id}_semantic_part_candidate.npy"
    if output_mask_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output mask exists. Use --overwrite: {output_mask_path}")
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_mask_path, output_mask)

    part_plan_path = resolve_path(root, args.part_plan_root) / pilot_id / "combined_part_plan.json"
    label_source = dict(sample.get("label_source", {}))
    label_source[executor] = "mixed"
    feasibility = dict(sample.get("feasibility", {}))
    feasibility[executor] = bool(int(candidate_channel.sum()) > 0)
    negative_reason = dict(sample.get("negative_reason", {}))
    negative_reason[executor] = None if feasibility[executor] else "semantic_part_candidate_empty"

    sample["multi_channel_mask_path"] = relative_to_dataset(root, output_mask_path)
    sample["label_source"] = label_source
    sample["feasibility"] = feasibility
    sample["negative_reason"] = negative_reason
    sample["quality_flag"] = "weak"
    sample["split"] = "val"
    sample["semantic_part_update"] = {
        "pilot_id": pilot_id,
        "executor": executor,
        "source": "vlm_semantic_part_grounding_candidate",
        "provenance": ["qwen3vl_part_planner", "open_vocab_grounding", "sam2_or_box_mask", "point_index_projection"],
        "part_plan": relative_to_dataset(root, part_plan_path) if part_plan_path.exists() else "",
        "projected_votes": relative_to_dataset(root, projected_path),
        "score_threshold": float(args.score_threshold),
        "min_visible": int(args.min_visible),
        "positive_points": int(candidate_channel.sum()),
        "requires_human_review": True,
    }
    sample["notes"] = (
        str(sample.get("notes", ""))
        + " | semantic_part_candidate: VLM semantic part proposal; requires human review."
    )
    summary = {
        "pilot_id": pilot_id,
        "sample_id": sample_id,
        "executor": executor,
        "positive_points": int(candidate_channel.sum()),
        "output_mask_path": relative_to_dataset(root, output_mask_path),
    }
    return sample, summary


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    rows = selected_rows(root, args)
    checked_samples = read_jsonl(resolve_path(root, args.samples))
    sample_by_id = {row["sample_id"]: row for row in checked_samples}

    output_samples: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for row in rows:
        sample, summary = build_for_row(root, args, row, sample_by_id)
        output_samples.append(sample)
        summaries.append(summary)

    output_samples_path = resolve_path(root, args.output_samples)
    write_jsonl(output_samples_path, output_samples, args.overwrite)

    split_dir = resolve_path(root, args.output_split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ["train", "val", "test", "contrast_test"]:
        split_path = split_dir / f"{split_name}.txt"
        if split_path.exists() and not args.overwrite:
            raise FileExistsError(f"Split file exists. Use --overwrite: {split_path}")
        ids = [row["sample_id"] for row in output_samples] if split_name == "val" else []
        split_path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")

    summary = {
        "rows": len(summaries),
        "output_samples": relative_to_dataset(root, output_samples_path),
        "summaries": summaries,
        "notes": "Semantic-part masks are candidates and must be checked by human review.",
    }
    write_json(resolve_path(root, args.summary_json), summary, args.overwrite)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
