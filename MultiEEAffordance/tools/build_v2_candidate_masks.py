#!/usr/bin/env python3
"""Build four-channel candidate masks from v2 filtered candidates."""

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
    parser = argparse.ArgumentParser(description="Build v2 candidate four-channel masks.")
    parser.add_argument("--dataset-root", default=".", help="Dataset root directory.")
    parser.add_argument(
        "--pilot-csv",
        default="processed/metadata/vlm_pilot_samples_v0_1.csv",
        help="Pilot CSV relative to dataset root.",
    )
    parser.add_argument(
        "--samples",
        default="processed/metadata/samples_checked_v0_1.jsonl",
        help="Checked sample JSONL relative to dataset root.",
    )
    parser.add_argument(
        "--candidate-root",
        default="processed/vlm_candidate_v2/3d_candidates",
        help="Candidate root relative to dataset root.",
    )
    parser.add_argument(
        "--rule-root",
        default="processed/vlm_candidate_v2/rule_filter",
        help="Rule filter root relative to dataset root.",
    )
    parser.add_argument(
        "--output-mask-root",
        default="processed/vlm_candidate_v2/fused_masks",
        help="Output mask root relative to dataset root.",
    )
    parser.add_argument(
        "--output-samples",
        default="processed/metadata/v2_candidate_samples_v0_1.jsonl",
        help="Output candidate sample JSONL relative to dataset root.",
    )
    parser.add_argument(
        "--output-split-dir",
        default="splits_v2_candidates",
        help="Output split dir relative to dataset root.",
    )
    parser.add_argument(
        "--summary-json",
        default="processed/metadata/v2_candidate_summary_v0_1.json",
        help="Output summary JSON relative to dataset root.",
    )
    parser.add_argument("--pilot-id", default=None, help="Build only one pilot row.")
    parser.add_argument("--limit", type=int, default=None, help="Limit selected pilot rows.")
    parser.add_argument("--include-uncertain", action="store_true", help="Also include uncertain candidates in masks.")
    parser.add_argument(
        "--selected-candidates",
        default="",
        help="Optional comma-separated manual override candidate ids for debugging.",
    )
    parser.add_argument("--allow-empty", action="store_true", help="Allow empty target channel.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite outputs.")
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


def parse_candidate_override(value: str) -> list[str]:
    out: list[str] = []
    for item in str(value or "").split(","):
        cid = item.strip().upper()
        if cid and cid not in out:
            out.append(cid)
    return out


def selected_ids_from_rule(rule: dict[str, Any], include_uncertain: bool) -> list[str]:
    selected = [str(item).strip().upper() for item in rule.get("accepted_candidates", [])]
    if include_uncertain:
        for item in rule.get("uncertain_candidates", []):
            cid = str(item).strip().upper()
            if cid and cid not in selected:
                selected.append(cid)
    return selected


def select_rows(root: Path, args: argparse.Namespace) -> list[dict[str, str]]:
    rows = read_csv(resolve_path(root, args.pilot_csv))
    if args.pilot_id:
        rows = [row for row in rows if row.get("pilot_id") == args.pilot_id]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No pilot rows selected.")
    return rows


def merge_sample(row: dict[str, str], sample_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sample = dict(sample_by_id.get(row.get("sample_id", ""), {}))
    for key, value in row.items():
        if value not in (None, ""):
            sample[key] = value
    return sample


def build_for_row(
    root: Path,
    args: argparse.Namespace,
    row: dict[str, str],
    sample_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    pilot_id = row["pilot_id"]
    sample_id = row["sample_id"]
    sample = merge_sample(row, sample_by_id)
    executor = row.get("executor", "")
    if executor not in EXECUTOR_ORDER:
        raise ValueError(f"Unknown executor '{executor}' for {pilot_id}")

    candidate_manifest_path = resolve_path(root, args.candidate_root) / pilot_id / "candidate_manifest.json"
    rule_path = resolve_path(root, args.rule_root) / pilot_id / "rule_filter.json"
    candidate_manifest = read_json(candidate_manifest_path)
    rule = read_json(rule_path)
    candidate_npz_path = resolve_portable_path(root, candidate_manifest["candidate_npz"], candidate_manifest_path.parent)
    data = np.load(candidate_npz_path, allow_pickle=True)
    candidate_ids = [str(item).upper() for item in data["candidate_ids"].tolist()]
    candidate_masks = data["candidate_masks"].astype(np.uint8)

    manual_ids = parse_candidate_override(args.selected_candidates)
    selected_ids = manual_ids or selected_ids_from_rule(rule, args.include_uncertain)
    selected_indices = [candidate_ids.index(cid) for cid in selected_ids if cid in candidate_ids]
    target_mask = np.zeros((candidate_masks.shape[1],), dtype=np.uint8)
    if selected_indices:
        target_mask = (candidate_masks[selected_indices].sum(axis=0) > 0).astype(np.uint8)
    if int(target_mask.sum()) == 0 and not args.allow_empty:
        raise ValueError(f"Target candidate mask is empty for {pilot_id}. Use --allow-empty if expected.")

    base_mask_value = row.get("checked_mask_path") or sample.get("multi_channel_mask_path") or row.get("source_mask_path")
    base_mask_path = resolve_portable_path(root, base_mask_value)
    if not base_mask_path.exists():
        raise FileNotFoundError(f"Base mask not found: {base_mask_path}")
    base_mask = np.load(base_mask_path)
    if base_mask.ndim != 2 or base_mask.shape[1] != len(EXECUTOR_ORDER):
        raise ValueError(f"Expected base mask shape [N,4], got {base_mask.shape}: {base_mask_path}")
    if base_mask.shape[0] != target_mask.shape[0]:
        raise ValueError(f"Candidate length {target_mask.shape[0]} does not match base mask {base_mask.shape[0]}")

    out_mask = base_mask.astype(np.uint8).copy()
    channel = EXECUTOR_ORDER.index(executor)
    out_mask[:, channel] = target_mask
    output_mask_path = resolve_path(root, args.output_mask_root) / f"{sample_id}_{pilot_id}_v2_candidate.npy"
    if output_mask_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output mask exists. Use --overwrite: {output_mask_path}")
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_mask_path, out_mask)

    feasibility = dict(sample.get("feasibility", {}))
    label_source = dict(sample.get("label_source", {}))
    negative_reason = dict(sample.get("negative_reason", {}))
    feasibility[executor] = bool(int(target_mask.sum()) > 0)
    # Keep the existing metadata schema valid; detailed provenance is stored in
    # v2_candidate_update below.
    label_source[executor] = "mixed"
    negative_reason[executor] = None if feasibility[executor] else f"no_selected_{executor}_candidate"

    sample["multi_channel_mask_path"] = relative_to_dataset(root, output_mask_path)
    sample["executor_order"] = EXECUTOR_ORDER
    sample["feasibility"] = feasibility
    sample["label_source"] = label_source
    sample["negative_reason"] = negative_reason
    sample["quality_flag"] = "weak"
    sample["split"] = "val"
    sample["v2_candidate_update"] = {
        "pilot_id": pilot_id,
        "executor": executor,
        "source": "vlm_guided_candidate_selection_v2",
        "provenance": [
            "geometry_proposal_v2",
            "vlm_candidate_selection",
            "executor_rule_filter",
        ],
        "candidate_manifest": relative_to_dataset(root, candidate_manifest_path),
        "rule_filter_path": relative_to_dataset(root, rule_path),
        "selected_candidates": selected_ids,
        "manual_override": bool(manual_ids),
        "include_uncertain": bool(args.include_uncertain),
        "positive_points": int(target_mask.sum()),
        "requires_human_review": True,
    }
    sample["notes"] = (
        str(sample.get("notes", ""))
        + f" | v2_candidate: {executor} channel generated by geometry proposals + VLM selection + rule filtering; requires human review."
    )
    summary = {
        "pilot_id": pilot_id,
        "sample_id": sample_id,
        "executor": executor,
        "selected_candidates": selected_ids,
        "positive_points": int(target_mask.sum()),
        "output_mask_path": relative_to_dataset(root, output_mask_path),
    }
    return sample, summary


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    rows = select_rows(root, args)
    checked_samples = read_jsonl(resolve_path(root, args.samples))
    sample_by_id = {str(row.get("sample_id")): row for row in checked_samples}

    output_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for row in rows:
        sample, summary = build_for_row(root, args, row, sample_by_id)
        output_rows.append(sample)
        summaries.append(summary)

    output_samples = resolve_path(root, args.output_samples)
    write_jsonl(output_samples, output_rows, args.overwrite)

    split_dir = resolve_path(root, args.output_split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test", "contrast_test"]:
        split_path = split_dir / f"{split}.txt"
        if split_path.exists() and not args.overwrite:
            raise FileExistsError(f"Split file exists. Use --overwrite: {split_path}")
        ids = [row["sample_id"] for row in output_rows] if split == "val" else []
        split_path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")

    summary = {
        "version": "v2",
        "rows": len(summaries),
        "output_samples": relative_to_dataset(root, output_samples),
        "summaries": summaries,
        "notes": "v2 masks are candidate labels and must be reviewed before use as checked data.",
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
