#!/usr/bin/env python3
"""Build a four-channel mask from selected geometry hook candidates.

This writes a normal Multi-EE sample JSONL that can be opened by the existing
review app. The output is still a candidate label, not verified ground truth.
"""

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
    parser = argparse.ArgumentParser(description="Build selected hook candidate 3D masks.")
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
        "--candidate-root",
        default="processed/vlm_pilot/hook_candidates",
        help="Candidate root relative to dataset root.",
    )
    parser.add_argument(
        "--selection-root",
        default="processed/vlm_pilot/hook_candidate_selection",
        help="Selection root relative to dataset root.",
    )
    parser.add_argument(
        "--output-mask-root",
        default="processed/vlm_pilot/hook_candidate_masks",
        help="Output mask root relative to dataset root.",
    )
    parser.add_argument(
        "--output-samples",
        default="processed/metadata/hook_candidate_samples_v0_1.jsonl",
        help="Output candidate samples JSONL relative to dataset root.",
    )
    parser.add_argument(
        "--output-split-dir",
        default="splits_hook_candidates",
        help="Output split directory relative to dataset root.",
    )
    parser.add_argument(
        "--summary-json",
        default="processed/metadata/hook_candidate_summary_v0_1.json",
        help="Output summary JSON relative to dataset root.",
    )
    parser.add_argument("--pilot-id", default=None, help="Build only one pilot row.")
    parser.add_argument("--limit", type=int, default=None, help="Limit selected hook pilot rows.")
    parser.add_argument("--executor", default="hook", help="Executor to process. Default: hook.")
    parser.add_argument("--min-votes", type=int, default=1, help="Minimum view votes required to use a candidate.")
    parser.add_argument(
        "--selected-candidates",
        default="",
        help="Optional comma-separated candidate ids, for manual/debug override such as A or A,B.",
    )
    parser.add_argument("--allow-empty", action="store_true", help="Allow writing an empty hook channel.")
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


def select_rows(args: argparse.Namespace, root: Path) -> list[dict[str, str]]:
    rows = read_csv(resolve_path(root, args.pilot_csv))
    rows = [row for row in rows if row.get("executor") == args.executor]
    if args.pilot_id:
        rows = [row for row in rows if row.get("pilot_id") == args.pilot_id]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No hook pilot rows selected.")
    return rows


def candidate_ids_from_selection(selection: dict[str, Any], min_votes: int) -> list[str]:
    ranked = selection.get("ranked_candidates", [])
    selected: list[str] = []
    if isinstance(ranked, list):
        for item in ranked:
            if not isinstance(item, dict):
                continue
            candidate_id = str(item.get("candidate_id", "")).strip().upper()
            try:
                votes = int(item.get("votes", 0))
            except (TypeError, ValueError):
                votes = 0
            if candidate_id and votes >= min_votes and candidate_id not in selected:
                selected.append(candidate_id)
    if selected:
        return selected
    raw_selected = selection.get("selected_candidates", [])
    if isinstance(raw_selected, str):
        raw_selected = [raw_selected]
    if isinstance(raw_selected, list):
        for item in raw_selected:
            candidate_id = str(item).strip().upper()
            if candidate_id and candidate_id not in selected:
                selected.append(candidate_id)
    return selected


def parse_selected_candidates(value: str) -> list[str]:
    selected: list[str] = []
    for item in str(value or "").split(","):
        candidate_id = item.strip().upper()
        if candidate_id and candidate_id not in selected:
            selected.append(candidate_id)
    return selected


def build_for_row(
    root: Path,
    args: argparse.Namespace,
    row: dict[str, str],
    sample_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    pilot_id = row["pilot_id"]
    sample_id = row["sample_id"]
    if sample_id not in sample_by_id:
        raise KeyError(f"Pilot sample is not in checked samples: {sample_id}")
    sample = dict(sample_by_id[sample_id])

    candidate_manifest_path = resolve_path(root, args.candidate_root) / pilot_id / "candidate_manifest.json"
    selection_path = resolve_path(root, args.selection_root) / pilot_id / "combined_selection.json"
    candidate_manifest = read_json(candidate_manifest_path)
    manual_selected_ids = parse_selected_candidates(args.selected_candidates)
    if selection_path.exists():
        selection = read_json(selection_path)
    elif manual_selected_ids:
        selection = {
            "selected_candidates": manual_selected_ids,
            "ranked_candidates": [
                {"candidate_id": candidate_id, "votes": int(args.min_votes), "mean_confidence": 1.0}
                for candidate_id in manual_selected_ids
            ],
        }
    else:
        raise FileNotFoundError(f"Selection JSON not found: {selection_path}")
    candidate_npz_path = resolve_portable_path(root, candidate_manifest["candidate_npz"], candidate_manifest_path.parent)
    if not candidate_npz_path.exists():
        raise FileNotFoundError(f"Candidate NPZ not found: {candidate_npz_path}")

    data = np.load(candidate_npz_path, allow_pickle=True)
    candidate_ids = [str(item).upper() for item in data["candidate_ids"].tolist()]
    candidate_masks = data["candidate_masks"].astype(np.uint8)
    selected_ids = manual_selected_ids or candidate_ids_from_selection(selection, args.min_votes)
    selected_indices = [candidate_ids.index(item) for item in selected_ids if item in candidate_ids]

    hook_mask = np.zeros((candidate_masks.shape[1],), dtype=np.uint8)
    if selected_indices:
        hook_mask = (candidate_masks[selected_indices].sum(axis=0) > 0).astype(np.uint8)
    if int(hook_mask.sum()) == 0 and not args.allow_empty:
        raise ValueError(
            f"Selected hook mask is empty for {pilot_id}. "
            "Inspect combined_selection.json or rerun with --allow-empty if this is expected."
        )

    base_mask_path = resolve_portable_path(root, row.get("checked_mask_path") or sample["multi_channel_mask_path"])
    if not base_mask_path.exists():
        raise FileNotFoundError(f"Base checked mask not found: {base_mask_path}")
    base_mask = np.load(base_mask_path)
    if base_mask.ndim != 2 or base_mask.shape[1] != len(EXECUTOR_ORDER):
        raise ValueError(f"Expected base mask shape [N,4], got {base_mask.shape}: {base_mask_path}")
    if base_mask.shape[0] != hook_mask.shape[0]:
        raise ValueError(f"Hook candidate length {hook_mask.shape[0]} does not match base mask {base_mask.shape[0]}")

    output_mask = base_mask.astype(np.uint8).copy()
    hook_channel = EXECUTOR_ORDER.index("hook")
    output_mask[:, hook_channel] = hook_mask
    output_mask_path = resolve_path(root, args.output_mask_root) / f"{sample_id}_{pilot_id}_hook_candidate.npy"
    if output_mask_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output mask exists. Use --overwrite: {output_mask_path}")
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_mask_path, output_mask)

    label_source = dict(sample.get("label_source", {}))
    label_source["hook"] = "geometry_rule"
    feasibility = dict(sample.get("feasibility", {}))
    feasibility["hook"] = bool(int(hook_mask.sum()) > 0)
    negative_reason = dict(sample.get("negative_reason", {}))
    negative_reason["hook"] = None if feasibility["hook"] else "no_selected_hook_candidate"

    sample["multi_channel_mask_path"] = relative_to_dataset(root, output_mask_path)
    sample["label_source"] = label_source
    sample["feasibility"] = feasibility
    sample["negative_reason"] = negative_reason
    sample["quality_flag"] = "weak"
    sample["split"] = "val"
    sample["hook_candidate_update"] = {
        "pilot_id": pilot_id,
        "source": "geometry_hook_candidate_vlm_selected",
        "provenance": ["geometry_rule", "vlm_candidate_selection"],
        "candidate_manifest": relative_to_dataset(root, candidate_manifest_path),
        "selection_path": relative_to_dataset(root, selection_path),
        "selected_candidates": selected_ids,
        "manual_override": bool(manual_selected_ids),
        "min_votes": int(args.min_votes),
        "positive_points": int(hook_mask.sum()),
        "requires_human_review": True,
    }
    sample["notes"] = (
        str(sample.get("notes", ""))
        + " | hook_candidate: geometry proposal selected by Qwen3-VL; requires human review."
    )

    summary = {
        "pilot_id": pilot_id,
        "sample_id": sample_id,
        "selected_candidates": selected_ids,
        "positive_points": int(hook_mask.sum()),
        "output_mask_path": relative_to_dataset(root, output_mask_path),
    }
    return sample, summary


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    rows = select_rows(args, root)
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
        "notes": "These hook masks are candidate labels and must be checked in the review app.",
    }
    summary_path = resolve_path(root, args.summary_json)
    write_json(summary_path, summary, args.overwrite)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
