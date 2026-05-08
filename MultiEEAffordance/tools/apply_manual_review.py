#!/usr/bin/env python3
"""Apply manual review decisions to produce a cleaned v0.1 dataset subset."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


EXECUTOR_ORDER = ["gripper", "suction", "hook", "dexterous_hand"]
SPLITS = ["train", "val", "test", "contrast_test"]
NEGATIVE_REASON_DEFAULTS = {
    "gripper": "no_graspable_region",
    "suction": "no_flat_suction_surface",
    "hook": "no_hookable_structure",
    "dexterous_hand": "ordinary_surface_without_operation_meaning",
}
REFINE_DECISIONS = {"refine", "add_missing", "uncertain"}
DISABLE_DECISIONS = {"disable", "not_applicable"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply manual review results to Multi-EE v0.1 samples.")
    parser.add_argument("--dataset-root", default=".", help="Dataset root.")
    parser.add_argument("--samples", default="processed/metadata/samples.jsonl", help="Input samples.jsonl.")
    parser.add_argument("--review-csv", default="processed/metadata/manual_review_v0_1.csv", help="Manual review CSV.")
    parser.add_argument(
        "--output-samples",
        default="processed/metadata/samples_checked_v0_1.jsonl",
        help="Output checked samples JSONL.",
    )
    parser.add_argument(
        "--output-mask-dir",
        default="processed/masks_checked_v0_1",
        help="Output checked mask directory.",
    )
    parser.add_argument(
        "--output-split-dir",
        default="splits_checked_v0_1",
        help="Output checked split directory.",
    )
    parser.add_argument(
        "--refine-queue",
        default="processed/metadata/refine_queue_v0_1.csv",
        help="Output refine queue CSV.",
    )
    parser.add_argument(
        "--rejected-samples",
        default="processed/metadata/rejected_samples_v0_1.csv",
        help="Output rejected samples CSV.",
    )
    parser.add_argument(
        "--summary-json",
        default="processed/metadata/manual_review_apply_summary_v0_1.json",
        help="Output machine-readable summary JSON.",
    )
    parser.add_argument("--min-positive-points", type=int, default=1, help="Minimum positive points for feasibility.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output files.")
    return parser.parse_args()


def error(message: str) -> None:
    raise ValueError(message)


def resolve_path(root: Path, path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else root / path


def relative_to_root(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def read_samples(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        error(f"samples.jsonl does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            row.setdefault("sample_id", f"{row.get('object_id', '')}_{row.get('task', '')}".strip("_"))
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def read_review(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        error(f"manual review CSV does not exist: {path}")
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            error(f"manual review CSV has no header: {path}")
        for row in reader:
            sample_id = (row.get("sample_id") or "").strip()
            if sample_id:
                rows[sample_id] = {key: (value or "").strip() for key, value in row.items() if key is not None}
    return rows


def ensure_can_write(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        error(f"Output already exists. Use --overwrite to replace: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def load_mask(root: Path, sample: dict[str, Any]) -> np.ndarray:
    mask_path = resolve_path(root, sample["multi_channel_mask_path"])
    if not mask_path.exists():
        error(f"Mask file does not exist for {sample['sample_id']}: {mask_path}")
    mask = np.load(mask_path, allow_pickle=False)
    if mask.ndim != 2 or mask.shape[1] != len(EXECUTOR_ORDER):
        error(f"Mask must have shape [N,4] for {sample['sample_id']}, got {mask.shape}")
    return (mask > 0).astype(np.uint8)


def infer_feasibility(mask: np.ndarray, min_positive_points: int) -> dict[str, bool]:
    return {
        executor: bool(int(mask[:, index].sum()) >= min_positive_points)
        for index, executor in enumerate(EXECUTOR_ORDER)
    }


def clean_negative_reason(
    sample: dict[str, Any],
    review: dict[str, str],
    feasibility: dict[str, bool],
) -> dict[str, str | None]:
    original = sample.get("negative_reason") if isinstance(sample.get("negative_reason"), dict) else {}
    result: dict[str, str | None] = {}
    for executor in EXECUTOR_ORDER:
        if feasibility[executor]:
            result[executor] = None
            continue
        decision = review.get(f"{executor}_decision", "")
        issue = review.get(f"{executor}_issue_type", "")
        if decision == "disable" and issue:
            result[executor] = issue
        elif decision == "not_applicable":
            result[executor] = "not_task_relevant"
        else:
            result[executor] = original.get(executor) or NEGATIVE_REASON_DEFAULTS[executor]
    return result


def clean_label_source(sample: dict[str, Any], feasibility: dict[str, bool]) -> dict[str, str]:
    original = sample.get("label_source") if isinstance(sample.get("label_source"), dict) else {}
    return {
        executor: (original.get(executor) or "mixed") if feasibility[executor] else "unavailable"
        for executor in EXECUTOR_ORDER
    }


def review_quality(sample: dict[str, Any], review: dict[str, str]) -> str:
    value = review.get("quality_after_review") or sample.get("quality_flag") or "weak"
    return value if value in {"weak", "checked", "verified"} else "weak"


def append_review_note(sample: dict[str, Any], review: dict[str, str]) -> str:
    pieces = []
    if sample.get("notes"):
        pieces.append(str(sample["notes"]))
    status = review.get("review_status", "")
    keep = review.get("keep_sample", "")
    issue = review.get("sample_issue_type", "")
    notes = review.get("sample_notes", "")
    pieces.append(f"manual_review_v0_1: status={status or 'empty'}, keep={keep or 'empty'}, issue={issue or 'empty'}")
    if notes:
        pieces.append(f"review_notes={notes}")
    return " | ".join(pieces)


def write_jsonl(path: Path, samples: list[dict[str, Any]], overwrite: bool) -> None:
    ensure_can_write(path, overwrite)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for sample in samples:
            cleaned = {key: value for key, value in sample.items() if not key.startswith("_")}
            f.write(json.dumps(cleaned, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str], overwrite: bool) -> None:
    ensure_can_write(path, overwrite)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_splits(path: Path, samples: list[dict[str, Any]], overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    grouped = {split: [] for split in SPLITS}
    for sample in samples:
        split = sample.get("split", "train")
        if split not in grouped:
            grouped[split] = []
        grouped[split].append(sample["sample_id"])
    for split, sample_ids in grouped.items():
        split_path = path / f"{split}.txt"
        if split_path.exists() and not overwrite:
            error(f"Output already exists. Use --overwrite to replace: {split_path}")
        with split_path.open("w", encoding="utf-8", newline="\n") as f:
            for sample_id in sample_ids:
                f.write(sample_id)
                f.write("\n")


def summarize_rule_error(issue_type: str, decision: str) -> str:
    if issue_type == "missing_positive" or decision == "add_missing":
        return "漏标：弱规则没有生成应有正样本"
    if issue_type == "over_label":
        return "过标：弱规则生成区域过大"
    if issue_type == "wrong_region":
        return "错区：弱规则标到错误部位"
    if issue_type == "executor_mismatch":
        return "执行器不匹配：物理可行性规则不足"
    if issue_type == "task_mismatch":
        return "任务不匹配：object-task 过滤不足"
    if issue_type == "needs_geometry_rule":
        return "缺少几何规则：需要法向/曲率/面积约束"
    if issue_type == "needs_part_annotation":
        return "缺少部件标注：需要 PartNet-Mobility 等结构信息"
    if decision == "refine":
        return "待精修：当前弱标签需要人工或规则修正"
    if decision == "disable":
        return "禁用通道：自动规则误判可行"
    return "其他"


def apply_review(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.dataset_root)
    samples_path = resolve_path(root, args.samples)
    review_path = resolve_path(root, args.review_csv)
    output_samples_path = resolve_path(root, args.output_samples)
    output_mask_dir = resolve_path(root, args.output_mask_dir)
    output_split_dir = resolve_path(root, args.output_split_dir)
    refine_queue_path = resolve_path(root, args.refine_queue)
    rejected_path = resolve_path(root, args.rejected_samples)
    summary_json_path = resolve_path(root, args.summary_json)

    samples = read_samples(samples_path)
    reviews = read_review(review_path)
    checked_samples: list[dict[str, Any]] = []
    refine_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []

    counters: dict[str, Counter[str]] = {
        "review_status": Counter(),
        "keep_sample": Counter(),
        "quality_after_review": Counter(),
        "sample_issue_type": Counter(),
        "rule_error": Counter(),
    }
    executor_decisions: dict[str, Counter[str]] = {executor: Counter() for executor in EXECUTOR_ORDER}
    executor_issues: dict[str, Counter[str]] = {executor: Counter() for executor in EXECUTOR_ORDER}
    task_keep: dict[str, Counter[str]] = defaultdict(Counter)

    for sample in samples:
        sample_id = sample["sample_id"]
        review = reviews.get(sample_id, {})
        keep = review.get("keep_sample", "")
        status = review.get("review_status", "")
        sample_issue = review.get("sample_issue_type", "")
        quality = review_quality(sample, review)

        counters["review_status"][status or "<empty>"] += 1
        counters["keep_sample"][keep or "<empty>"] += 1
        counters["quality_after_review"][quality or "<empty>"] += 1
        counters["sample_issue_type"][sample_issue or "<empty>"] += 1
        task_keep[sample.get("task", "<empty>")][keep or "<empty>"] += 1

        for executor in EXECUTOR_ORDER:
            decision = review.get(f"{executor}_decision", "")
            issue = review.get(f"{executor}_issue_type", "")
            executor_decisions[executor][decision or "<empty>"] += 1
            executor_issues[executor][issue or "<empty>"] += 1
            counters["rule_error"][summarize_rule_error(issue, decision)] += 1

        if keep == "no" or status == "reject":
            rejected_rows.append(
                {
                    "sample_id": sample_id,
                    "object_category": sample.get("object_category", ""),
                    "task": sample.get("task", ""),
                    "review_status": status,
                    "keep_sample": keep,
                    "sample_issue_type": sample_issue,
                    "sample_notes": review.get("sample_notes", ""),
                }
            )
            continue

        mask = load_mask(root, sample)
        for index, executor in enumerate(EXECUTOR_ORDER):
            decision = review.get(f"{executor}_decision", "")
            issue = review.get(f"{executor}_issue_type", "")
            notes = review.get(f"{executor}_notes", "")
            if decision in DISABLE_DECISIONS:
                mask[:, index] = 0
            if decision in REFINE_DECISIONS or issue in {
                "missing_positive",
                "over_label",
                "under_label",
                "wrong_region",
                "needs_geometry_rule",
                "needs_part_annotation",
            }:
                refine_rows.append(
                    {
                        "sample_id": sample_id,
                        "object_category": sample.get("object_category", ""),
                        "task": sample.get("task", ""),
                        "executor": executor,
                        "decision": decision,
                        "issue_type": issue,
                        "notes": notes,
                        "point_cloud_path": sample.get("point_cloud_path", ""),
                        "source_mask_path": sample.get("multi_channel_mask_path", ""),
                    }
                )

        checked_mask_path = output_mask_dir / f"{sample_id}.npy"
        ensure_can_write(checked_mask_path, args.overwrite)
        np.save(checked_mask_path, mask.astype(np.uint8))

        feasibility = infer_feasibility(mask, args.min_positive_points)
        cleaned_sample = dict(sample)
        cleaned_sample["multi_channel_mask_path"] = relative_to_root(checked_mask_path, root)
        cleaned_sample["feasibility"] = feasibility
        cleaned_sample["label_source"] = clean_label_source(sample, feasibility)
        cleaned_sample["negative_reason"] = clean_negative_reason(sample, review, feasibility)
        cleaned_sample["quality_flag"] = quality
        cleaned_sample["review_status"] = status or "checked"
        cleaned_sample["review_source"] = "manual_review_v0_1"
        cleaned_sample["notes"] = append_review_note(sample, review)
        checked_samples.append(cleaned_sample)

    write_jsonl(output_samples_path, checked_samples, args.overwrite)
    write_splits(output_split_dir, checked_samples, args.overwrite)
    write_csv(
        refine_queue_path,
        refine_rows,
        [
            "sample_id",
            "object_category",
            "task",
            "executor",
            "decision",
            "issue_type",
            "notes",
            "point_cloud_path",
            "source_mask_path",
        ],
        args.overwrite,
    )
    write_csv(
        rejected_path,
        rejected_rows,
        [
            "sample_id",
            "object_category",
            "task",
            "review_status",
            "keep_sample",
            "sample_issue_type",
            "sample_notes",
        ],
        args.overwrite,
    )

    summary = {
        "input_samples": len(samples),
        "checked_samples": len(checked_samples),
        "rejected_samples": len(rejected_rows),
        "refine_queue_rows": len(refine_rows),
        "output_samples": relative_to_root(output_samples_path, root),
        "output_mask_dir": relative_to_root(output_mask_dir, root),
        "output_split_dir": relative_to_root(output_split_dir, root),
        "refine_queue": relative_to_root(refine_queue_path, root),
        "rejected_samples_path": relative_to_root(rejected_path, root),
        "counters": {name: dict(counter) for name, counter in counters.items()},
        "executor_decisions": {executor: dict(counter) for executor, counter in executor_decisions.items()},
        "executor_issues": {executor: dict(counter) for executor, counter in executor_issues.items()},
        "task_keep": {task: dict(counter) for task, counter in task_keep.items()},
    }
    ensure_can_write(summary_json_path, args.overwrite)
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return summary


def main() -> int:
    args = parse_args()
    try:
        summary = apply_review(args)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
