#!/usr/bin/env python3
"""Build a channel-level review queue CSV for large-scale v3 annotation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXECUTOR_ORDER = ["gripper", "suction", "hook", "dexterous_hand"]
KNOWN_TASKS = {"pick_up", "lift_carry", "open_pull", "press_push"}
DEFAULT_ACTIVE_TASKS = {"pick_up", "open_pull", "press_push"}
FIELDNAMES = [
    "pilot_id",
    "sample_id",
    "object_category",
    "task",
    "executor",
    "decision",
    "issue_type",
    "review_mode",
    "object_task_feasible",
    "executor_feasible",
    "negative_reason",
    "pilot_reason",
    "point_cloud_path",
    "source_mask_path",
    "checked_mask_path",
    "render_output_dir",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a large-scale channel-level review queue for v3.")
    parser.add_argument("--dataset-root", default=".", help="Dataset root directory.")
    parser.add_argument("--samples", default="processed/metadata/samples_checked_v0_1.jsonl")
    parser.add_argument("--output-csv", default="processed/metadata/v3_large_scale_review_queue_v0_1.csv")
    parser.add_argument("--summary-json", default="processed/metadata/v3_large_scale_review_queue_summary_v0_1.json")
    parser.add_argument("--pilot-prefix", default="v3_review")
    parser.add_argument("--include-tasks", default="pick_up,open_pull,press_push", help="Comma-separated tasks or 'all'.")
    parser.add_argument("--exclude-tasks", default="lift_carry", help="Comma-separated tasks to drop.")
    parser.add_argument(
        "--executor-scope",
        choices=["feasible", "all"],
        default="all",
        help="Use only feasible channels or all four executor channels.",
    )
    parser.add_argument(
        "--quality-scope",
        choices=["needs_review", "all"],
        default="all",
        help="needs_review keeps weak or needs_fix samples; all keeps every selected task sample.",
    )
    parser.add_argument(
        "--empty-policy",
        choices=["review", "skip"],
        default="review",
        help="How to handle infeasible executor channels when --executor-scope all.",
    )
    parser.add_argument(
        "--common-sense-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter clearly impossible object-task pairs before expensive VLM stages.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def relative_to_dataset(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def write_csv(path: Path, rows: list[dict[str, str]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def parse_task_filter(value: str) -> set[str] | None:
    raw = str(value or "").strip()
    if raw.lower() == "all":
        return None
    tasks = set(parse_list(raw))
    unknown = sorted(tasks.difference(KNOWN_TASKS))
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}. Known tasks: {sorted(KNOWN_TASKS)}")
    return tasks


def sample_needs_review(row: dict[str, Any], quality_scope: str) -> bool:
    if quality_scope == "all":
        return True
    quality = str(row.get("quality_flag") or "")
    review_status = str(row.get("review_status") or "")
    return quality != "checked" or review_status in {"needs_fix", "pending", "refine_needed"}


def executor_feasible(row: dict[str, Any], executor: str, executor_scope: str) -> bool:
    if executor_scope == "all":
        return True
    feasibility = row.get("feasibility", {})
    if isinstance(feasibility, dict):
        return bool(feasibility.get(executor, False))
    return False


def source_executor_feasible(row: dict[str, Any], executor: str) -> bool:
    feasibility = row.get("feasibility", {})
    if isinstance(feasibility, dict):
        return bool(feasibility.get(executor, False))
    return False


OPEN_PULL_CATEGORIES = {
    "Door",
    "Dishwasher",
    "Faucet",
    "Laptop",
    "Microwave",
    "Refrigerator",
    "StorageFurniture",
    "TrashCan",
}
PRESS_PUSH_CATEGORIES = {
    "Door",
    "Display",
    "Dishwasher",
    "Faucet",
    "Keyboard",
    "Laptop",
    "Microwave",
    "Refrigerator",
    "StorageFurniture",
}


def object_task_feasible(category: str, task: str, common_sense_filter: bool) -> tuple[bool, str]:
    if not common_sense_filter:
        return True, ""
    if task == "open_pull" and category not in OPEN_PULL_CATEGORIES:
        return False, f"common_sense_object_task_mismatch:{category}:{task}"
    if task == "press_push" and category not in PRESS_PUSH_CATEGORIES:
        return False, f"common_sense_object_task_mismatch:{category}:{task}"
    return True, ""


def executor_negative_reason(sample: dict[str, Any], executor: str, feasible: bool) -> str:
    negative = sample.get("negative_reason", {})
    if isinstance(negative, dict):
        value = negative.get(executor)
        if value:
            return str(value)
    if feasible:
        return ""
    return f"no_{executor}_feasible_region"


def build_queue(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.dataset_root).resolve()
    include_tasks = parse_task_filter(args.include_tasks)
    exclude_tasks = set(parse_list(args.exclude_tasks))
    unknown_excluded = sorted(exclude_tasks.difference(KNOWN_TASKS))
    if unknown_excluded:
        raise ValueError(f"Unknown excluded tasks: {unknown_excluded}. Known tasks: {sorted(KNOWN_TASKS)}")

    input_path = resolve_path(root, args.samples)
    samples = read_jsonl(input_path)
    out_rows: list[dict[str, str]] = []
    skipped = Counter()
    for sample in samples:
        task = str(sample.get("task") or "")
        category = str(sample.get("object_category") or "")
        if include_tasks is not None and task not in include_tasks:
            skipped[f"task_not_included:{task}"] += 1
            continue
        if task in exclude_tasks:
            skipped[f"task_excluded:{task}"] += 1
            continue
        ok_task, task_reason = object_task_feasible(category, task, args.common_sense_filter)
        if not ok_task:
            skipped[task_reason] += 1
            continue
        if not sample_needs_review(sample, args.quality_scope):
            skipped["quality_not_selected"] += 1
            continue
        for executor in EXECUTOR_ORDER:
            if not executor_feasible(sample, executor, args.executor_scope):
                continue
            feasible = source_executor_feasible(sample, executor)
            if not feasible and args.empty_policy == "skip":
                skipped[f"executor_infeasible_skipped:{executor}"] += 1
                continue
            pilot_id = f"{args.pilot_prefix}_{len(out_rows) + 1:06d}"
            sample_id = str(sample.get("sample_id") or "")
            negative_reason = executor_negative_reason(sample, executor, feasible)
            decision = "review" if feasible else "empty_review_required"
            issue_type = "large_scale_review" if feasible else "executor_infeasible_empty_label"
            review_mode = "point_refine" if feasible else "confirm_empty"
            pilot_reason = (
                "Large-scale first-pass human review queue; "
                "candidate generation should propose top-k regions and reviewers finalize the mask."
                if feasible
                else (
                    "This object-task pair is kept, but this executor channel is marked infeasible by source metadata. "
                    "Reviewer should confirm the empty label unless a valid affordance region is visible."
                )
            )
            out_rows.append(
                {
                    "pilot_id": pilot_id,
                    "sample_id": sample_id,
                    "object_category": category,
                    "task": task,
                    "executor": executor,
                    "decision": decision,
                    "issue_type": issue_type,
                    "review_mode": review_mode,
                    "object_task_feasible": "true",
                    "executor_feasible": "true" if feasible else "false",
                    "negative_reason": negative_reason,
                    "pilot_reason": pilot_reason,
                    "point_cloud_path": str(sample.get("point_cloud_path") or ""),
                    "source_mask_path": str(sample.get("multi_channel_mask_path") or ""),
                    "checked_mask_path": str(sample.get("multi_channel_mask_path") or ""),
                    "render_output_dir": f"processed/vlm_semantic_part/renders/{sample_id}",
                }
            )
            if args.limit is not None and len(out_rows) >= args.limit:
                break
        if args.limit is not None and len(out_rows) >= args.limit:
            break

    output_csv = resolve_path(root, args.output_csv)
    write_csv(output_csv, out_rows, args.overwrite)
    summary = {
        "version": "v0_1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_samples": relative_to_dataset(root, input_path),
        "output_csv": relative_to_dataset(root, output_csv),
        "rows": len(out_rows),
        "task_policy": {
            "include_tasks": sorted(include_tasks) if include_tasks is not None else "all",
            "exclude_tasks": sorted(exclude_tasks),
            "default_active_tasks": sorted(DEFAULT_ACTIVE_TASKS),
        },
        "executor_scope": args.executor_scope,
        "quality_scope": args.quality_scope,
        "empty_policy": args.empty_policy,
        "common_sense_filter": bool(args.common_sense_filter),
        "counts_by_task": dict(sorted(Counter(row["task"] for row in out_rows).items())),
        "counts_by_executor": dict(sorted(Counter(row["executor"] for row in out_rows).items())),
        "counts_by_decision": dict(sorted(Counter(row["decision"] for row in out_rows).items())),
        "counts_by_category": dict(sorted(Counter(row["object_category"] for row in out_rows).items())),
        "skipped": dict(sorted(skipped.items())),
    }
    write_json(resolve_path(root, args.summary_json), summary, args.overwrite)
    return summary


def main() -> int:
    args = parse_args()
    summary = build_queue(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
