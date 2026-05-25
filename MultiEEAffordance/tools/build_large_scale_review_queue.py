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
        default="feasible",
        help="Use only feasible channels or all four executor channels.",
    )
    parser.add_argument(
        "--quality-scope",
        choices=["needs_review", "all"],
        default="needs_review",
        help="needs_review keeps weak or needs_fix samples; all keeps every selected task sample.",
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
        if include_tasks is not None and task not in include_tasks:
            skipped[f"task_not_included:{task}"] += 1
            continue
        if task in exclude_tasks:
            skipped[f"task_excluded:{task}"] += 1
            continue
        if not sample_needs_review(sample, args.quality_scope):
            skipped["quality_not_selected"] += 1
            continue
        for executor in EXECUTOR_ORDER:
            if not executor_feasible(sample, executor, args.executor_scope):
                continue
            pilot_id = f"{args.pilot_prefix}_{len(out_rows) + 1:06d}"
            sample_id = str(sample.get("sample_id") or "")
            out_rows.append(
                {
                    "pilot_id": pilot_id,
                    "sample_id": sample_id,
                    "object_category": str(sample.get("object_category") or ""),
                    "task": task,
                    "executor": executor,
                    "decision": "review",
                    "issue_type": "large_scale_review",
                    "pilot_reason": (
                        "Large-scale first-pass human review queue; "
                        "candidate generation should propose top-k regions and reviewers finalize the mask."
                    ),
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
        "counts_by_task": dict(sorted(Counter(row["task"] for row in out_rows).items())),
        "counts_by_executor": dict(sorted(Counter(row["executor"] for row in out_rows).items())),
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
