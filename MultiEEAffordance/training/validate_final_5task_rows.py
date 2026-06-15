#!/usr/bin/env python3
"""Validate the final five-task object-task-executor JSONL before training prep."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .constants import EXECUTORS, TASKS, TASK_TAXONOMY_VERSION
from .prepare_final_5task_training_dataset import (
    ArrayCache,
    read_jsonl,
    relative_to,
    resolve,
    summarize_object_coverage,
    validate_row,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate final five-task row-level JSONL.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--final-samples", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    final_samples = resolve(root, args.final_samples)
    output_json = resolve(root, args.output_json)
    if not final_samples.exists():
        raise FileNotFoundError(f"Final samples JSONL not found: {final_samples}")

    raw_rows = read_jsonl(final_samples)
    cache = ArrayCache(root)
    valid_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for row in raw_rows:
        valid, error = validate_row(row, root, cache)
        if valid is not None:
            valid_rows.append(valid)
        if error is not None:
            errors.append(error)

    coverage = summarize_object_coverage(valid_rows)
    task_counts = Counter(row["task"] for row in valid_rows)
    executor_counts = Counter(row["executor"] for row in valid_rows)
    source_counts = Counter(row["source_dataset"] for row in valid_rows)
    synthetic_empty_counts = Counter(row["executor"] for row in valid_rows if row.get("is_synthetic_empty_combo"))
    missing_or_duplicate = bool(coverage["missing_combination_count"] or coverage["duplicate_combination_count"])
    status = "ok" if not errors and not missing_or_duplicate else "failed"
    summary = {
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "final_samples": relative_to(root, final_samples),
        "task_taxonomy_version": TASK_TAXONOMY_VERSION,
        "tasks": list(TASKS),
        "executor_order": list(EXECUTORS),
        "input_rows": len(raw_rows),
        "valid_rows": len(valid_rows),
        "error_count": len(errors),
        "errors": errors[:500],
        "coverage": coverage,
        "counts_by_task": dict(sorted(task_counts.items())),
        "counts_by_executor": dict(sorted(executor_counts.items())),
        "counts_by_source_dataset": dict(sorted(source_counts.items())),
        "synthetic_empty_rows_by_executor": dict(sorted(synthetic_empty_counts.items())),
        "checks": {
            "task_must_be_five_task": True,
            "executor_order_must_be_fixed": list(EXECUTORS),
            "points_mask_n_must_match": True,
            "mask_shape_must_be_n_by_4": True,
            "positive_points_after_must_match_executor_channel_sum": True,
            "object_should_have_20_rows": True,
        },
    }
    write_json(output_json, summary, overwrite=args.overwrite)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if status != "ok" and args.fail_on_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
