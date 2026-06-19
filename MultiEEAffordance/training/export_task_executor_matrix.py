#!/usr/bin/env python3
"""Export the 5 task x 4 executor metric matrix from a metrics JSON file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .constants import EXECUTORS, TASKS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export task-executor matrix metrics to CSV.")
    parser.add_argument("--metrics-json", required=True, help="test_metrics.json or evaluate_calibrated output JSON.")
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def extract_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    nested = payload.get("metrics")
    if isinstance(nested, dict) and "task_executor_iou" in nested:
        return nested
    return payload


def main() -> int:
    args = parse_args()
    metrics_path = Path(args.metrics_json).resolve()
    metrics = extract_metrics(read_json(metrics_path))
    matrix = metrics.get("task_executor_iou")
    if not isinstance(matrix, dict):
        raise ValueError(f"No task_executor_iou found in {metrics_path}.")
    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["task", "executor", "iou", "dice", "feasible_samples", "supervised_samples"],
        )
        writer.writeheader()
        for task in TASKS:
            task_row = matrix.get(task, {})
            if not isinstance(task_row, dict):
                task_row = {}
            for executor in EXECUTORS:
                cell = task_row.get(executor, {})
                if not isinstance(cell, dict):
                    cell = {}
                writer.writerow(
                    {
                        "task": task,
                        "executor": executor,
                        "iou": cell.get("iou", ""),
                        "dice": cell.get("dice", ""),
                        "feasible_samples": cell.get("feasible_samples", ""),
                        "supervised_samples": cell.get("supervised_samples", ""),
                    }
                )
    print(json.dumps({"metrics_json": metrics_path.as_posix(), "output_csv": output_csv.as_posix()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
