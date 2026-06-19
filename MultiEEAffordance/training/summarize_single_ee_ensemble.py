#!/usr/bin/env python3
"""Summarize four single-executor runs as an independent Single-EE ensemble.

The script does not re-run inference. It reads each single_ee_* test_metrics.json
and assembles the executor-specific cells into one comparable summary table.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .constants import EXECUTORS, TASKS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize single-EE metrics as a four-executor ensemble.")
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True, help="Per-executor summary CSV.")
    parser.add_argument("--output-matrix-csv", default="", help="Optional 5x4 task-executor matrix CSV.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def run_path(runs_root: Path, executor: str) -> Path:
    return runs_root / f"single_ee_{executor}_5tasks" / "test_metrics.json"


def main() -> int:
    args = parse_args()
    runs_root = Path(args.runs_root).resolve()
    per_executor: dict[str, Any] = {}
    matrix: dict[str, dict[str, Any]] = {task: {} for task in TASKS}
    metric_paths: dict[str, str] = {}

    for executor in EXECUTORS:
        path = run_path(runs_root, executor)
        if not path.exists():
            raise FileNotFoundError(f"Missing single-EE metrics for {executor}: {path}")
        metrics = read_json(path)
        metric_paths[executor] = path.as_posix()
        executor_metrics = metrics.get("per_executor", {}).get(executor, {})
        if not isinstance(executor_metrics, dict):
            executor_metrics = {}
        per_executor[executor] = {
            "source_run": path.parent.name,
            "metrics_path": path.as_posix(),
            "iou": executor_metrics.get("iou", 0.0),
            "dice": executor_metrics.get("dice", 0.0),
            "feasibility_f1": executor_metrics.get("feasibility_f1", 0.0),
            "feasibility_precision": executor_metrics.get("feasibility_precision", 0.0),
            "feasibility_recall": executor_metrics.get("feasibility_recall", 0.0),
            "empty_mask_accuracy": executor_metrics.get("empty_mask_accuracy", 0.0),
            "small_part_recall": executor_metrics.get("small_part_recall", 0.0),
            "supervised_samples": executor_metrics.get("supervised_samples", 0),
            "feasible_samples": executor_metrics.get("feasible_samples", 0),
        }
        task_matrix = metrics.get("task_executor_iou", {})
        if not isinstance(task_matrix, dict):
            task_matrix = {}
        for task in TASKS:
            task_row = task_matrix.get(task, {})
            if not isinstance(task_row, dict):
                task_row = {}
            cell = task_row.get(executor, {})
            matrix[task][executor] = cell if isinstance(cell, dict) else {}

    matrix_iou_values = [
        float(cell.get("iou", 0.0))
        for task in TASKS
        for cell in [matrix[task].get(executor, {}) for executor in EXECUTORS]
        if int(cell.get("feasible_samples", 0) or 0) > 0
    ]
    matrix_dice_values = [
        float(cell.get("dice", 0.0))
        for task in TASKS
        for cell in [matrix[task].get(executor, {}) for executor in EXECUTORS]
        if int(cell.get("feasible_samples", 0) or 0) > 0
    ]
    payload = {
        "experiment": "single_ee_ensemble_5tasks",
        "summary_type": "metric_level_assembly",
        "notes": (
            "Each executor column is taken from its own independently trained Single-EE model. "
            "This is a fairness baseline for joint multi-executor prediction; overlap matrix error is not computed here."
        ),
        "metric_paths": metric_paths,
        "macro_iou": mean(matrix_iou_values),
        "macro_dice": mean(matrix_dice_values),
        "macro_feasibility_f1": mean([float(per_executor[e].get("feasibility_f1", 0.0)) for e in EXECUTORS]),
        "empty_mask_accuracy": mean([float(per_executor[e].get("empty_mask_accuracy", 0.0)) for e in EXECUTORS]),
        "small_part_recall": mean([float(per_executor[e].get("small_part_recall", 0.0)) for e in EXECUTORS]),
        "executor_overlap_matrix_error": None,
        "per_executor": per_executor,
        "task_executor_iou": matrix,
    }

    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "executor",
                "source_run",
                "iou",
                "dice",
                "feasibility_f1",
                "feasibility_precision",
                "feasibility_recall",
                "empty_mask_accuracy",
                "small_part_recall",
                "supervised_samples",
                "feasible_samples",
                "metrics_path",
            ],
        )
        writer.writeheader()
        for executor in EXECUTORS:
            writer.writerow({"executor": executor, **per_executor[executor]})

    matrix_csv = Path(args.output_matrix_csv).resolve() if args.output_matrix_csv else output_json.with_name(output_json.stem + "_task_executor_matrix.csv")
    matrix_csv.parent.mkdir(parents=True, exist_ok=True)
    with matrix_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["task", "executor", "iou", "dice", "feasible_samples", "supervised_samples"],
        )
        writer.writeheader()
        for task in TASKS:
            for executor in EXECUTORS:
                cell = matrix[task].get(executor, {})
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
    print(
        json.dumps(
            {
                "output_json": output_json.as_posix(),
                "output_csv": output_csv.as_posix(),
                "matrix_csv": matrix_csv.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
