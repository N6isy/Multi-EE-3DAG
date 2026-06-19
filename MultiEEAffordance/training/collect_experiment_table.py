#!/usr/bin/env python3
"""Collect experiment metrics into an AAAI-style CSV/JSON table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_COLUMNS = [
    "experiment",
    "macro_iou",
    "macro_dice",
    "macro_feasibility_f1",
    "feasibility_auroc",
    "empty_mask_accuracy",
    "small_part_recall",
    "executor_overlap_matrix_error",
    "params",
    "flops",
    "metrics_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect training run metrics into a CSV table.")
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument(
        "--metric-pattern",
        action="append",
        default=[],
        help="Glob pattern relative to runs-root. Can be repeated. Defaults to */test_metrics.json and */metrics.json.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return value


def find_metric_files(root: Path, patterns: list[str]) -> list[Path]:
    if not patterns:
        patterns = ["*/test_metrics.json", "*/metrics.json"]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(root.glob(pattern))
    return sorted({path.resolve() for path in candidates})


def extract_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    nested = payload.get("metrics")
    if isinstance(nested, dict) and "macro_iou" in nested:
        return nested
    return payload


def row_from_metrics(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    metrics = extract_metrics(payload)
    experiment = path.parent.name
    if path.stem not in {"test_metrics", "metrics", "summary"}:
        experiment = f"{experiment}:{path.stem}"
    if payload.get("experiment"):
        experiment = str(payload["experiment"])
    config_path = path.parent / "resolved_config.json"
    params = metrics.get("params")
    flops = metrics.get("flops")
    if config_path.exists():
        try:
            config = read_json(config_path)
            experiment = str(config.get("experiment_name") or experiment)
            params = params if params is not None else config.get("params")
            flops = flops if flops is not None else config.get("flops")
        except Exception:
            pass
    return {
        "experiment": experiment,
        "macro_iou": metrics.get("macro_iou", ""),
        "macro_dice": metrics.get("macro_dice", ""),
        "macro_feasibility_f1": metrics.get("macro_feasibility_f1", metrics.get("macro_feasibility_accuracy", "")),
        "feasibility_auroc": metrics.get("feasibility_auroc", ""),
        "empty_mask_accuracy": metrics.get("empty_mask_accuracy", ""),
        "small_part_recall": metrics.get("small_part_recall", ""),
        "executor_overlap_matrix_error": metrics.get("executor_overlap_matrix_error", ""),
        "params": params or "",
        "flops": flops or "",
        "metrics_path": path.as_posix(),
    }


def main() -> int:
    args = parse_args()
    runs_root = Path(args.runs_root).resolve()
    rows = [row_from_metrics(path, read_json(path)) for path in find_metric_files(runs_root, args.metric_pattern)]
    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=DEFAULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    if args.output_json:
        output_json = Path(args.output_json).resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output_csv": output_csv.as_posix()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
