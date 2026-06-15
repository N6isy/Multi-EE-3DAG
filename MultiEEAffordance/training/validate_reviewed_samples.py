#!/usr/bin/env python3
"""Validate human-refined five-task samples before training dataset merge."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .constants import EXECUTORS, infer_source_asset_id, infer_source_dataset, make_asset_uid, require_executor, require_five_task
from .prepare_training_dataset import parse_csv, resolve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate refined human review JSONL files for five-task training.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--reviewed-samples", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--allow-errors", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_source_file"] = path.as_posix()
            row["_source_line"] = line_number
            rows.append(row)
    return rows


def reviewer(row: dict[str, Any]) -> str:
    return str(row.get("reviewer") or row.get("point_review_reviewer") or row.get("reviewer_id") or "").strip()


def mask_path(row: dict[str, Any]) -> str:
    edit = row.get("v2_point_edit") if isinstance(row.get("v2_point_edit"), dict) else {}
    return str(row.get("multi_channel_mask_path") or edit.get("output_mask_path") or "").strip()


def validate_row(root: Path, row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        require_five_task(str(row.get("task") or ""))
    except ValueError as exc:
        errors.append(str(exc))
    try:
        require_executor(str(row.get("executor") or ""))
    except ValueError as exc:
        errors.append(str(exc))
    if not str(row.get("object_id") or "").strip():
        errors.append("missing object_id")
    if not reviewer(row):
        errors.append("missing reviewer")
    points_value = str(row.get("point_cloud_path") or "").strip()
    mask_value = mask_path(row)
    if not points_value:
        errors.append("missing point_cloud_path")
    if not mask_value:
        errors.append("missing multi_channel_mask_path")
    if errors:
        return errors
    points_file = resolve(root, points_value)
    mask_file = resolve(root, mask_value)
    if not points_file.exists():
        errors.append(f"missing point cloud: {points_value}")
        return errors
    if not mask_file.exists():
        errors.append(f"missing mask: {mask_value}")
        return errors
    try:
        points = np.load(points_file, allow_pickle=False)
        mask = np.load(mask_file, allow_pickle=False)
    except Exception as exc:
        errors.append(f"failed to load point/mask npy: {exc}")
        return errors
    if points.ndim != 2 or points.shape[1] not in (3, 6):
        errors.append(f"bad point shape {points.shape}")
    if mask.ndim == 2:
        if mask.shape != (points.shape[0], len(EXECUTORS)):
            errors.append(f"bad mask shape {mask.shape}; expected {(points.shape[0], len(EXECUTORS))}")
    elif mask.ndim == 1:
        if mask.shape[0] != points.shape[0]:
            errors.append(f"bad single-channel mask shape {mask.shape}; expected {(points.shape[0],)}")
    else:
        errors.append(f"bad mask ndim {mask.ndim}")
    return errors


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    rows: list[dict[str, Any]] = []
    for item in parse_csv(args.reviewed_samples):
        rows.extend(read_jsonl(resolve(root, item)))

    errors: list[dict[str, Any]] = []
    counts_by_task: Counter[str] = Counter()
    counts_by_executor: Counter[str] = Counter()
    counts_by_reviewer: Counter[str] = Counter()
    counts_by_source: Counter[str] = Counter()
    asset_rows: Counter[str] = Counter()
    for row in rows:
        row_errors = validate_row(root, row)
        if row_errors:
            errors.append(
                {
                    "source_file": row.get("_source_file"),
                    "source_line": row.get("_source_line"),
                    "sample_id": row.get("sample_id", ""),
                    "row_key": row.get("row_key", ""),
                    "errors": row_errors,
                }
            )
            continue
        counts_by_task[str(row["task"])] += 1
        counts_by_executor[str(row["executor"])] += 1
        counts_by_reviewer[reviewer(row)] += 1
        counts_by_source[infer_source_dataset(row)] += 1
        asset_rows[make_asset_uid(row)] += 1

    summary = {
        "status": "ok" if not errors else "failed",
        "rows": len(rows),
        "valid_rows": len(rows) - len(errors),
        "error_rows": len(errors),
        "unique_assets": len(asset_rows),
        "counts_by_task": dict(sorted(counts_by_task.items())),
        "counts_by_executor": dict(sorted(counts_by_executor.items())),
        "counts_by_reviewer": dict(sorted(counts_by_reviewer.items())),
        "counts_by_source_dataset": dict(sorted(counts_by_source.items())),
        "asset_rule": "source_asset_id is inferred from explicit source asset fields; 3D AffordanceNet falls back to object_id.",
        "sample_asset_examples": [
            {"asset_uid": asset_uid, "rows": count} for asset_uid, count in asset_rows.most_common(10)
        ],
        "errors": errors[:200],
    }
    output = resolve(root, args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if args.allow_errors or not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
