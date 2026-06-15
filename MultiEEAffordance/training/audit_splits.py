#!/usr/bin/env python3
"""Audit train/val/test manifests for CAD-asset leakage and distribution drift."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .constants import EXECUTORS, TASKS, make_asset_uid
from .prepare_training_dataset import resolve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit five-task training splits.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--fail-on-leakage", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def counter_to_dict(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    manifest = resolve(root, args.manifest)
    rows = read_jsonl(manifest)

    asset_to_splits: dict[str, set[str]] = defaultdict(set)
    object_to_splits: dict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    split_task_counts: dict[str, Counter[str]] = defaultdict(Counter)
    split_category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    split_source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    split_executor_supervised: dict[str, Counter[str]] = defaultdict(Counter)
    split_executor_feasible: dict[str, Counter[str]] = defaultdict(Counter)
    split_empty_channels: Counter[str] = Counter()
    split_supervised_channels: Counter[str] = Counter()
    missing_fields: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        split = str(row.get("split") or "unknown")
        explicit_asset_uid = str(row.get("asset_uid") or "").strip()
        asset_uid = explicit_asset_uid or make_asset_uid(row)
        object_id = str(row.get("object_id") or "")
        if not explicit_asset_uid:
            missing_fields.append({"line": index, "field": "asset_uid", "training_id": row.get("training_id", "")})
        if not str(row.get("source_asset_id") or "").strip():
            missing_fields.append({"line": index, "field": "source_asset_id", "training_id": row.get("training_id", "")})
        if not str(row.get("split_key") or "").strip():
            missing_fields.append({"line": index, "field": "split_key", "training_id": row.get("training_id", "")})
        if not asset_uid or asset_uid == "unknown:unknown_asset":
            missing_fields.append({"line": index, "field": "asset_uid", "training_id": row.get("training_id", "")})
        if not object_id:
            missing_fields.append({"line": index, "field": "object_id", "training_id": row.get("training_id", "")})
        split_counts[split] += 1
        asset_to_splits[asset_uid].add(split)
        object_to_splits[object_id].add(split)
        split_task_counts[split][str(row.get("task") or "")] += 1
        split_category_counts[split][str(row.get("object_category") or "unknown")] += 1
        split_source_counts[split][str(row.get("source_dataset") or "unknown")] += 1
        supervision = row.get("channel_supervision") or [1] * len(EXECUTORS)
        feasibility = row.get("feasibility") or [0] * len(EXECUTORS)
        for executor_index, executor in enumerate(EXECUTORS):
            supervised = int(executor_index < len(supervision) and bool(supervision[executor_index]))
            feasible = int(executor_index < len(feasibility) and bool(feasibility[executor_index]))
            if supervised:
                split_executor_supervised[split][executor] += 1
                split_supervised_channels[split] += 1
                if feasible:
                    split_executor_feasible[split][executor] += 1
                else:
                    split_empty_channels[split] += 1

    asset_leakage = {asset: sorted(splits) for asset, splits in asset_to_splits.items() if len(splits) > 1}
    object_leakage = {object_id: sorted(splits) for object_id, splits in object_to_splits.items() if object_id and len(splits) > 1}
    empty_ratio = {
        split: float(split_empty_channels[split] / max(split_supervised_channels[split], 1))
        for split in sorted(split_counts)
    }
    task_missing = {
        split: sorted(set(TASKS).difference(split_task_counts[split].keys()))
        for split in sorted(split_counts)
    }
    executor_missing = {
        split: sorted(set(EXECUTORS).difference(split_executor_supervised[split].keys()))
        for split in sorted(split_counts)
    }
    warnings: list[str] = []
    for split, missing in task_missing.items():
        if missing:
            warnings.append(f"{split} missing tasks: {missing}")
    for split, missing in executor_missing.items():
        if missing:
            warnings.append(f"{split} missing supervised executors: {missing}")
    if max(empty_ratio.values() or [0.0]) - min(empty_ratio.values() or [0.0]) > 0.2:
        warnings.append("empty-mask ratio differs by more than 0.20 across splits")

    summary = {
        "status": "ok" if not asset_leakage and not object_leakage and not missing_fields else "failed",
        "manifest": args.manifest,
        "rows": len(rows),
        "split_counts": counter_to_dict(split_counts),
        "asset_leakage": asset_leakage,
        "object_leakage": object_leakage,
        "missing_fields": missing_fields[:200],
        "empty_ratio_by_split": empty_ratio,
        "counts_by_split_task": {split: counter_to_dict(counter) for split, counter in sorted(split_task_counts.items())},
        "counts_by_split_category": {
            split: counter_to_dict(counter) for split, counter in sorted(split_category_counts.items())
        },
        "counts_by_split_source_dataset": {
            split: counter_to_dict(counter) for split, counter in sorted(split_source_counts.items())
        },
        "supervised_by_split_executor": {
            split: counter_to_dict(counter) for split, counter in sorted(split_executor_supervised.items())
        },
        "feasible_by_split_executor": {
            split: counter_to_dict(counter) for split, counter in sorted(split_executor_feasible.items())
        },
        "task_missing_by_split": task_missing,
        "executor_missing_by_split": executor_missing,
        "warnings": warnings,
    }
    output = resolve(root, args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    failed = bool(asset_leakage or object_leakage or missing_fields)
    return 1 if failed and args.fail_on_leakage else 0


if __name__ == "__main__":
    raise SystemExit(main())
