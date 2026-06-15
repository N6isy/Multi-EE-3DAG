#!/usr/bin/env python3
"""Compute reviewer agreement on duplicated five-task annotation rows."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from .constants import EXECUTOR_TO_INDEX, make_asset_uid, require_executor, require_five_task
from .prepare_training_dataset import parse_csv, resolve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit point-level consistency between reviewers.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--reviewed-samples", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default="")
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


def group_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("object_id") or ""), require_five_task(str(row.get("task") or "")), require_executor(str(row.get("executor") or "")))


def load_channel(root: Path, row: dict[str, Any]) -> np.ndarray:
    executor_index = EXECUTOR_TO_INDEX[require_executor(str(row.get("executor") or ""))]
    mask = np.load(resolve(root, mask_path(row)), allow_pickle=False)
    if mask.ndim == 2:
        channel = mask[:, executor_index]
    elif mask.ndim == 1:
        channel = mask
    else:
        raise ValueError(f"Bad mask ndim {mask.ndim}: {mask_path(row)}")
    return (channel > 0).astype(np.uint8)


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.logical_and(left > 0, right > 0).sum()
    union = np.logical_or(left > 0, right > 0).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    rows: list[dict[str, Any]] = []
    for item in parse_csv(args.reviewed_samples):
        rows.extend(read_jsonl(resolve(root, item)))

    groups: dict[tuple[str, str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    skipped: list[dict[str, Any]] = []
    for row in rows:
        try:
            rev = reviewer(row)
            if not rev:
                raise ValueError("missing reviewer")
            groups[group_key(row)][rev].append(row)
        except ValueError as exc:
            skipped.append({"sample_id": row.get("sample_id", ""), "reason": str(exc)})

    pair_rows: list[dict[str, Any]] = []
    ious: list[float] = []
    feasibility_agree = 0
    empty_agree = 0
    total_pairs = 0
    disagreements_by_task: Counter[str] = Counter()
    disagreements_by_executor: Counter[str] = Counter()
    for key, by_reviewer in sorted(groups.items()):
        if len(by_reviewer) < 2:
            continue
        for left_reviewer, right_reviewer in combinations(sorted(by_reviewer), 2):
            left_row = by_reviewer[left_reviewer][-1]
            right_row = by_reviewer[right_reviewer][-1]
            try:
                left_mask = load_channel(root, left_row)
                right_mask = load_channel(root, right_row)
            except Exception as exc:
                skipped.append({"sample_id": left_row.get("sample_id", ""), "reason": str(exc)})
                continue
            iou = mask_iou(left_mask, right_mask)
            left_feasible = bool(left_mask.sum() > 0)
            right_feasible = bool(right_mask.sum() > 0)
            feasible_same = left_feasible == right_feasible
            empty_same = (not left_feasible) == (not right_feasible)
            total_pairs += 1
            ious.append(iou)
            feasibility_agree += int(feasible_same)
            empty_agree += int(empty_same)
            if iou < 0.5 or not feasible_same:
                disagreements_by_task[key[1]] += 1
                disagreements_by_executor[key[2]] += 1
            pair_rows.append(
                {
                    "object_id": key[0],
                    "task": key[1],
                    "executor": key[2],
                    "asset_uid": make_asset_uid(left_row),
                    "left_reviewer": left_reviewer,
                    "right_reviewer": right_reviewer,
                    "point_iou": iou,
                    "left_positive": int(left_mask.sum()),
                    "right_positive": int(right_mask.sum()),
                    "feasibility_agree": feasible_same,
                    "empty_agree": empty_same,
                }
            )

    ious_sorted = sorted(ious)
    summary = {
        "rows": len(rows),
        "overlap_groups": sum(1 for value in groups.values() if len(value) >= 2),
        "reviewer_pairs": total_pairs,
        "mean_point_iou": float(sum(ious) / max(len(ious), 1)),
        "median_point_iou": float(ious_sorted[len(ious_sorted) // 2]) if ious_sorted else None,
        "feasibility_agreement": float(feasibility_agree / max(total_pairs, 1)),
        "empty_mask_agreement": float(empty_agree / max(total_pairs, 1)),
        "disagreements_by_task": dict(sorted(disagreements_by_task.items())),
        "disagreements_by_executor": dict(sorted(disagreements_by_executor.items())),
        "skipped": skipped[:200],
        "lowest_iou_cases": sorted(pair_rows, key=lambda item: item["point_iou"])[:50],
    }
    output_json = resolve(root, args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_csv:
        output_csv = resolve(root, args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "object_id",
                    "task",
                    "executor",
                    "asset_uid",
                    "left_reviewer",
                    "right_reviewer",
                    "point_iou",
                    "left_positive",
                    "right_positive",
                    "feasibility_agree",
                    "empty_agree",
                ],
            )
            writer.writeheader()
            writer.writerows(pair_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
