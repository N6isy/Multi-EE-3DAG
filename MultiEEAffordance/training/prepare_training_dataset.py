#!/usr/bin/env python3
"""Build strict five-task training rows from point-level human review outputs.

Each annotation row normally records one reviewed executor channel. This script
merges those channel-level reviews into one object-task row with a canonical
[N, 4] mask and an explicit channel_supervision vector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .constants import (
    EXECUTORS,
    EXECUTOR_TO_INDEX,
    TASKS,
    TASK_TAXONOMY_VERSION,
    TASK_TO_INDEX,
    require_executor,
    require_five_task,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge human-refined five-task annotation rows into strict training manifests."
    )
    parser.add_argument("--dataset-root", required=True, help="Dataset root containing referenced point clouds and masks.")
    parser.add_argument(
        "--reviewed-samples",
        required=True,
        help="Comma-separated human-refined JSONL files, relative to dataset root unless absolute.",
    )
    parser.add_argument(
        "--output-root",
        default="processed/training/v0_2_5tasks",
        help="Training artifact directory relative to dataset root.",
    )
    parser.add_argument("--dataset-version", default="v0_2_5tasks")
    parser.add_argument("--split-seed", default="multi-ee-affordance-v0_2")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--min-reviewed-channels",
        type=int,
        default=1,
        help="Minimum reviewed executor channels required for a training row. Use 4 for fully reviewed releases.",
    )
    parser.add_argument(
        "--allowed-quality-flags",
        default="checked,verified",
        help="Comma-separated quality flags accepted for supervised training.",
    )
    parser.add_argument(
        "--allowed-review-statuses",
        default="checked,verified",
        help="Comma-separated point_review_status values accepted for supervised training.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def relative_to(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", value).strip("_")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}.")
            row["_training_source_file"] = relative_to(path.parent, path)
            row["_training_source_line"] = line_number
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            file.write("\n")


def write_json(path: Path, value: Any, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def reviewed_executor(row: dict[str, Any]) -> str:
    point_edit = row.get("v2_point_edit") if isinstance(row.get("v2_point_edit"), dict) else {}
    return require_executor(str(point_edit.get("executor") or row.get("executor") or ""))


def reviewed_mask_path(row: dict[str, Any]) -> str:
    point_edit = row.get("v2_point_edit") if isinstance(row.get("v2_point_edit"), dict) else {}
    value = str(row.get("multi_channel_mask_path") or point_edit.get("output_mask_path") or "").strip()
    if not value:
        raise ValueError("Reviewed row is missing multi_channel_mask_path.")
    return value


def load_points(path: Path) -> np.ndarray:
    points = np.load(path, allow_pickle=False)
    if points.ndim != 2 or points.shape[1] not in (3, 6):
        raise ValueError(f"Point cloud must have shape [N,3] or [N,6], got {points.shape}: {path}")
    return points


def load_reviewed_channel(path: Path, executor_index: int, point_count: int) -> np.ndarray:
    mask = np.load(path, allow_pickle=False)
    if mask.ndim == 2 and mask.shape == (point_count, len(EXECUTORS)):
        channel = mask[:, executor_index]
    elif mask.ndim == 1 and mask.shape[0] == point_count:
        channel = mask
    else:
        raise ValueError(
            f"Reviewed mask must have shape [N,4] or [N], got {mask.shape} for N={point_count}: {path}"
        )
    return (np.asarray(channel) > 0).astype(np.uint8)


def deterministic_split(object_id: str, seed: str, train_ratio: float, val_ratio: float) -> str:
    digest = hashlib.sha256(f"{seed}|{object_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def derive_quality(rows: list[dict[str, Any]]) -> str:
    values = {str(row.get("quality_flag") or "") for row in rows}
    return "verified" if values == {"verified"} else "checked"


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.min_reviewed_channels < 1 or args.min_reviewed_channels > len(EXECUTORS):
        raise ValueError("--min-reviewed-channels must be between 1 and 4.")
    if args.train_ratio < 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("Require train_ratio >= 0, val_ratio >= 0, and train_ratio + val_ratio < 1.")

    root = Path(args.dataset_root).resolve()
    output_root = resolve(root, args.output_root)
    masks_root = output_root / "masks"
    manifests_root = output_root / "manifests"
    allowed_quality = set(parse_csv(args.allowed_quality_flags))
    allowed_status = set(parse_csv(args.allowed_review_statuses))

    input_paths = [resolve(root, item) for item in parse_csv(args.reviewed_samples)]
    missing_inputs = [str(path) for path in input_paths if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError(f"Reviewed sample JSONL files not found: {missing_inputs}")

    input_rows: list[dict[str, Any]] = []
    for path in input_paths:
        input_rows.extend(read_jsonl(path))

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    rejected_rows: list[dict[str, Any]] = []
    for row in input_rows:
        try:
            task = require_five_task(str(row.get("task") or ""))
            require_executor(reviewed_executor(row))
            status = str(row.get("point_review_status") or "").strip()
            quality = str(row.get("quality_flag") or "").strip()
            if status not in allowed_status:
                raise ValueError(f"point_review_status={status!r} is not accepted")
            if quality not in allowed_quality:
                raise ValueError(f"quality_flag={quality!r} is not accepted")
            object_id = str(row.get("object_id") or "").strip()
            if not object_id:
                raise ValueError("missing object_id")
            grouped[(object_id, task)].append(row)
        except ValueError as exc:
            rejected_rows.append(
                {
                    "reason": str(exc),
                    "sample_id": row.get("sample_id", ""),
                    "row_key": row.get("row_key", ""),
                    "task": row.get("task", ""),
                }
            )

    output_rows: list[dict[str, Any]] = []
    incomplete_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    split_counter: Counter[str] = Counter()
    task_counter: Counter[str] = Counter()
    supervised_channel_counter: Counter[str] = Counter()

    for (object_id, task), rows in sorted(grouped.items()):
        point_paths = {str(row.get("point_cloud_path") or "").strip() for row in rows}
        if len(point_paths) != 1 or not next(iter(point_paths), ""):
            conflict_rows.append({"object_id": object_id, "task": task, "reason": "inconsistent point_cloud_path"})
            continue
        point_path_value = next(iter(point_paths))
        point_path = resolve(root, point_path_value)
        if not point_path.exists():
            conflict_rows.append({"object_id": object_id, "task": task, "reason": f"missing points: {point_path_value}"})
            continue
        try:
            points = load_points(point_path)
        except ValueError as exc:
            conflict_rows.append({"object_id": object_id, "task": task, "reason": str(exc)})
            continue

        merged = np.zeros((points.shape[0], len(EXECUTORS)), dtype=np.uint8)
        supervision = np.zeros(len(EXECUTORS), dtype=np.uint8)
        channel_sources: dict[str, dict[str, Any]] = {}
        group_conflict = False
        for row in rows:
            executor = reviewed_executor(row)
            executor_index = EXECUTOR_TO_INDEX[executor]
            mask_path_value = reviewed_mask_path(row)
            mask_path = resolve(root, mask_path_value)
            if not mask_path.exists():
                conflict_rows.append(
                    {"object_id": object_id, "task": task, "executor": executor, "reason": f"missing mask: {mask_path_value}"}
                )
                group_conflict = True
                break
            try:
                channel = load_reviewed_channel(mask_path, executor_index, points.shape[0])
            except ValueError as exc:
                conflict_rows.append({"object_id": object_id, "task": task, "executor": executor, "reason": str(exc)})
                group_conflict = True
                break
            if supervision[executor_index] and not np.array_equal(merged[:, executor_index], channel):
                conflict_rows.append(
                    {
                        "object_id": object_id,
                        "task": task,
                        "executor": executor,
                        "reason": "conflicting reviewed masks for the same object-task-executor",
                    }
                )
                group_conflict = True
                break
            merged[:, executor_index] = channel
            supervision[executor_index] = 1
            channel_sources[executor] = {
                "reviewer": str(row.get("reviewer") or row.get("point_review_reviewer") or ""),
                "source_mask_path": mask_path_value,
                "positive_points": int(channel.sum()),
            }

        if group_conflict:
            continue
        reviewed_channels = int(supervision.sum())
        if reviewed_channels < args.min_reviewed_channels:
            incomplete_rows.append(
                {
                    "object_id": object_id,
                    "task": task,
                    "reviewed_channels": reviewed_channels,
                    "required_channels": args.min_reviewed_channels,
                    "channel_supervision": supervision.tolist(),
                }
            )
            continue

        training_id = f"{object_id}_{task}"
        output_mask = masks_root / f"{safe_name(training_id)}.npy"
        if output_mask.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists. Use --overwrite: {output_mask}")
        output_mask.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_mask, merged)

        split = deterministic_split(object_id, args.split_seed, args.train_ratio, args.val_ratio)
        base = rows[0]
        output_row = {
            "training_id": training_id,
            "object_id": object_id,
            "source_dataset": str(base.get("source_dataset") or ""),
            "object_category": str(base.get("object_category") or ""),
            "task": task,
            "task_id": TASK_TO_INDEX[task],
            "task_taxonomy_version": TASK_TAXONOMY_VERSION,
            "point_cloud_path": relative_to(root, point_path),
            "multi_channel_mask_path": relative_to(root, output_mask),
            "executor_order": list(EXECUTORS),
            "channel_supervision": supervision.tolist(),
            "feasibility": (merged.sum(axis=0) > 0).astype(np.uint8).tolist(),
            "positive_points": merged.sum(axis=0).astype(int).tolist(),
            "quality_flag": derive_quality(rows),
            "human_review_only": True,
            "split": split,
            "channel_sources": channel_sources,
        }
        output_rows.append(output_row)
        split_counter[split] += 1
        task_counter[task] += 1
        for index, executor in enumerate(EXECUTORS):
            if supervision[index]:
                supervised_channel_counter[executor] += 1

    manifests_root.mkdir(parents=True, exist_ok=True)
    all_path = manifests_root / "all.jsonl"
    write_jsonl(all_path, output_rows, args.overwrite)
    for split in ("train", "val", "test"):
        write_jsonl(manifests_root / f"{split}.jsonl", [row for row in output_rows if row["split"] == split], args.overwrite)
    write_json(manifests_root / "rejected_rows.json", rejected_rows, args.overwrite)
    write_json(manifests_root / "incomplete_rows.json", incomplete_rows, args.overwrite)
    write_json(manifests_root / "conflict_rows.json", conflict_rows, args.overwrite)

    summary = {
        "version": args.dataset_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task_policy": {
            "tasks": list(TASKS),
            "task_taxonomy_version": TASK_TAXONOMY_VERSION,
            "legacy_tasks_accepted": False,
        },
        "executor_order": list(EXECUTORS),
        "input_files": [relative_to(root, path) for path in input_paths],
        "output_root": relative_to(root, output_root),
        "input_rows": len(input_rows),
        "grouped_object_tasks": len(grouped),
        "training_rows": len(output_rows),
        "rejected_input_rows": len(rejected_rows),
        "incomplete_object_tasks": len(incomplete_rows),
        "conflicting_object_tasks": len(conflict_rows),
        "min_reviewed_channels": args.min_reviewed_channels,
        "counts_by_task": dict(sorted(task_counter.items())),
        "counts_by_split": dict(sorted(split_counter.items())),
        "supervised_channels": dict(sorted(supervised_channel_counter.items())),
    }
    write_json(output_root / "summary.json", summary, args.overwrite)
    return summary


def main() -> int:
    args = parse_args()
    summary = prepare(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

