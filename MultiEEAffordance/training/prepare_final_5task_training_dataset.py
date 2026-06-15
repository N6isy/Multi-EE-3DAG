#!/usr/bin/env python3
"""Prepare training manifests from the final five-task row-level JSONL.

The final annotation file has one row for one object-task-executor combination.
This script validates those rows and compacts them into the training format used
by MultiEEFiveTaskDataset: one row per object-task with a full [N, 4] mask.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
    infer_source_asset_id,
    infer_source_dataset,
    make_asset_uid,
    require_executor,
    require_five_task,
)


EXPECTED_ROWS_PER_OBJECT = len(TASKS) * len(EXECUTORS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build strict five-task training manifests from the final object-task-executor JSONL."
    )
    parser.add_argument("--dataset-root", required=True, help="Dataset root containing point clouds and masks.")
    parser.add_argument(
        "--final-samples",
        required=True,
        help="Final cleaned JSONL; one row is one object-task-executor sample.",
    )
    parser.add_argument(
        "--output-root",
        default="processed/training/v0_4_final_5tasks",
        help="Training artifact directory relative to dataset root.",
    )
    parser.add_argument("--dataset-version", default="v0_4_final_5tasks")
    parser.add_argument("--split-seed", default="multi-ee-affordance-v0_4_final_5tasks")
    parser.add_argument(
        "--split-unit",
        choices=["source_asset", "object"],
        default="source_asset",
        help="Use source_asset for CAD-asset-disjoint splits; object is only for debugging.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--copy-masks",
        action="store_true",
        help=(
            "Always copy each object-task mask into output_root/masks. By default the script reuses an existing "
            "final mask only when all four executor rows already point to the same [N,4] mask; otherwise it writes "
            "a canonical merged mask automatically."
        ),
    )
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help="Write manifests from valid groups even if row validation finds errors. Not recommended for final experiments.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def relative_to(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def safe_id(value: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
    return "".join(ch if ch in allowed else "_" for ch in value).strip("_")


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
            row["_source_jsonl"] = str(path)
            row["_source_line"] = line_number
            rows.append(row)
    return rows


def write_json(path: Path, value: Any, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            file.write("\n")


def normalize_executor_order(value: Any) -> list[str]:
    if value is None or value == "":
        return list(EXECUTORS)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                value = parsed
            else:
                value = [item.strip() for item in value.split(",") if item.strip()]
        except json.JSONDecodeError:
            value = [item.strip() for item in value.split(",") if item.strip()]
    if not isinstance(value, list):
        raise ValueError(f"executor_order must be a list or comma-separated string; got {type(value).__name__}")
    order = [require_executor(str(item)) for item in value]
    if order != EXECUTORS:
        raise ValueError(f"executor_order must be exactly {EXECUTORS}; got {order}")
    return order


def row_executor(row: dict[str, Any]) -> str:
    return require_executor(str(row.get("executor") or row.get("target_executor") or ""))


def row_positive_count(row: dict[str, Any]) -> int:
    value = row.get("positive_points_after", row.get("positive_points", 0))
    if value in ("", None):
        return 0
    return int(float(value))


def split_key_for(row: dict[str, Any], split_unit: str) -> str:
    object_id = str(row.get("object_id") or "").strip()
    if split_unit == "object":
        return object_id
    explicit = str(row.get("split_key") or "").strip()
    if explicit:
        return explicit
    explicit_asset_uid = str(row.get("asset_uid") or "").strip()
    if explicit_asset_uid:
        return explicit_asset_uid
    return make_asset_uid(row)


def deterministic_split(split_key: str, seed: str, train_ratio: float, val_ratio: float) -> str:
    digest = hashlib.sha256(f"{seed}|{split_key}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


class ArrayCache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.points: dict[str, dict[str, Any]] = {}
        self.masks: dict[str, dict[str, Any]] = {}

    def load_points_info(self, rel_path: str) -> dict[str, Any]:
        if rel_path in self.points:
            return self.points[rel_path]
        path = resolve(self.root, rel_path)
        if not path.exists():
            raise FileNotFoundError(f"Point cloud not found: {rel_path}")
        points = np.load(path, allow_pickle=False)
        if points.ndim != 2 or points.shape[1] not in (3, 6):
            raise ValueError(f"Point cloud must be [N,3] or [N,6], got {points.shape}: {rel_path}")
        info = {"shape": tuple(int(v) for v in points.shape), "path": path}
        self.points[rel_path] = info
        return info

    def load_mask_info(self, rel_path: str) -> dict[str, Any]:
        if rel_path in self.masks:
            return self.masks[rel_path]
        path = resolve(self.root, rel_path)
        if not path.exists():
            raise FileNotFoundError(f"Mask not found: {rel_path}")
        mask = np.load(path, allow_pickle=False)
        if mask.ndim != 2 or mask.shape[1] != len(EXECUTORS):
            raise ValueError(f"Mask must be [N,4], got {mask.shape}: {rel_path}")
        binary = (mask > 0).astype(np.uint8)
        info = {
            "shape": tuple(int(v) for v in mask.shape),
            "path": path,
            "positive_points": binary.sum(axis=0).astype(int).tolist(),
        }
        self.masks[rel_path] = info
        return info


def validate_row(row: dict[str, Any], root: Path, cache: ArrayCache) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    line_ref = {
        "line": row.get("_source_line"),
        "object_id": row.get("object_id", ""),
        "task": row.get("task", ""),
        "executor": row.get("executor", row.get("target_executor", "")),
    }
    try:
        object_id = str(row.get("object_id") or "").strip()
        if not object_id:
            raise ValueError("missing object_id")
        task = require_five_task(str(row.get("task") or ""))
        executor = row_executor(row)
        executor_order = normalize_executor_order(row.get("executor_order"))
        point_path = str(row.get("point_cloud_path") or "").strip()
        mask_path = str(row.get("multi_channel_mask_path") or "").strip()
        if not point_path:
            raise ValueError("missing point_cloud_path")
        if not mask_path:
            raise ValueError("missing multi_channel_mask_path")
        points_info = cache.load_points_info(point_path)
        mask_info = cache.load_mask_info(mask_path)
        if int(points_info["shape"][0]) != int(mask_info["shape"][0]):
            raise ValueError(
                f"points/mask N mismatch: points={points_info['shape']} mask={mask_info['shape']}"
            )
        declared_point_count = row.get("point_count")
        if declared_point_count not in ("", None) and int(declared_point_count) != int(points_info["shape"][0]):
            raise ValueError(f"point_count mismatch: row={declared_point_count} actual={points_info['shape'][0]}")
        executor_index = executor_order.index(executor)
        actual_positive = int(mask_info["positive_points"][executor_index])
        declared_positive = row_positive_count(row)
        if declared_positive != actual_positive:
            raise ValueError(
                f"positive_points_after mismatch for {executor}: row={declared_positive} actual={actual_positive}"
            )
        normalized = dict(row)
        normalized.update(
            {
                "object_id": object_id,
                "task": task,
                "executor": executor,
                "target_executor": executor,
                "executor_order": executor_order,
                "point_cloud_path": relative_to(root, resolve(root, point_path)),
                "multi_channel_mask_path": relative_to(root, resolve(root, mask_path)),
                "point_count": int(points_info["shape"][0]),
                "positive_points_after": actual_positive,
                "source_dataset": infer_source_dataset(row),
                "source_asset_id": infer_source_asset_id(row),
                "asset_uid": make_asset_uid(row),
            }
        )
        return normalized, None
    except Exception as exc:  # noqa: BLE001 - diagnostics should keep all malformed rows.
        error = dict(line_ref)
        error["reason"] = str(exc)
        return None, error


def summarize_object_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_object: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    duplicate_counter: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        key = (str(row["object_id"]), str(row["task"]), str(row["executor"]))
        duplicate_counter[key] += 1
        by_object[key[0]][key[1]].add(key[2])

    missing: list[dict[str, Any]] = []
    for object_id, task_map in sorted(by_object.items()):
        for task in TASKS:
            missing_executors = sorted(set(EXECUTORS).difference(task_map.get(task, set())))
            if missing_executors:
                missing.append({"object_id": object_id, "task": task, "missing_executors": missing_executors})
    duplicates = [
        {"object_id": object_id, "task": task, "executor": executor, "count": count}
        for (object_id, task, executor), count in sorted(duplicate_counter.items())
        if count > 1
    ]
    complete_objects = sum(
        1
        for task_map in by_object.values()
        if all(set(task_map.get(task, set())) == set(EXECUTORS) for task in TASKS)
    )
    return {
        "objects": len(by_object),
        "expected_rows_per_object": EXPECTED_ROWS_PER_OBJECT,
        "complete_objects": complete_objects,
        "incomplete_objects": len(by_object) - complete_objects,
        "missing_combinations": missing[:500],
        "duplicate_combinations": duplicates[:500],
        "missing_combination_count": len(missing),
        "duplicate_combination_count": len(duplicates),
    }


def load_mask_channel(root: Path, row: dict[str, Any]) -> np.ndarray:
    mask = np.load(resolve(root, row["multi_channel_mask_path"]), allow_pickle=False)
    if mask.ndim != 2 or mask.shape[1] != len(EXECUTORS):
        raise ValueError(f"Mask must be [N,4], got {mask.shape}: {row['multi_channel_mask_path']}")
    executor_index = EXECUTOR_TO_INDEX[row["executor"]]
    return (mask[:, executor_index] > 0).astype(np.uint8)


def build_training_rows(
    rows: list[dict[str, Any]],
    *,
    root: Path,
    output_root: Path,
    copy_masks: bool,
    split_unit: str,
    split_seed: str,
    train_ratio: float,
    val_ratio: float,
    overwrite: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["object_id"], row["task"])].append(row)

    training_rows: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    split_assignments: dict[str, dict[str, Any]] = {}

    for (object_id, task), group in sorted(grouped.items()):
        rows_by_executor: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            rows_by_executor[row["executor"]].append(row)
        if sorted(rows_by_executor.keys()) != sorted(EXECUTORS):
            conflicts.append(
                {
                    "object_id": object_id,
                    "task": task,
                    "reason": "object-task must have exactly one row for each executor",
                    "executors": sorted(rows_by_executor.keys()),
                }
            )
            continue
        selected_rows: list[dict[str, Any]] = []
        duplicate_conflict = False
        for executor in EXECUTORS:
            executor_rows = rows_by_executor[executor]
            if len(executor_rows) == 1:
                selected_rows.append(executor_rows[0])
                continue
            try:
                channels = [load_mask_channel(root, row) for row in executor_rows]
            except ValueError as exc:
                conflicts.append({"object_id": object_id, "task": task, "executor": executor, "reason": str(exc)})
                duplicate_conflict = True
                break
            first = channels[0]
            if not all(np.array_equal(first, channel) for channel in channels[1:]):
                conflicts.append(
                    {
                        "object_id": object_id,
                        "task": task,
                        "executor": executor,
                        "reason": "duplicate object-task-executor rows have different mask channels",
                        "rows": [row.get("_source_line") for row in executor_rows],
                    }
                )
                duplicate_conflict = True
                break
            selected_rows.append(executor_rows[0])
        if duplicate_conflict:
            continue
        group = selected_rows
        point_paths = sorted({row["point_cloud_path"] for row in group})
        mask_paths = sorted({row["multi_channel_mask_path"] for row in group})
        if len(point_paths) != 1:
            conflicts.append({"object_id": object_id, "task": task, "reason": "inconsistent point_cloud_path"})
            continue

        base = group[0]
        source_dataset_values = sorted({row["source_dataset"] for row in group})
        source_asset_values = sorted({row["source_asset_id"] for row in group})
        asset_uid_values = sorted({row["asset_uid"] for row in group})
        if len(source_dataset_values) != 1 or len(source_asset_values) != 1 or len(asset_uid_values) != 1:
            conflicts.append(
                {
                    "object_id": object_id,
                    "task": task,
                    "reason": "inconsistent source asset metadata within object-task group",
                    "source_datasets": source_dataset_values,
                    "source_asset_ids": source_asset_values,
                    "asset_uids": asset_uid_values,
                }
            )
            continue

        mask = np.zeros((int(base["point_count"]), len(EXECUTORS)), dtype=np.uint8)
        group_positive_matches = True
        for row in group:
            executor_index = EXECUTOR_TO_INDEX[row["executor"]]
            channel = load_mask_channel(root, row)
            if channel.shape[0] != mask.shape[0]:
                conflicts.append(
                    {
                        "object_id": object_id,
                        "task": task,
                        "executor": row["executor"],
                        "reason": f"mask channel N mismatch: expected {mask.shape[0]}, got {channel.shape[0]}",
                    }
                )
                group_positive_matches = False
                break
            mask[:, executor_index] = channel
        if not group_positive_matches:
            continue
        output_mask = output_root / "masks" / f"{safe_id(object_id)}_{task}.npy"
        if output_mask.exists() and not overwrite:
            raise FileExistsError(f"Output exists. Use --overwrite: {output_mask}")
        output_mask.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_mask, mask)
        output_mask_path = relative_to(root, output_mask)

        positive_points = mask.sum(axis=0).astype(int).tolist()
        feasibility = [int(count > 0) for count in positive_points]
        split_key = split_key_for(base, split_unit)
        split = deterministic_split(split_key, split_seed, train_ratio, val_ratio)
        split_assignments.setdefault(
            split_key,
            {
                "split_key": split_key,
                "split": split,
                "source_dataset": source_dataset_values[0],
                "source_asset_id": source_asset_values[0],
                "asset_uid": asset_uid_values[0],
                "object_ids": [],
            },
        )
        if object_id not in split_assignments[split_key]["object_ids"]:
            split_assignments[split_key]["object_ids"].append(object_id)

        source_rows = {
            row["executor"]: {
                "line": row.get("_source_line"),
                "positive_points_after": row.get("positive_points_after", 0),
                "is_synthetic_empty_combo": bool(row.get("is_synthetic_empty_combo", False)),
            }
            for row in group
        }
        training_rows.append(
            {
                "training_id": f"{object_id}_{task}",
                "object_id": object_id,
                "source_dataset": source_dataset_values[0],
                "source_asset_id": source_asset_values[0],
                "asset_uid": asset_uid_values[0],
                "split_key": split_key,
                "split_unit": split_unit,
                "object_category": str(base.get("object_category") or ""),
                "task": task,
                "task_id": TASK_TO_INDEX[task],
                "task_taxonomy_version": TASK_TAXONOMY_VERSION,
                "point_cloud_path": point_paths[0],
                "multi_channel_mask_path": output_mask_path,
                "canonical_mask_source_path": mask_paths[0] if len(mask_paths) == 1 else "",
                "canonical_mask_source_paths": mask_paths,
                "canonical_mask_was_merged": bool(len(mask_paths) > 1),
                "executor_order": list(EXECUTORS),
                "channel_supervision": [1, 1, 1, 1],
                "feasibility": feasibility,
                "positive_points": positive_points,
                "point_count": int(mask.shape[0]),
                "annotation_source": "final_human_review",
                "quality_flag": "human_review",
                "human_review_only": True,
                "dataset_version": "final_5tasks",
                "split": split,
                "source_rows_by_executor": source_rows,
                "synthetic_empty_by_executor": {
                    row["executor"]: bool(row.get("is_synthetic_empty_combo", False)) for row in group
                },
            }
        )

    return training_rows, conflicts, list(split_assignments.values())


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    train_ratio = float(args.train_ratio)
    val_ratio = float(args.val_ratio)
    if train_ratio < 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Require train_ratio >= 0, val_ratio >= 0, and train_ratio + val_ratio < 1.")

    root = Path(args.dataset_root).resolve()
    final_samples = resolve(root, args.final_samples)
    output_root = resolve(root, args.output_root)
    manifests_root = output_root / "manifests"
    if not final_samples.exists():
        raise FileNotFoundError(f"Final samples JSONL not found: {final_samples}")

    raw_rows = read_jsonl(final_samples)
    cache = ArrayCache(root)
    valid_rows: list[dict[str, Any]] = []
    validation_errors: list[dict[str, Any]] = []
    for row in raw_rows:
        valid, error = validate_row(row, root, cache)
        if valid is not None:
            valid_rows.append(valid)
        if error is not None:
            validation_errors.append(error)

    coverage = summarize_object_coverage(valid_rows)
    if (validation_errors or coverage["missing_combination_count"] or coverage["duplicate_combination_count"]) and not args.allow_errors:
        write_json(output_root / "validation_errors.json", validation_errors, overwrite=args.overwrite)
        write_json(output_root / "object_task_coverage.json", coverage, overwrite=args.overwrite)
        raise SystemExit(
            "Final samples failed validation. Inspect validation_errors.json and object_task_coverage.json; "
            "rerun with --allow-errors only for debugging."
        )

    training_rows, conflicts, split_assignments = build_training_rows(
        valid_rows,
        root=root,
        output_root=output_root,
        copy_masks=bool(args.copy_masks),
        split_unit=str(args.split_unit),
        split_seed=str(args.split_seed),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        overwrite=bool(args.overwrite),
    )
    if conflicts and not args.allow_errors:
        write_json(output_root / "validation_errors.json", validation_errors, overwrite=args.overwrite)
        write_json(output_root / "object_task_coverage.json", coverage, overwrite=args.overwrite)
        write_json(output_root / "object_task_conflicts.json", conflicts, overwrite=args.overwrite)
        raise SystemExit("Object-task compaction failed. Inspect object_task_conflicts.json.")

    split_counts = Counter(row["split"] for row in training_rows)
    task_counts = Counter(row["task"] for row in training_rows)
    source_counts = Counter(row["source_dataset"] for row in training_rows)
    category_counts = Counter(row["object_category"] or "unknown" for row in training_rows)
    synthetic_empty_counts = Counter()
    feasible_counts = Counter()
    for row in training_rows:
        for executor, is_synthetic in row["synthetic_empty_by_executor"].items():
            if is_synthetic:
                synthetic_empty_counts[executor] += 1
        for index, executor in enumerate(EXECUTORS):
            if row["feasibility"][index]:
                feasible_counts[executor] += 1

    write_jsonl(manifests_root / "all.jsonl", training_rows, overwrite=args.overwrite)
    for split in ("train", "val", "test"):
        split_rows = [row for row in training_rows if row["split"] == split]
        write_jsonl(manifests_root / f"{split}.jsonl", split_rows, overwrite=args.overwrite)
    write_json(output_root / "validation_errors.json", validation_errors, overwrite=args.overwrite)
    write_json(output_root / "object_task_coverage.json", coverage, overwrite=args.overwrite)
    write_json(output_root / "object_task_conflicts.json", conflicts, overwrite=args.overwrite)
    write_json(output_root / "split_assignments.json", split_assignments, overwrite=args.overwrite)

    summary = {
        "version": str(args.dataset_version),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "final_samples": relative_to(root, final_samples),
        "output_root": relative_to(root, output_root),
        "input_rows": len(raw_rows),
        "valid_input_rows": len(valid_rows),
        "validation_errors": len(validation_errors),
        "training_object_task_rows": len(training_rows),
        "object_task_conflicts": len(conflicts),
        "objects": coverage["objects"],
        "complete_objects": coverage["complete_objects"],
        "incomplete_objects": coverage["incomplete_objects"],
        "tasks": list(TASKS),
        "executor_order": list(EXECUTORS),
        "task_taxonomy_version": TASK_TAXONOMY_VERSION,
        "split_policy": {
            "split_unit": str(args.split_unit),
            "split_seed": str(args.split_seed),
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": 1.0 - train_ratio - val_ratio,
            "asset_rule": "3D AffordanceNet rows without explicit source_asset_id use object_id as source_asset_id.",
        },
        "copy_masks": True,
        "copy_masks_note": (
            "Final row-level JSONL may store one mask path per executor. "
            "This script always writes merged canonical [N,4] masks into output_root/masks."
        ),
        "counts_by_split": dict(sorted(split_counts.items())),
        "counts_by_task": dict(sorted(task_counts.items())),
        "counts_by_source_dataset": dict(sorted(source_counts.items())),
        "counts_by_category": dict(sorted(category_counts.items())),
        "feasible_object_tasks_by_executor": dict(sorted(feasible_counts.items())),
        "synthetic_empty_rows_by_executor": dict(sorted(synthetic_empty_counts.items())),
        "manifests": {
            "all": relative_to(root, manifests_root / "all.jsonl"),
            "train": relative_to(root, manifests_root / "train.jsonl"),
            "val": relative_to(root, manifests_root / "val.jsonl"),
            "test": relative_to(root, manifests_root / "test.jsonl"),
        },
    }
    write_json(output_root / "summary.json", summary, overwrite=args.overwrite)
    return summary


def main() -> int:
    args = parse_args()
    summary = prepare(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
