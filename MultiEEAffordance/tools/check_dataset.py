#!/usr/bin/env python3
"""Validate Multi-EE Affordance Dataset v0.1 metadata, points, and masks."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.task_taxonomy import ALL_TASKS, EXECUTOR_ORDER

TASKS = set(ALL_TASKS)
SOURCE_DATASETS = {"3d_affordancenet", "partnet_mobility", "shapenet", "objaverse", "manual"}
QUALITY_FLAGS = {"weak", "checked", "verified"}
SPLITS = {"train", "val", "test", "contrast_test"}
LABEL_SOURCES = {
    "existing_affordance_mask",
    "part_annotation",
    "geometry_rule",
    "manual_refinement",
    "mixed",
    "unavailable",
}

REQUIRED_FIELDS = [
    "object_id",
    "source_dataset",
    "object_category",
    "task",
    "task_instruction",
    "point_cloud_path",
    "multi_channel_mask_path",
    "executor_order",
    "feasibility",
    "label_source",
    "negative_reason",
    "quality_flag",
    "split",
]


@dataclass
class Issue:
    level: str
    line_no: int
    object_id: str
    message: str

    def format(self) -> str:
        sample = self.object_id if self.object_id else "<unknown>"
        return f"{self.level}: line {self.line_no}, object {sample}: {self.message}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Multi-EE Affordance Dataset v0.1 files.")
    parser.add_argument("--dataset-root", required=True, help="Dataset root, e.g. MultiEEAffordance.")
    parser.add_argument(
        "--samples",
        default="processed/metadata/samples.jsonl",
        help="samples.jsonl path, relative to --dataset-root unless absolute.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Optionally validate only the first K non-empty metadata rows.",
    )
    parser.add_argument(
        "--split-dir",
        default="splits",
        help="Split directory, relative to --dataset-root unless absolute.",
    )
    parser.add_argument(
        "--skip-split-check",
        action="store_true",
        help="Skip consistency checks between samples.jsonl and split txt files.",
    )
    return parser.parse_args()


def resolve_path(dataset_root: Path, path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else dataset_root / path


def add_issue(issues: list[Issue], level: str, line_no: int, sample: dict[str, Any] | None, message: str) -> None:
    object_id = ""
    if isinstance(sample, dict):
        object_id = str(sample.get("object_id", ""))
    issues.append(Issue(level, line_no, object_id, message))


def read_samples(samples_path: Path, issues: list[Issue], max_samples: int | None) -> list[tuple[int, dict[str, Any]]]:
    if not samples_path.exists():
        add_issue(issues, "ERROR", 0, None, f"samples file does not exist: {samples_path}")
        return []

    samples: list[tuple[int, dict[str, Any]]] = []
    with samples_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                sample = json.loads(stripped)
            except json.JSONDecodeError as exc:
                add_issue(issues, "ERROR", line_no, None, f"invalid JSON: {exc}")
                continue
            if not isinstance(sample, dict):
                add_issue(issues, "ERROR", line_no, None, "each JSONL row must be an object")
                continue
            samples.append((line_no, sample))
            if max_samples is not None and len(samples) >= max_samples:
                break
    return samples


def require_fields(sample: dict[str, Any], line_no: int, issues: list[Issue]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in sample]
    if missing:
        add_issue(issues, "ERROR", line_no, sample, f"missing required fields: {missing}")


def check_enum(value: Any, allowed: set[str], field: str, sample: dict[str, Any], line_no: int, issues: list[Issue]) -> None:
    if value not in allowed:
        add_issue(issues, "ERROR", line_no, sample, f"{field} must be one of {sorted(allowed)}, got {value!r}")


def check_executor_order(sample: dict[str, Any], line_no: int, issues: list[Issue]) -> None:
    if sample.get("executor_order") != EXECUTOR_ORDER:
        add_issue(issues, "ERROR", line_no, sample, f"executor_order must be {EXECUTOR_ORDER}")


def check_executor_objects(sample: dict[str, Any], line_no: int, issues: list[Issue]) -> None:
    feasibility = sample.get("feasibility")
    label_source = sample.get("label_source")
    negative_reason = sample.get("negative_reason")

    for field, value in (
        ("feasibility", feasibility),
        ("label_source", label_source),
        ("negative_reason", negative_reason),
    ):
        if not isinstance(value, dict):
            add_issue(issues, "ERROR", line_no, sample, f"{field} must be an object keyed by executor")
            return
        keys = set(value.keys())
        expected = set(EXECUTOR_ORDER)
        if keys != expected:
            add_issue(issues, "ERROR", line_no, sample, f"{field} keys must be {EXECUTOR_ORDER}, got {sorted(keys)}")

    if not isinstance(feasibility, dict) or not isinstance(label_source, dict) or not isinstance(negative_reason, dict):
        return

    for executor in EXECUTOR_ORDER:
        feasible = feasibility.get(executor)
        if not isinstance(feasible, bool):
            add_issue(issues, "ERROR", line_no, sample, f"feasibility.{executor} must be boolean")

        source = label_source.get(executor)
        if source not in LABEL_SOURCES:
            add_issue(issues, "ERROR", line_no, sample, f"label_source.{executor} must be one of {sorted(LABEL_SOURCES)}")

        reason = negative_reason.get(executor)
        if feasible is False and not (isinstance(reason, str) and reason.strip()):
            add_issue(issues, "ERROR", line_no, sample, f"negative_reason.{executor} is required when infeasible")


def load_npy_array(path: Path, sample: dict[str, Any], line_no: int, issues: list[Issue], field: str) -> np.ndarray | None:
    if not path.exists():
        add_issue(issues, "ERROR", line_no, sample, f"{field} does not exist: {path}")
        return None
    try:
        return np.load(path, allow_pickle=False)
    except Exception as exc:
        add_issue(issues, "ERROR", line_no, sample, f"failed to load {field} {path}: {exc}")
        return None


def check_points_and_masks(dataset_root: Path, sample: dict[str, Any], line_no: int, issues: list[Issue]) -> None:
    point_path_value = sample.get("point_cloud_path")
    mask_path_value = sample.get("multi_channel_mask_path")
    if not isinstance(point_path_value, str) or not point_path_value:
        add_issue(issues, "ERROR", line_no, sample, "point_cloud_path must be a non-empty string")
        return
    if not isinstance(mask_path_value, str) or not mask_path_value:
        add_issue(issues, "ERROR", line_no, sample, "multi_channel_mask_path must be a non-empty string")
        return

    points = load_npy_array(resolve_path(dataset_root, point_path_value), sample, line_no, issues, "point_cloud_path")
    masks = load_npy_array(resolve_path(dataset_root, mask_path_value), sample, line_no, issues, "multi_channel_mask_path")
    if points is None or masks is None:
        return

    if points.ndim != 2 or points.shape[1] not in (3, 6):
        add_issue(issues, "ERROR", line_no, sample, f"points must have shape [N,3] or [N,6], got {points.shape}")
        return
    if masks.ndim != 2:
        add_issue(issues, "ERROR", line_no, sample, f"mask must be 2D with shape [N,4], got {masks.shape}")
        return
    expected_mask_shape = (points.shape[0], len(EXECUTOR_ORDER))
    if masks.shape != expected_mask_shape:
        add_issue(issues, "ERROR", line_no, sample, f"mask shape must be {expected_mask_shape}, got {masks.shape}")
    if not np.issubdtype(masks.dtype, np.number) and masks.dtype != bool:
        add_issue(issues, "ERROR", line_no, sample, f"mask dtype must be numeric or bool, got {masks.dtype}")
    if np.issubdtype(masks.dtype, np.number) and not np.all(np.isin(masks, [0, 1])):
        add_issue(issues, "WARNING", line_no, sample, "mask contains values outside {0, 1}; checker will treat >0 as positive")


def sample_identifier(sample: dict[str, Any]) -> str:
    sample_id = sample.get("sample_id")
    if isinstance(sample_id, str) and sample_id.strip():
        return sample_id.strip()
    object_id = sample.get("object_id", "")
    task = sample.get("task", "")
    return f"{object_id}_{task}".strip("_")


def check_sample(dataset_root: Path, sample: dict[str, Any], line_no: int, issues: list[Issue]) -> None:
    require_fields(sample, line_no, issues)
    if any(field not in sample for field in REQUIRED_FIELDS):
        return

    for field in ("object_id", "source_dataset", "object_category", "task", "task_instruction", "quality_flag", "split"):
        if not isinstance(sample.get(field), str) or not sample.get(field):
            add_issue(issues, "ERROR", line_no, sample, f"{field} must be a non-empty string")

    check_enum(sample.get("source_dataset"), SOURCE_DATASETS, "source_dataset", sample, line_no, issues)
    check_enum(sample.get("task"), TASKS, "task", sample, line_no, issues)
    check_enum(sample.get("quality_flag"), QUALITY_FLAGS, "quality_flag", sample, line_no, issues)
    check_enum(sample.get("split"), SPLITS, "split", sample, line_no, issues)
    check_executor_order(sample, line_no, issues)
    check_executor_objects(sample, line_no, issues)
    check_points_and_masks(dataset_root, sample, line_no, issues)


def check_duplicate_sample_ids(samples: list[tuple[int, dict[str, Any]]], issues: list[Issue]) -> None:
    seen: dict[str, int] = {}
    for line_no, sample in samples:
        sample_id = sample_identifier(sample)
        if not sample_id:
            add_issue(issues, "ERROR", line_no, sample, "sample_id is empty and cannot be inferred")
            continue
        if sample_id in seen:
            add_issue(issues, "ERROR", line_no, sample, f"duplicate sample id {sample_id!r}; first seen at line {seen[sample_id]}")
        else:
            seen[sample_id] = line_no


def read_split_ids(path: Path, split: str, issues: list[Issue]) -> list[str]:
    if not path.exists():
        add_issue(issues, "WARNING", 0, {"object_id": split}, f"split file does not exist: {path}")
        return []
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            value = line.strip()
            if value:
                ids.append(value)
    duplicates = sorted({sample_id for sample_id in ids if ids.count(sample_id) > 1})
    if duplicates:
        add_issue(issues, "ERROR", 0, {"object_id": split}, f"split file has duplicate sample ids: {duplicates[:10]}")
    return ids


def check_split_files(dataset_root: Path, split_dir_value: str, samples: list[tuple[int, dict[str, Any]]], issues: list[Issue]) -> None:
    split_dir = resolve_path(dataset_root, split_dir_value)
    expected = {split: set() for split in SPLITS}
    id_to_split: dict[str, str] = {}

    for line_no, sample in samples:
        if "split" not in sample:
            continue
        split = sample.get("split")
        if split not in SPLITS:
            continue
        sample_id = sample_identifier(sample)
        if not sample_id:
            continue
        expected[split].add(sample_id)
        if sample_id in id_to_split:
            add_issue(issues, "ERROR", line_no, sample, f"sample id {sample_id!r} appears more than once in metadata")
        id_to_split[sample_id] = split

    actual_all: dict[str, str] = {}
    for split in sorted(SPLITS):
        actual_ids = read_split_ids(split_dir / f"{split}.txt", split, issues)
        actual = set(actual_ids)
        missing = sorted(expected[split] - actual)
        extra = sorted(actual - expected[split])
        if missing:
            add_issue(issues, "ERROR", 0, {"object_id": split}, f"split file is missing sample ids: {missing[:10]}")
        if extra:
            add_issue(issues, "ERROR", 0, {"object_id": split}, f"split file contains unexpected sample ids: {extra[:10]}")
        for sample_id in actual:
            previous = actual_all.get(sample_id)
            if previous is not None:
                add_issue(issues, "ERROR", 0, {"object_id": sample_id}, f"sample id appears in both {previous} and {split} split files")
            actual_all[sample_id] = split


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    samples_path = resolve_path(dataset_root, args.samples)
    issues: list[Issue] = []

    if not dataset_root.exists():
        add_issue(issues, "ERROR", 0, None, f"dataset root does not exist: {dataset_root}")
    samples = read_samples(samples_path, issues, args.max_samples)
    for line_no, sample in samples:
        check_sample(dataset_root, sample, line_no, issues)
    check_duplicate_sample_ids(samples, issues)
    if not args.skip_split_check and args.max_samples is None:
        check_split_files(dataset_root, args.split_dir, samples, issues)

    errors = [issue for issue in issues if issue.level == "ERROR"]
    warnings = [issue for issue in issues if issue.level == "WARNING"]

    for issue in issues:
        print(issue.format(), file=sys.stderr if issue.level == "ERROR" else sys.stdout)

    if not samples and not errors:
        print(f"WARNING: no samples found in {samples_path}")

    print(
        json.dumps(
            {
                "samples_checked": len(samples),
                "errors": len(errors),
                "warnings": len(warnings),
            },
            indent=2,
        )
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
