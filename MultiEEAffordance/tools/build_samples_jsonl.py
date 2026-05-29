#!/usr/bin/env python3
"""Build processed/metadata/samples.jsonl from a curated manifest."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.task_taxonomy import EXECUTOR_ORDER, LEGACY_TASKS, task_instruction

TASKS = set(LEGACY_TASKS)
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
MISSING = object()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Multi-EE samples.jsonl from a JSONL or CSV manifest.")
    parser.add_argument("--dataset-root", default=".", help="Dataset root used to resolve relative paths.")
    parser.add_argument("--manifest", required=True, help="Input manifest path: .jsonl or .csv.")
    parser.add_argument(
        "--output",
        default="processed/metadata/samples.jsonl",
        help="Output samples.jsonl path, relative to --dataset-root unless absolute.",
    )
    parser.add_argument(
        "--mask-template",
        default="processed/masks/{sample_id}.npy",
        help="Mask path template used when a manifest row omits multi_channel_mask_path.",
    )
    parser.add_argument("--default-split", default="train", choices=sorted(SPLITS), help="Default split.")
    parser.add_argument("--default-quality", default="weak", choices=sorted(QUALITY_FLAGS), help="Default quality flag.")
    parser.add_argument(
        "--default-label-source",
        default="mixed",
        choices=sorted(LABEL_SOURCES),
        help="Default label source for feasible executors when omitted.",
    )
    parser.add_argument(
        "--default-negative-reason",
        default="missing_candidate_label",
        help="Default negative reason for infeasible executors when omitted.",
    )
    parser.add_argument(
        "--infer-feasibility-from-mask",
        action="store_true",
        help="If feasibility is missing, infer it from existing [N,4] mask positives.",
    )
    parser.add_argument(
        "--strict-files",
        action="store_true",
        help="Require point cloud and mask files to exist while building metadata.",
    )
    parser.add_argument("--append", action="store_true", help="Append to an existing output JSONL instead of replacing it.")
    parser.add_argument("--write-splits", action="store_true", help="Write split txt files from the manifest rows.")
    parser.add_argument("--split-dir", default="splits", help="Split directory, relative to --dataset-root unless absolute.")
    return parser.parse_args()


def error(message: str) -> None:
    raise ValueError(message)


def resolve_path(dataset_root: Path, path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else dataset_root / path


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return None
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"null", "none"}:
        return None
    if value.startswith("{") or value.startswith("["):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def set_nested(row: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return

    flat_mappings = [
        ("feasibility_", "feasibility"),
        ("feasible_", "feasibility"),
        ("label_source_", "label_source"),
        ("negative_reason_", "negative_reason"),
    ]
    for prefix, target in flat_mappings:
        if key.startswith(prefix):
            executor = key[len(prefix) :]
            row.setdefault(target, {})[executor] = value
            return

    if "." in key:
        first, rest = key.split(".", 1)
        row.setdefault(first, {})[rest] = value
        return

    row[key] = value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                error(f"Invalid JSON at {path}:{line_no}: {exc}")
            if not isinstance(row, dict):
                error(f"Each JSONL row must be an object at {path}:{line_no}")
            rows.append(row)
    return rows


def read_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            error(f"CSV manifest has no header: {path}")
        for csv_row in reader:
            row: dict[str, Any] = {}
            for key, value in csv_row.items():
                if key is None:
                    continue
                set_nested(row, key.strip(), parse_scalar(value or ""))
            rows.append(row)
    return rows


def read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        error(f"Manifest does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl(path)
    if suffix == ".csv":
        return read_csv(path)
    error(f"Unsupported manifest extension: {path.suffix}. Use .jsonl or .csv")


def sanitize_id(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_")


def ensure_executor_object(
    value: Any,
    field: str,
    row_id: str,
    fill_value: Any = MISSING,
) -> dict[str, Any]:
    if value is None:
        if fill_value is MISSING:
            error(f"{row_id}: missing {field}")
        value = {executor: fill_value for executor in EXECUTOR_ORDER}
    if not isinstance(value, dict):
        error(f"{row_id}: {field} must be an object keyed by executor")
    unknown = set(value.keys()).difference(EXECUTOR_ORDER)
    if unknown:
        error(f"{row_id}: {field} contains unknown executors {sorted(unknown)}")
    return {executor: value.get(executor) for executor in EXECUTOR_ORDER}


def coerce_bool(value: Any, field: str, row_id: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower == "true":
            return True
        if lower == "false":
            return False
    error(f"{row_id}: {field} must be boolean, got {value!r}")


def infer_feasibility_from_mask(dataset_root: Path, mask_path: str, row_id: str) -> dict[str, bool]:
    path = resolve_path(dataset_root, mask_path)
    if not path.exists():
        error(f"{row_id}: cannot infer feasibility because mask file does not exist: {path}")
    masks = np.load(path, allow_pickle=False)
    if masks.ndim != 2 or masks.shape[1] != len(EXECUTOR_ORDER):
        error(f"{row_id}: mask must have shape [N,4] to infer feasibility, got {masks.shape}")
    return {executor: bool((masks[:, index] > 0).any()) for index, executor in enumerate(EXECUTOR_ORDER)}


def require_string(sample: dict[str, Any], field: str, row_id: str) -> str:
    value = sample.get(field)
    if not isinstance(value, str) or not value.strip():
        error(f"{row_id}: {field} must be a non-empty string")
    return value.strip()


def normalize_sample(row: dict[str, Any], args: argparse.Namespace, dataset_root: Path, row_index: int) -> dict[str, Any]:
    row_id = f"row {row_index}"
    object_id = require_string(row, "object_id", row_id)
    task = require_string(row, "task", row_id)
    if task not in TASKS:
        error(f"{row_id}: task must be one of {sorted(TASKS)}, got {task!r}")

    sample_id = row.get("sample_id") or f"{object_id}_{task}"
    sample_id = sanitize_id(str(sample_id))
    if not sample_id:
        error(f"{row_id}: sample_id becomes empty after sanitization")
    row_id = sample_id

    source_dataset = require_string(row, "source_dataset", row_id)
    if source_dataset not in SOURCE_DATASETS:
        error(f"{row_id}: source_dataset must be one of {sorted(SOURCE_DATASETS)}, got {source_dataset!r}")

    object_category = require_string(row, "object_category", row_id)
    point_cloud_path = require_string(row, "point_cloud_path", row_id)
    mask_path = row.get("multi_channel_mask_path")
    if not isinstance(mask_path, str) or not mask_path.strip():
        mask_path = args.mask_template.format(sample_id=sample_id, object_id=object_id, task=task)

    split = str(row.get("split") or args.default_split)
    if split not in SPLITS:
        error(f"{row_id}: split must be one of {sorted(SPLITS)}, got {split!r}")
    quality_flag = str(row.get("quality_flag") or args.default_quality)
    if quality_flag not in QUALITY_FLAGS:
        error(f"{row_id}: quality_flag must be one of {sorted(QUALITY_FLAGS)}, got {quality_flag!r}")

    if row.get("feasibility") is None and args.infer_feasibility_from_mask:
        feasibility = infer_feasibility_from_mask(dataset_root, str(mask_path), row_id)
    else:
        feasibility_raw = ensure_executor_object(row.get("feasibility"), "feasibility", row_id)
        feasibility = {
            executor: coerce_bool(feasibility_raw[executor], f"feasibility.{executor}", row_id)
            for executor in EXECUTOR_ORDER
        }

    label_source_raw = ensure_executor_object(row.get("label_source"), "label_source", row_id, fill_value=None)
    label_source: dict[str, str] = {}
    for executor in EXECUTOR_ORDER:
        source = label_source_raw.get(executor)
        if source is None or source == "":
            source = args.default_label_source if feasibility[executor] else "unavailable"
        source = str(source)
        if source not in LABEL_SOURCES:
            error(f"{row_id}: label_source.{executor} must be one of {sorted(LABEL_SOURCES)}, got {source!r}")
        label_source[executor] = source

    negative_raw = ensure_executor_object(row.get("negative_reason"), "negative_reason", row_id, fill_value=None)
    negative_reason: dict[str, str | None] = {}
    for executor in EXECUTOR_ORDER:
        reason = negative_raw.get(executor)
        if feasibility[executor]:
            negative_reason[executor] = None if reason in ("", None) else str(reason)
        else:
            if reason is None or str(reason).strip() == "":
                reason = args.default_negative_reason
            negative_reason[executor] = str(reason)

    sample = {
        "sample_id": sample_id,
        "object_id": object_id,
        "source_dataset": source_dataset,
        "object_category": object_category,
        "task": task,
        "task_instruction": row.get("task_instruction") or task_instruction(task),
        "point_cloud_path": point_cloud_path,
        "multi_channel_mask_path": str(mask_path),
        "executor_order": EXECUTOR_ORDER,
        "feasibility": feasibility,
        "label_source": label_source,
        "negative_reason": negative_reason,
        "quality_flag": quality_flag,
        "split": split,
    }
    for optional_field in ("candidate_region_path", "part_annotation_path", "notes"):
        if row.get(optional_field) not in (None, ""):
            sample[optional_field] = row[optional_field]

    if args.strict_files:
        for field in ("point_cloud_path", "multi_channel_mask_path"):
            path = resolve_path(dataset_root, sample[field])
            if not path.exists():
                error(f"{row_id}: {field} does not exist: {path}")

    return sample


def write_samples(path: Path, samples: list[dict[str, Any]], append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8", newline="\n") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def write_splits(dataset_root: Path, split_dir_value: str, samples: list[dict[str, Any]]) -> None:
    split_dir = resolve_path(dataset_root, split_dir_value)
    split_dir.mkdir(parents=True, exist_ok=True)
    grouped = {split: [] for split in SPLITS}
    for sample in samples:
        grouped[sample["split"]].append(sample["sample_id"])
    for split, sample_ids in grouped.items():
        path = split_dir / f"{split}.txt"
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for sample_id in sample_ids:
                f.write(sample_id)
                f.write("\n")


def main() -> int:
    args = parse_args()
    try:
        dataset_root = Path(args.dataset_root)
        manifest_path = resolve_path(dataset_root, args.manifest)
        output_path = resolve_path(dataset_root, args.output)

        rows = read_manifest(manifest_path)
        if not rows:
            error(f"Manifest contains no rows: {manifest_path}")

        samples = [normalize_sample(row, args, dataset_root, index) for index, row in enumerate(rows, start=1)]
        sample_ids = [sample["sample_id"] for sample in samples]
        duplicates = sorted({sample_id for sample_id in sample_ids if sample_ids.count(sample_id) > 1})
        if duplicates:
            error(f"Duplicate sample_id values in manifest: {duplicates}")

        write_samples(output_path, samples, args.append)
        if args.write_splits:
            write_splits(dataset_root, args.split_dir, samples)

        summary = {
            "manifest": str(manifest_path),
            "output": str(output_path),
            "samples_written": len(samples),
            "splits": {split: sum(1 for sample in samples if sample["split"] == split) for split in sorted(SPLITS)},
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
