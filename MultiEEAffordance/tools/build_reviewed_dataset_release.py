#!/usr/bin/env python3
"""Build a reviewed dataset release from point-level human review outputs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.task_taxonomy import ALL_TASKS, EXECUTOR_ORDER, NEW_DEFAULT_ACTIVE_TASKS


KNOWN_TASKS = set(ALL_TASKS)
DEFAULT_ACTIVE_TASKS = set(NEW_DEFAULT_ACTIVE_TASKS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a first reviewed Multi-EE dataset release.")
    parser.add_argument("--dataset-root", default=".", help="Dataset root directory.")
    parser.add_argument(
        "--reviewed-samples",
        default=(
            "processed/metadata/v2_manual_refined_samples_v0_1.jsonl,"
            "processed/metadata/v3_manual_refined_samples_v0_1.jsonl"
        ),
        help="Comma-separated reviewed sample JSONL paths relative to dataset root.",
    )
    parser.add_argument(
        "--output-samples",
        default="processed/metadata/reviewed_dataset_v0_1.jsonl",
        help="Output reviewed dataset JSONL relative to dataset root.",
    )
    parser.add_argument(
        "--summary-json",
        default="processed/metadata/reviewed_dataset_summary_v0_1.json",
        help="Output summary JSON relative to dataset root.",
    )
    parser.add_argument(
        "--output-split-dir",
        default="splits_reviewed_v0_1",
        help="Output split directory relative to dataset root.",
    )
    parser.add_argument(
        "--include-tasks",
        default=",".join(NEW_DEFAULT_ACTIVE_TASKS),
        help="Comma-separated tasks to keep, or 'all'. Default is the five-task review taxonomy.",
    )
    parser.add_argument("--exclude-tasks", default="", help="Comma-separated tasks to drop after include filtering.")
    parser.add_argument(
        "--review-statuses",
        default="checked,verified",
        help="Allowed point_review_status values. Use 'all' to disable this filter.",
    )
    parser.add_argument(
        "--quality-flags",
        default="checked,verified",
        help="Allowed quality_flag values. Use 'all' to disable this filter.",
    )
    parser.add_argument(
        "--split-policy",
        choices=["preserve", "all-val"],
        default="preserve",
        help="Preserve sample split values or put all reviewed samples into val.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def relative_to_dataset(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def write_json(path: Path, data: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def parse_task_filter(value: str, allow_all: bool) -> set[str] | None:
    raw = str(value or "").strip()
    if not raw:
        return set()
    if raw.lower() == "all":
        if allow_all:
            return None
        raise ValueError("'all' is only valid for include filters")
    tasks = set(parse_list(raw))
    unknown = sorted(tasks.difference(KNOWN_TASKS))
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}. Known tasks: {sorted(KNOWN_TASKS)}")
    return tasks


def parse_value_filter(value: str) -> set[str] | None:
    raw = str(value or "").strip()
    if raw.lower() == "all":
        return None
    return set(parse_list(raw))


def sample_key(row: dict[str, Any]) -> str:
    explicit = str(row.get("row_key") or "").strip()
    if explicit:
        return explicit
    update = row.get("v2_candidate_update", {}) if isinstance(row.get("v2_candidate_update"), dict) else {}
    point_edit = row.get("v2_point_edit", {}) if isinstance(row.get("v2_point_edit"), dict) else {}
    executor = str(point_edit.get("executor") or update.get("executor") or row.get("executor") or "").strip()
    return "|".join(part for part in (str(row.get("pilot_id") or ""), str(row.get("sample_id") or ""), executor) if part)


def load_reviewed_rows(root: Path, values: str) -> tuple[list[dict[str, Any]], list[str]]:
    rows_by_key: dict[str, dict[str, Any]] = {}
    sources: list[str] = []
    for item in parse_list(values):
        path = resolve_path(root, item)
        if not path.exists():
            continue
        sources.append(relative_to_dataset(root, path))
        for row in read_jsonl(path):
            key = sample_key(row)
            if key:
                rows_by_key[key] = row
    return list(rows_by_key.values()), sources


def load_mask_count(root: Path, row: dict[str, Any]) -> tuple[int | None, str | None]:
    value = str(row.get("multi_channel_mask_path") or "")
    if not value:
        return None, "missing multi_channel_mask_path"
    path = resolve_path(root, value)
    if not path.exists():
        return None, f"mask not found: {value}"
    try:
        mask = np.load(path, allow_pickle=False)
    except Exception as exc:
        return None, f"mask load failed: {value} ({exc})"
    update = row.get("v2_candidate_update", {}) if isinstance(row.get("v2_candidate_update"), dict) else {}
    point_edit = row.get("v2_point_edit", {}) if isinstance(row.get("v2_point_edit"), dict) else {}
    executor = str(point_edit.get("executor") or update.get("executor") or row.get("executor") or "")
    if executor not in EXECUTOR_ORDER:
        return None, f"unknown executor: {executor}"
    if mask.ndim == 2 and mask.shape[1] == len(EXECUTOR_ORDER):
        return int((mask[:, EXECUTOR_ORDER.index(executor)] > 0).sum()), None
    if mask.ndim == 1:
        return int((mask > 0).sum()), None
    return None, f"bad mask shape: {mask.shape}"


def passes_filters(
    row: dict[str, Any],
    include_tasks: set[str] | None,
    exclude_tasks: set[str],
    statuses: set[str] | None,
    qualities: set[str] | None,
) -> tuple[bool, str]:
    task = str(row.get("task") or "")
    if include_tasks is not None and task not in include_tasks:
        return False, f"task_not_included:{task}"
    if task in exclude_tasks:
        return False, f"task_excluded:{task}"
    status = str(row.get("point_review_status") or "")
    if statuses is not None and status not in statuses:
        return False, f"status_filtered:{status or '<empty>'}"
    quality = str(row.get("quality_flag") or "")
    if qualities is not None and quality not in qualities:
        return False, f"quality_filtered:{quality or '<empty>'}"
    if str(row.get("point_review_status") or "") == "reject":
        return False, "review_rejected"
    return True, ""


def build_release(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.dataset_root).resolve()
    include_tasks = parse_task_filter(args.include_tasks, allow_all=True)
    exclude_tasks = parse_task_filter(args.exclude_tasks, allow_all=False) or set()
    statuses = parse_value_filter(args.review_statuses)
    qualities = parse_value_filter(args.quality_flags)
    rows, sources = load_reviewed_rows(root, args.reviewed_samples)
    if not rows:
        raise FileNotFoundError(
            "No reviewed sample rows found. Pass --reviewed-samples or run the annotation app and save reviews first."
        )

    kept: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    invalid: Counter[str] = Counter()
    for row in rows:
        ok, reason = passes_filters(row, include_tasks, exclude_tasks, statuses, qualities)
        if not ok:
            skipped[reason] += 1
            continue
        count, error = load_mask_count(root, row)
        if error is not None:
            invalid[error] += 1
            continue
        release_row = dict(row)
        if args.split_policy == "all-val":
            release_row["split"] = "val"
        elif not release_row.get("split"):
            release_row["split"] = "val"
        release_row["requires_human_review"] = False
        release_row["reviewed_dataset_release"] = {
            "version": "v0_1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": "point_level_human_review",
            "target_positive_points": int(count or 0),
            "task_policy": "five_task_review_default",
        }
        kept.append(release_row)
        if args.limit is not None and len(kept) >= args.limit:
            break

    output_samples = resolve_path(root, args.output_samples)
    write_jsonl(output_samples, kept, args.overwrite)

    split_dir = resolve_path(root, args.output_split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    split_names = ["train", "val", "test", "contrast_test"]
    split_to_ids: dict[str, list[str]] = {name: [] for name in split_names}
    for row in kept:
        split = str(row.get("split") or "val")
        if split not in split_to_ids:
            split = "val"
        split_to_ids[split].append(str(row.get("sample_id") or ""))
    for split, ids in split_to_ids.items():
        path = split_dir / f"{split}.txt"
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Split file exists. Use --overwrite: {path}")
        path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")

    summary = {
        "version": "v0_1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_reviewed_samples": sources,
        "output_samples": relative_to_dataset(root, output_samples),
        "output_split_dir": relative_to_dataset(root, split_dir),
        "task_policy": {
            "include_tasks": sorted(include_tasks) if include_tasks is not None else "all",
            "exclude_tasks": sorted(exclude_tasks),
            "default_active_tasks": sorted(DEFAULT_ACTIVE_TASKS),
        },
        "rows_read": len(rows),
        "rows_written": len(kept),
        "skipped": dict(sorted(skipped.items())),
        "invalid": dict(sorted(invalid.items())),
        "counts_by_task": dict(sorted(Counter(str(row.get("task") or "") for row in kept).items())),
        "counts_by_executor": dict(sorted(Counter(str((row.get("v2_point_edit") or {}).get("executor") or (row.get("v2_candidate_update") or {}).get("executor") or row.get("executor") or "") for row in kept).items())),
        "counts_by_category": dict(sorted(Counter(str(row.get("object_category") or "") for row in kept).items())),
        "split_counts": {split: len(ids) for split, ids in split_to_ids.items()},
    }
    write_json(resolve_path(root, args.summary_json), summary, args.overwrite)
    return summary


def main() -> int:
    args = parse_args()
    summary = build_release(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
