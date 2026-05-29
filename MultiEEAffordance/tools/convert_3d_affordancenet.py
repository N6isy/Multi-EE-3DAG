#!/usr/bin/env python3
"""Convert 3D AffordanceNet full-shape pickle data into the v0.1 prototype format.

This converter reads the original zip file directly. It does not need to
extract multi-GB pickle files into raw/.

Current scope:
  - full-shape train/val pkl entries
  - object-level points.npy with [N,3]
  - original affordance candidate masks as .npz
  - weak four-channel masks for selected tasks
  - a manifest JSONL that can be fed to build_samples_jsonl.py

The mapping from 3D AffordanceNet affordances to Multi-EE channels is a weak
rule set. It is deliberately conservative for hook and dexterous_hand to avoid
turning every touchable or pushable surface into a positive label.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.task_taxonomy import EXECUTOR_ORDER, LEGACY_TASKS, task_instruction

TASKS = list(LEGACY_TASKS)
NEGATIVE_REASONS = {
    "gripper": "no_graspable_region",
    "suction": "no_flat_suction_surface",
    "hook": "no_hookable_structure",
    "dexterous_hand": "ordinary_surface_without_operation_meaning",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert 3D AffordanceNet full-shape data to Multi-EE v0.1 files.")
    parser.add_argument("--dataset-root", default=".", help="Dataset root, e.g. MultiEEAffordance.")
    parser.add_argument(
        "--zip",
        default="raw/3d_affordancenet/full-shape.zip",
        help="Path to full-shape.zip, relative to --dataset-root unless absolute.",
    )
    parser.add_argument("--source-split", default="val", choices=["train", "val"], help="3D AffordanceNet split to read.")
    parser.add_argument(
        "--target-split",
        default="val",
        choices=["train", "val", "test", "contrast_test"],
        help="Split value written into the generated manifest.",
    )
    parser.add_argument(
        "--tasks",
        default="all",
        help=f"Comma-separated legacy tasks or 'all'. Choices: {','.join(TASKS)}.",
    )
    parser.add_argument("--categories", help="Optional comma-separated semantic classes to include.")
    parser.add_argument("--max-objects", type=int, help="Maximum number of objects to convert after filtering.")
    parser.add_argument("--max-per-category", type=int, help="Maximum number of objects per semantic class.")
    parser.add_argument("--points-dir", default="processed/points", help="Output points dir relative to dataset root.")
    parser.add_argument("--candidate-dir", default="processed/candidates", help="Output candidate dir relative to dataset root.")
    parser.add_argument("--mask-dir", default="processed/masks", help="Output mask dir relative to dataset root.")
    parser.add_argument(
        "--manifest",
        default="manifests/3d_affordancenet_full_shape_manifest.jsonl",
        help="Output manifest JSONL path relative to dataset root.",
    )
    parser.add_argument(
        "--min-positive-points",
        type=int,
        default=8,
        help="Minimum positive points required for an executor to be feasible.",
    )
    parser.add_argument(
        "--small-region-max-ratio",
        type=float,
        default=0.35,
        help="Max positive ratio for treating pull/open/grasp masks as handle-like contact regions.",
    )
    parser.add_argument(
        "--hook-max-ratio",
        type=float,
        default=0.15,
        help="Max positive ratio for treating pull/open masks as hook-like weak regions.",
    )
    parser.add_argument(
        "--large-panel-min-ratio",
        type=float,
        default=0.30,
        help="Min positive ratio for treating pull/open masks as suction panel regions.",
    )
    parser.add_argument(
        "--skip-all-negative",
        action="store_true",
        help="Skip object-task rows when all four executor channels are infeasible.",
    )
    parser.add_argument("--summary", help="Optional JSON summary path relative to dataset root.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting generated files.")
    return parser.parse_args()


def error(message: str) -> None:
    raise ValueError(message)


def resolve_path(root: Path, path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else root / path


def parse_tasks(value: str) -> list[str]:
    if value == "all":
        return TASKS[:]
    tasks = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(tasks).difference(TASKS))
    if unknown:
        error(f"Unknown tasks: {unknown}")
    if not tasks:
        error("--tasks cannot be empty")
    return tasks


def parse_categories(value: str | None) -> set[str] | None:
    if not value:
        return None
    categories = {item.strip() for item in value.split(",") if item.strip()}
    return categories or None


def sanitize_id(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_")


def sanitize_key(value: str) -> str:
    key = value.strip().replace(" ", "_")
    key = re.sub(r"[^A-Za-z0-9_]+", "_", key)
    return key.strip("_").lower()


def zip_entry_for_split(source_split: str) -> str:
    return f"full_shape_{source_split}_data.pkl"


def load_full_shape_items(zip_path: Path, source_split: str) -> list[dict[str, Any]]:
    if not zip_path.exists():
        error(f"3D AffordanceNet zip does not exist: {zip_path}")
    entry = zip_entry_for_split(source_split)
    with zipfile.ZipFile(zip_path) as zf:
        if entry not in zf.namelist():
            error(f"Entry {entry!r} not found in {zip_path}. Available: {zf.namelist()}")
        with zf.open(entry) as f:
            data = pickle.load(f, encoding="latin1")
    if not isinstance(data, list):
        error(f"Expected {entry} to contain a list, got {type(data).__name__}")
    return data


def filter_items(
    items: list[dict[str, Any]],
    categories: set[str] | None,
    max_objects: int | None,
    max_per_category: int | None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    per_category: Counter[str] = Counter()
    for item in items:
        category = str(item.get("semantic class", "unknown"))
        if categories is not None and category not in categories:
            continue
        if max_per_category is not None and per_category[category] >= max_per_category:
            continue
        selected.append(item)
        per_category[category] += 1
        if max_objects is not None and len(selected) >= max_objects:
            break
    return selected


def labels_to_masks(label_dict: dict[str, Any], n_points: int) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for name, values in label_dict.items():
        key = sanitize_key(str(name))
        arr = np.asarray(values).reshape(-1)
        if arr.shape != (n_points,):
            error(f"Affordance label {name!r} has shape {np.asarray(values).shape}; expected [{n_points}]")
        masks[key] = arr > 0
    return masks


def union(masks: dict[str, np.ndarray], names: Iterable[str]) -> np.ndarray:
    result: np.ndarray | None = None
    for name in names:
        mask = masks.get(name)
        if mask is None:
            continue
        result = mask.copy() if result is None else np.logical_or(result, mask)
    if result is None:
        first = next(iter(masks.values()))
        result = np.zeros_like(first, dtype=bool)
    return result


def gated(mask: np.ndarray, min_points: int, max_ratio: float | None = None, min_ratio: float | None = None) -> np.ndarray:
    count = int(mask.sum())
    ratio = count / max(1, mask.shape[0])
    if count < min_points:
        return np.zeros_like(mask, dtype=bool)
    if max_ratio is not None and ratio > max_ratio:
        return np.zeros_like(mask, dtype=bool)
    if min_ratio is not None and ratio < min_ratio:
        return np.zeros_like(mask, dtype=bool)
    return mask


def gated_union(
    masks: dict[str, np.ndarray],
    names: Iterable[str],
    min_points: int,
    max_ratio: float | None = None,
    min_ratio: float | None = None,
) -> np.ndarray:
    result: np.ndarray | None = None
    for name in names:
        mask = masks.get(name)
        if mask is None:
            continue
        kept = gated(mask, min_points=min_points, max_ratio=max_ratio, min_ratio=min_ratio)
        result = kept.copy() if result is None else np.logical_or(result, kept)
    if result is None:
        first = next(iter(masks.values()))
        result = np.zeros_like(first, dtype=bool)
    return result


def make_weak_mask(
    affordance_masks: dict[str, np.ndarray],
    task: str,
    min_points: int,
    small_region_max_ratio: float,
    hook_max_ratio: float,
    large_panel_min_ratio: float,
) -> np.ndarray:
    n_points = next(iter(affordance_masks.values())).shape[0]
    mask = np.zeros((n_points, len(EXECUTOR_ORDER)), dtype=bool)

    if task in {"pick_up", "lift_carry"}:
        hand_contact = gated_union(
            affordance_masks,
            ["grasp", "wrap_grasp", "lift"],
            min_points=min_points,
            max_ratio=0.95,
        )
        suction_region = gated_union(
            affordance_masks,
            ["support", "layable"],
            min_points=min_points,
            max_ratio=0.95,
        )
        mask[:, 0] = hand_contact
        mask[:, 1] = suction_region
        mask[:, 3] = hand_contact

    elif task == "open_pull":
        handle_region = gated_union(
            affordance_masks,
            ["pull", "openable"],
            min_points=min_points,
            max_ratio=small_region_max_ratio,
        )
        hook_region = gated_union(
            affordance_masks,
            ["pull", "openable"],
            min_points=min_points,
            max_ratio=hook_max_ratio,
        )
        panel_region = np.logical_or(
            gated_union(affordance_masks, ["pushable"], min_points=min_points, max_ratio=1.0),
            gated_union(
                affordance_masks,
                ["pull", "openable"],
                min_points=min_points,
                min_ratio=large_panel_min_ratio,
                max_ratio=1.0,
            ),
        )
        mask[:, 0] = handle_region
        mask[:, 1] = panel_region
        mask[:, 2] = hook_region
        mask[:, 3] = handle_region

    elif task == "press_push":
        suction_region = gated_union(
            affordance_masks,
            ["pushable"],
            min_points=min_points,
            max_ratio=1.0,
        )
        dex_region = np.logical_or(
            gated_union(affordance_masks, ["press"], min_points=min_points, max_ratio=1.0),
            gated_union(
                affordance_masks,
                ["pushable"],
                min_points=min_points,
                max_ratio=0.25,
            ),
        )
        mask[:, 1] = suction_region
        mask[:, 3] = dex_region

    else:
        error(f"Unsupported task: {task}")

    return mask.astype(np.uint8)


def feasibility_from_mask(mask: np.ndarray, min_points: int) -> dict[str, bool]:
    return {
        executor: bool((mask[:, index] > 0).sum() >= min_points)
        for index, executor in enumerate(EXECUTOR_ORDER)
    }


def label_source_from_feasibility(feasibility: dict[str, bool]) -> dict[str, str]:
    return {
        executor: "existing_affordance_mask" if feasible else "unavailable"
        for executor, feasible in feasibility.items()
    }


def negative_reason_from_feasibility(feasibility: dict[str, bool]) -> dict[str, str | None]:
    return {
        executor: None if feasible else NEGATIVE_REASONS[executor]
        for executor, feasible in feasibility.items()
    }


def save_np(path: Path, array: np.ndarray, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        error(f"Output already exists. Use --overwrite to replace it: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def save_candidates(path: Path, masks: dict[str, np.ndarray], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        error(f"Output already exists. Use --overwrite to replace it: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: value.astype(np.uint8) for key, value in masks.items()}
    np.savez_compressed(path, **arrays)


def relative_to_root(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def convert(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.dataset_root)
    zip_path = resolve_path(root, args.zip)
    tasks = parse_tasks(args.tasks)
    categories = parse_categories(args.categories)
    points_dir = resolve_path(root, args.points_dir)
    candidate_dir = resolve_path(root, args.candidate_dir)
    mask_dir = resolve_path(root, args.mask_dir)
    manifest_path = resolve_path(root, args.manifest)

    if args.max_objects is not None and args.max_objects <= 0:
        error("--max-objects must be positive")
    if args.max_per_category is not None and args.max_per_category <= 0:
        error("--max-per-category must be positive")
    if args.min_positive_points <= 0:
        error("--min-positive-points must be positive")

    items = load_full_shape_items(zip_path, args.source_split)
    selected = filter_items(items, categories, args.max_objects, args.max_per_category)
    if not selected:
        error("No 3D AffordanceNet objects selected after filtering")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    category_counter: Counter[str] = Counter()
    task_counter: Counter[str] = Counter()
    feasible_counter: dict[str, Counter[str]] = {task: Counter() for task in tasks}
    object_ids_seen: set[str] = set()

    with manifest_path.open("w", encoding="utf-8", newline="\n") as manifest_file:
        for item in selected:
            shape_id = str(item["shape_id"])
            category = str(item["semantic class"])
            object_id = sanitize_id(f"3danet_full_{shape_id}")
            if object_id in object_ids_seen:
                error(f"Duplicate object_id generated: {object_id}")
            object_ids_seen.add(object_id)
            category_counter[category] += 1

            full_shape = item.get("full_shape")
            if not isinstance(full_shape, dict):
                error(f"{shape_id}: missing full_shape dict")
            coordinates = np.asarray(full_shape.get("coordinate"), dtype=np.float32)
            if coordinates.ndim != 2 or coordinates.shape[1] != 3:
                error(f"{shape_id}: coordinate must have shape [N,3], got {coordinates.shape}")
            if not np.all(np.isfinite(coordinates)):
                error(f"{shape_id}: coordinate contains NaN or Inf")

            label_dict = full_shape.get("label")
            if not isinstance(label_dict, dict):
                error(f"{shape_id}: full_shape.label must be a dict")
            affordance_masks = labels_to_masks(label_dict, coordinates.shape[0])

            point_path = points_dir / f"{object_id}.npy"
            candidate_path = candidate_dir / f"{object_id}_affordance_candidates.npz"
            save_np(point_path, coordinates, args.overwrite)
            save_candidates(candidate_path, affordance_masks, args.overwrite)

            for task in tasks:
                sample_id = sanitize_id(f"{object_id}_{task}")
                mask = make_weak_mask(
                    affordance_masks,
                    task=task,
                    min_points=args.min_positive_points,
                    small_region_max_ratio=args.small_region_max_ratio,
                    hook_max_ratio=args.hook_max_ratio,
                    large_panel_min_ratio=args.large_panel_min_ratio,
                )
                feasibility = feasibility_from_mask(mask, args.min_positive_points)
                if args.skip_all_negative and not any(feasibility.values()):
                    continue
                mask_path = mask_dir / f"{sample_id}.npy"
                save_np(mask_path, mask, args.overwrite)
                for executor, feasible in feasibility.items():
                    if feasible:
                        feasible_counter[task][executor] += 1
                task_counter[task] += 1

                sample = {
                    "sample_id": sample_id,
                    "object_id": object_id,
                    "source_dataset": "3d_affordancenet",
                    "object_category": category,
                    "task": task,
                    "task_instruction": task_instruction(task),
                    "point_cloud_path": relative_to_root(point_path, root),
                    "multi_channel_mask_path": relative_to_root(mask_path, root),
                    "candidate_region_path": relative_to_root(candidate_path, root),
                    "executor_order": EXECUTOR_ORDER,
                    "feasibility": feasibility,
                    "label_source": label_source_from_feasibility(feasibility),
                    "negative_reason": negative_reason_from_feasibility(feasibility),
                    "quality_flag": "weak",
                    "split": args.target_split,
                    "notes": (
                        "由 3D AffordanceNet full-shape 原始 affordance mask 通过保守规则映射生成；"
                        "hook 和 dexterous_hand 通道需要后续人工检查。"
                    ),
                }
                manifest_file.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")))
                manifest_file.write("\n")

    summary = {
        "zip": str(zip_path),
        "source_split": args.source_split,
        "target_split": args.target_split,
        "objects_loaded": len(items),
        "objects_converted": len(selected),
        "samples_written": sum(task_counter.values()),
        "tasks": dict(task_counter),
        "categories": dict(category_counter),
        "feasible_counts_by_task": {task: dict(counter) for task, counter in feasible_counter.items()},
        "manifest": str(manifest_path),
    }
    if args.summary:
        summary_path = resolve_path(root, args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            f.write("\n")
    return summary


def main() -> int:
    args = parse_args()
    try:
        summary = convert(args)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
