#!/usr/bin/env python3
"""Generate initial four-channel weak affordance masks.

This script intentionally does not use a learned model or an LLM to create
point-wise labels. It combines candidate executor masks, semantic part regions,
manual overrides, and optional simple geometry rules.

Supported candidate formats:

1. NPZ with executor keys:
   gripper, suction, hook, dexterous_hand

2. NPZ with semantic region keys:
   handle, flat_panel, button, ring, hole, grasp, hold, ...
   The script maps region names to executor channels with conservative rules.

3. JSON:
   {
     "executors": {
       "gripper": {"indices": [0, 1, 2]},
       "suction": {"mask_path": "flat_panel_mask.npy"}
     },
     "regions": [
       {
         "name": "handle_outer",
         "indices_path": "handle_indices.npy",
         "tasks": ["pick_up", "open_pull"],
         "executors": ["gripper", "dexterous_hand"]
       }
     ]
   }

All paths inside a JSON candidate file are resolved relative to that JSON file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


EXECUTOR_ORDER = ["gripper", "suction", "hook", "dexterous_hand"]
TASKS = ["pick_up", "lift_carry", "open_pull", "press_push"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate [N, 4] weak affordance masks for Multi-EE Affordance Dataset v0.1."
    )
    parser.add_argument("--points", required=True, help="Input point cloud .npy with shape [N,3] or [N,6].")
    parser.add_argument("--output", required=True, help="Output multi-channel mask .npy with shape [N,4].")
    parser.add_argument("--task", required=True, choices=TASKS, help="Task name for task-conditioned weak rules.")
    parser.add_argument(
        "--candidate",
        help="Optional candidate region file: .json, .npz, or .npy. See module docstring for supported layouts.",
    )
    parser.add_argument(
        "--candidate-executor",
        choices=EXECUTOR_ORDER,
        help="Executor name for a single-channel .npy candidate mask or index file.",
    )
    parser.add_argument(
        "--manual-overrides",
        help="Optional JSON file with add/remove overrides per executor.",
    )
    parser.add_argument(
        "--enable-geometry-rules",
        action="store_true",
        help="Add a simple normal-consistency rule for suction. Requires [N,6] points.",
    )
    parser.add_argument(
        "--suction-normal-threshold",
        type=float,
        default=0.95,
        help="Minimum absolute alignment of a normal with a coordinate axis for suction geometry rule.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output mask file.",
    )
    return parser.parse_args()


def error(message: str) -> None:
    raise ValueError(message)


def load_points(path: Path) -> np.ndarray:
    if not path.exists():
        error(f"Point cloud file does not exist: {path}")
    points = np.load(path)
    if points.ndim != 2:
        error(f"Point cloud must be a 2D array, got shape {points.shape} from {path}")
    if points.shape[1] not in (3, 6):
        error(f"Point cloud must have 3 or 6 channels, got shape {points.shape} from {path}")
    if points.shape[0] == 0:
        error(f"Point cloud is empty: {path}")
    return points


def resolve_path(path_like: str | Path, base_dir: Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else base_dir / path


def as_index_mask(indices: Any, n_points: int, description: str) -> np.ndarray:
    arr = np.asarray(indices)
    if arr.size == 0:
        return np.zeros(n_points, dtype=bool)
    arr = arr.reshape(-1)
    if not np.issubdtype(arr.dtype, np.integer):
        if not np.all(np.isfinite(arr)):
            error(f"{description} contains non-finite indices")
        rounded = np.rint(arr)
        if not np.allclose(arr, rounded):
            error(f"{description} must contain integer point indices")
        arr = rounded.astype(np.int64)
    else:
        arr = arr.astype(np.int64, copy=False)
    if np.any(arr < 0) or np.any(arr >= n_points):
        error(f"{description} contains indices outside [0, {n_points - 1}]")
    mask = np.zeros(n_points, dtype=bool)
    mask[arr] = True
    return mask


def as_binary_mask(values: Any, n_points: int, description: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.reshape(-1)
    if arr.shape != (n_points,):
        error(f"{description} mask must have shape [{n_points}], got {arr.shape}")
    if arr.dtype == bool:
        return arr.astype(bool, copy=False)
    if not np.issubdtype(arr.dtype, np.number):
        error(f"{description} mask must be numeric or boolean")
    finite = np.isfinite(arr)
    if not np.all(finite):
        error(f"{description} mask contains NaN or Inf")
    return arr > 0


def array_to_mask(values: Any, n_points: int, description: str, preferred_kind: str | None = None) -> np.ndarray:
    arr = np.asarray(values)
    if preferred_kind == "mask":
        return as_binary_mask(arr, n_points, description)
    if preferred_kind == "indices":
        return as_index_mask(arr, n_points, description)

    if arr.dtype == bool:
        return as_binary_mask(arr, n_points, description)
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.reshape(-1)
    if arr.shape == (n_points,) and np.issubdtype(arr.dtype, np.number):
        finite = np.isfinite(arr)
        if np.all(finite) and np.all(np.isin(arr, [0, 1])):
            return arr.astype(bool)
    return as_index_mask(arr, n_points, description)


def load_array_file(path: Path, n_points: int, description: str, preferred_kind: str | None = None) -> np.ndarray:
    if not path.exists():
        error(f"{description} file does not exist: {path}")

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            values = json.load(f)
        return array_to_mask(values, n_points, description, preferred_kind)

    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        keys = list(loaded.files)
        if preferred_kind == "mask" and "mask" in loaded:
            return array_to_mask(loaded["mask"], n_points, description, "mask")
        if preferred_kind == "indices" and "indices" in loaded:
            return array_to_mask(loaded["indices"], n_points, description, "indices")
        if len(keys) != 1:
            error(f"{description} NPZ must contain one array or a 'mask'/'indices' key, got keys {keys}")
        return array_to_mask(loaded[keys[0]], n_points, description, preferred_kind)

    return array_to_mask(loaded, n_points, description, preferred_kind)


def mask_from_spec(spec: Any, n_points: int, base_dir: Path, description: str) -> np.ndarray:
    if isinstance(spec, dict):
        if "mask" in spec:
            return array_to_mask(spec["mask"], n_points, description, "mask")
        if "indices" in spec:
            return array_to_mask(spec["indices"], n_points, description, "indices")
        if "mask_path" in spec:
            return load_array_file(resolve_path(spec["mask_path"], base_dir), n_points, description, "mask")
        if "indices_path" in spec:
            return load_array_file(resolve_path(spec["indices_path"], base_dir), n_points, description, "indices")
        if "path" in spec:
            return load_array_file(resolve_path(spec["path"], base_dir), n_points, description)
        if "array" in spec:
            return array_to_mask(spec["array"], n_points, description)
        error(f"{description} must include one of mask, indices, mask_path, indices_path, path, or array")

    if isinstance(spec, (list, tuple, str)):
        if isinstance(spec, str):
            return load_array_file(resolve_path(spec, base_dir), n_points, description)
        return array_to_mask(spec, n_points, description)

    error(f"Unsupported candidate spec for {description}: {type(spec).__name__}")


def normalize_name(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def has_any(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def ordered_executors(executors: Iterable[str]) -> list[str]:
    selected = set(executors)
    unknown = selected.difference(EXECUTOR_ORDER)
    if unknown:
        error(f"Unknown executor names: {sorted(unknown)}")
    return [executor for executor in EXECUTOR_ORDER if executor in selected]


def normalize_executor_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        error(f"executors must be a string or list of strings, got {type(value).__name__}")
    return ordered_executors([str(item) for item in value])


def region_tasks_match(region: dict[str, Any], task: str) -> bool:
    tasks = region.get("tasks", region.get("task"))
    if tasks is None:
        return True
    if isinstance(tasks, str):
        tasks = [tasks]
    if not isinstance(tasks, list):
        error("Region 'tasks' must be a string or list of strings")
    unknown = set(tasks).difference(TASKS)
    if unknown:
        error(f"Unknown task names in region spec: {sorted(unknown)}")
    return task in tasks


def infer_executors_from_region(region_name: str, task: str) -> list[str]:
    """Map semantic region names to executor channels.

    These rules are intentionally conservative. They are weak-label priors,
    not final annotation truth.
    """

    name = normalize_name(region_name)
    executors: set[str] = set()

    hookable = has_any(name, ["hole", "inner", "ring", "loop", "gap", "hook", "hanger", "bail", "arch"])
    handle_like = has_any(name, ["handle", "pull", "grasp", "hold", "knob", "lever", "stem", "neck"])
    flat_like = has_any(name, ["flat", "plane", "panel", "surface", "face", "top", "front", "lid", "plate_center"])
    body_like = has_any(name, ["body", "cup", "bottle", "cylinder", "shaft", "rim", "edge", "lip", "side"])
    press_like = has_any(name, ["button", "switch", "key", "trigger", "press", "push", "pad"])

    if task in ("pick_up", "lift_carry"):
        if handle_like or body_like:
            executors.update(["gripper", "dexterous_hand"])
        if flat_like:
            executors.add("suction")
        if hookable:
            executors.add("hook")

    if task == "open_pull":
        if handle_like or has_any(name, ["edge", "recess"]):
            executors.update(["gripper", "dexterous_hand"])
        if flat_like and has_any(name, ["panel", "front", "door", "drawer", "lid"]):
            executors.add("suction")
        if hookable:
            executors.add("hook")

    if task == "press_push":
        if press_like:
            executors.add("dexterous_hand")
        if flat_like and not hookable:
            executors.add("suction")

    return ordered_executors(executors)


def add_region_to_masks(
    masks: np.ndarray,
    region_mask: np.ndarray,
    executors: list[str],
) -> None:
    for executor in executors:
        channel = EXECUTOR_ORDER.index(executor)
        masks[:, channel] = np.logical_or(masks[:, channel], region_mask)


def apply_direct_executor_specs(data: dict[str, Any], masks: np.ndarray, n_points: int, base_dir: Path) -> bool:
    applied = False
    containers: list[dict[str, Any]] = []
    containers.append(data)
    for key in ("executors", "executor_masks", "masks"):
        value = data.get(key)
        if isinstance(value, dict):
            containers.append(value)

    for container in containers:
        for executor in EXECUTOR_ORDER:
            if executor not in container:
                continue
            region_mask = mask_from_spec(
                container[executor],
                n_points,
                base_dir,
                f"direct mask for executor '{executor}'",
            )
            masks[:, EXECUTOR_ORDER.index(executor)] = np.logical_or(
                masks[:, EXECUTOR_ORDER.index(executor)], region_mask
            )
            applied = True
    return applied


def iter_json_regions(data: dict[str, Any]) -> Iterable[tuple[str, Any]]:
    regions = data.get("regions")
    if regions is None:
        known_top_level = set(EXECUTOR_ORDER + ["executors", "executor_masks", "masks", "metadata"])
        for key, value in data.items():
            if key not in known_top_level:
                yield key, value
        return

    if isinstance(regions, list):
        for i, item in enumerate(regions):
            if not isinstance(item, dict):
                error(f"regions[{i}] must be an object")
            name = item.get("name", f"region_{i}")
            yield str(name), item
        return

    if isinstance(regions, dict):
        for name, item in regions.items():
            yield str(name), item
        return

    error("'regions' must be a list or object")


def load_json_candidates(path: Path, n_points: int, task: str) -> np.ndarray:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        error(f"Candidate JSON root must be an object: {path}")

    masks = np.zeros((n_points, len(EXECUTOR_ORDER)), dtype=bool)
    base_dir = path.parent
    apply_direct_executor_specs(data, masks, n_points, base_dir)

    for name, spec in iter_json_regions(data):
        if isinstance(spec, dict):
            region = dict(spec)
            if not region_tasks_match(region, task):
                continue
            executors = region.get("executors", region.get("executor"))
            if executors is None:
                executors = infer_executors_from_region(name, task)
            else:
                executors = normalize_executor_list(executors)
            if not executors:
                continue
            region_mask = mask_from_spec(region, n_points, base_dir, f"region '{name}'")
        else:
            executors = infer_executors_from_region(name, task)
            if not executors:
                continue
            region_mask = mask_from_spec(spec, n_points, base_dir, f"region '{name}'")
        add_region_to_masks(masks, region_mask, executors)

    return masks


def load_npz_candidates(path: Path, n_points: int, task: str) -> np.ndarray:
    loaded = np.load(path, allow_pickle=False)
    masks = np.zeros((n_points, len(EXECUTOR_ORDER)), dtype=bool)

    for key in loaded.files:
        normalized = normalize_name(key)
        if normalized in EXECUTOR_ORDER:
            executors = [normalized]
        else:
            executors = infer_executors_from_region(key, task)
        if not executors:
            continue
        region_mask = array_to_mask(loaded[key], n_points, f"NPZ candidate '{key}'")
        add_region_to_masks(masks, region_mask, executors)

    return masks


def load_npy_candidate(path: Path, n_points: int, candidate_executor: str | None) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    masks = np.zeros((n_points, len(EXECUTOR_ORDER)), dtype=bool)
    if arr.ndim == 2 and arr.shape == (n_points, len(EXECUTOR_ORDER)):
        return arr > 0
    if candidate_executor is None:
        error("--candidate-executor is required when --candidate is a single-channel .npy file")
    region_mask = array_to_mask(arr, n_points, f"single-channel candidate '{path.name}'")
    masks[:, EXECUTOR_ORDER.index(candidate_executor)] = region_mask
    return masks


def load_candidates(path: Path, n_points: int, task: str, candidate_executor: str | None) -> np.ndarray:
    if not path.exists():
        error(f"Candidate file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_json_candidates(path, n_points, task)
    if suffix == ".npz":
        return load_npz_candidates(path, n_points, task)
    if suffix == ".npy":
        return load_npy_candidate(path, n_points, candidate_executor)
    error(f"Unsupported candidate file extension: {path.suffix}")


def suction_geometry_mask(points: np.ndarray, threshold: float) -> np.ndarray:
    if points.shape[1] != 6:
        print(
            "WARNING: --enable-geometry-rules was set, but point cloud has no normals; "
            "skipping suction geometry rule.",
            file=sys.stderr,
        )
        return np.zeros(points.shape[0], dtype=bool)
    if not 0.0 <= threshold <= 1.0:
        error("--suction-normal-threshold must be in [0, 1]")

    normals = points[:, 3:6].astype(np.float64, copy=False)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-8
    unit = np.zeros_like(normals, dtype=np.float64)
    unit[valid] = normals[valid] / lengths[valid, None]

    # A minimal low-curvature proxy: normals that are strongly aligned with one
    # coordinate axis. This is a weak prior for flat manufactured surfaces.
    axis_alignment = np.max(np.abs(unit), axis=1)
    return valid & (axis_alignment >= threshold)


def apply_manual_overrides(path: Path, masks: np.ndarray, n_points: int) -> None:
    if not path.exists():
        error(f"Manual override file does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        error("Manual override JSON root must be an object")

    for executor, spec in data.items():
        if executor not in EXECUTOR_ORDER:
            error(f"Unknown executor in manual overrides: {executor}")
        if not isinstance(spec, dict):
            error(f"Manual override for {executor} must be an object")
        channel = EXECUTOR_ORDER.index(executor)

        for key in ("set", "set_mask", "set_indices", "set_mask_path", "set_indices_path"):
            if key in spec:
                masks[:, channel] = mask_from_spec(spec[key], n_points, path.parent, f"{executor}.{key}")

        for key in ("add", "add_mask", "add_indices", "add_mask_path", "add_indices_path"):
            if key in spec:
                add_mask = mask_from_spec(spec[key], n_points, path.parent, f"{executor}.{key}")
                masks[:, channel] = np.logical_or(masks[:, channel], add_mask)

        for key in ("remove", "remove_mask", "remove_indices", "remove_mask_path", "remove_indices_path"):
            if key in spec:
                remove_mask = mask_from_spec(spec[key], n_points, path.parent, f"{executor}.{key}")
                masks[:, channel] = np.logical_and(masks[:, channel], ~remove_mask)


def main() -> int:
    args = parse_args()
    try:
        points_path = Path(args.points)
        output_path = Path(args.output)
        if output_path.exists() and not args.overwrite:
            error(f"Output already exists. Use --overwrite to replace it: {output_path}")

        points = load_points(points_path)
        n_points = points.shape[0]
        masks = np.zeros((n_points, len(EXECUTOR_ORDER)), dtype=bool)

        if args.candidate:
            candidate_masks = load_candidates(Path(args.candidate), n_points, args.task, args.candidate_executor)
            if candidate_masks.shape != masks.shape:
                error(f"Candidate masks must have shape {masks.shape}, got {candidate_masks.shape}")
            masks = np.logical_or(masks, candidate_masks)
        else:
            print("WARNING: no --candidate was provided; starting from empty masks.", file=sys.stderr)

        if args.enable_geometry_rules:
            suction_mask = suction_geometry_mask(points, args.suction_normal_threshold)
            masks[:, EXECUTOR_ORDER.index("suction")] = np.logical_or(
                masks[:, EXECUTOR_ORDER.index("suction")], suction_mask
            )

        if args.manual_overrides:
            apply_manual_overrides(Path(args.manual_overrides), masks, n_points)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, masks.astype(np.uint8))

        counts = {executor: int(masks[:, i].sum()) for i, executor in enumerate(EXECUTOR_ORDER)}
        print(json.dumps({"output": str(output_path), "shape": list(masks.shape), "positive_counts": counts}, indent=2))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
