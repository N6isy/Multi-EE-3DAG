#!/usr/bin/env python3
"""Build five-task human-review rows from converted PartNet-Mobility objects.

The converter produces one object-level row, one normalized point cloud, and
URDF-link proposal masks. This adapter creates channel-level review rows for the
five-task annotation UI:

  object proposal -> plausible five-task rows -> four executor variants

Each row starts from an empty [N, 4] mask. URDF-link masks remain proposals for
human review; they are never treated as executor labels or ground truth.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.task_taxonomy import (  # noqa: E402
    EXECUTOR_ORDER,
    NEW_TASKS,
    TASK_TAXONOMY_VERSION,
    task_display,
    task_instruction,
)


PLAUSIBLE_TASKS_BY_CATEGORY: dict[str, tuple[str, ...]] = {
    "box": ("lift", "open", "pull"),
    "bucket": ("lift",),
    "cabinet": ("open", "pull", "push"),
    "camera": ("lift", "press"),
    "coffeemachine": ("press", "push"),
    "dispenser": ("press", "push"),
    "kettle": ("lift", "open"),
    "lighter": ("lift", "press"),
    "mouse": ("lift", "press"),
    "oven": ("open", "pull", "press", "push"),
    "phone": ("lift", "press"),
    "pliers": ("lift", "open", "press"),
    "remote": ("lift", "press"),
    "safe": ("open", "pull", "press", "push"),
    "stapler": ("lift", "press", "push"),
    "suitcase": ("lift", "open", "pull", "push"),
    "switch": ("press", "push"),
    "toaster": ("press", "push"),
    "toilet": ("open", "push"),
    "washingmachine": ("open", "pull", "press", "push"),
    "window": ("open", "pull", "push"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build lift/open/pull/press/push review rows from converted PartNet-Mobility link proposals."
    )
    parser.add_argument("--dataset-root", required=True, help="Dataset root used to resolve portable relative paths.")
    parser.add_argument(
        "--objects-manifest",
        default="manifests/partnet_mobility_v0_supplement_21cat_objects_manifest.jsonl",
        help="Object-level manifest produced by convert_partnet_mobility.py.",
    )
    parser.add_argument(
        "--output-samples",
        default="processed/metadata/partnet_mobility_v0_supplement_21cat_5tasks_review_samples.jsonl",
        help="Output channel-level samples JSONL.",
    )
    parser.add_argument(
        "--summary-json",
        default="processed/metadata/partnet_mobility_v0_supplement_21cat_5tasks_review_summary.json",
        help="Output summary JSON.",
    )
    parser.add_argument(
        "--initial-mask-dir",
        default="processed/masks/partnet_mobility_v0_supplement_21cat_initial_empty",
        help="Directory for reusable initial empty [N,4] masks.",
    )
    parser.add_argument(
        "--task-policy",
        choices=["plausible", "all"],
        default="plausible",
        help="plausible uses category-level high-recall task priors; all emits every five-task combination.",
    )
    parser.add_argument(
        "--tasks",
        default=",".join(NEW_TASKS),
        help="Comma-separated five-task subset to keep, or 'all'.",
    )
    parser.add_argument(
        "--executors",
        default=",".join(EXECUTOR_ORDER),
        help="Comma-separated executor subset to emit, or 'all'.",
    )
    parser.add_argument("--default-split", choices=["train", "val", "test", "contrast_test"], default="train")
    parser.add_argument("--pilot-prefix", default="partnet_review")
    parser.add_argument("--max-objects", type=int, help="Optional object limit for smoke tests.")
    parser.add_argument(
        "--strict-files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate point cloud, candidate manifest, and candidate npz before emitting rows.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing valid initial empty masks.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite JSONL, summary, and initial masks.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print summary without writing files.")
    return parser.parse_args()


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def relative_to_dataset(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Objects manifest does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_no}, got {type(row).__name__}")
            row["_source_line_no"] = line_no
            rows.append(row)
    return rows


def write_json(path: Path, payload: Any, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def sanitize_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_") or "unnamed"


def normalize_category(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def parse_selection(value: str, allowed: list[str], field: str) -> list[str]:
    raw = str(value or "").strip()
    if raw.lower() == "all":
        return list(allowed)
    selected = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(selected).difference(allowed))
    if unknown:
        raise ValueError(f"Unknown {field}: {unknown}. Allowed: {allowed}")
    return [item for item in allowed if item in selected]


def tasks_for_category(category: str, task_policy: str, selected_tasks: list[str]) -> list[str]:
    if task_policy == "all":
        return list(selected_tasks)
    priors = PLAUSIBLE_TASKS_BY_CATEGORY.get(normalize_category(category), ())
    return [task for task in selected_tasks if task in priors]


def load_points(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Point cloud does not exist: {path}")
    points = np.load(path, allow_pickle=False)
    if points.ndim != 2 or points.shape[1] not in (3, 6):
        raise ValueError(f"Expected point cloud shape [N,3] or [N,6], got {points.shape}: {path}")
    if points.shape[0] <= 0:
        raise ValueError(f"Point cloud is empty: {path}")
    return points


def validate_candidate_files(root: Path, row: dict[str, Any], n_points: int) -> tuple[dict[str, Any], Path]:
    manifest_value = str(row.get("candidate_manifest") or "")
    if not manifest_value:
        raise ValueError("Object row is missing candidate_manifest")
    manifest_path = resolve_path(root, manifest_value)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Candidate manifest does not exist: {manifest_path}")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Candidate manifest must be an object: {manifest_path}")
    npz_value = str(manifest.get("candidate_npz") or row.get("candidate_npz") or "")
    if not npz_value:
        raise ValueError(f"Candidate manifest is missing candidate_npz: {manifest_path}")
    npz_path = resolve_path(root, npz_value)
    if not npz_path.exists():
        candidate = manifest_path.parent / npz_value
        npz_path = candidate if candidate.exists() else npz_path
    if not npz_path.exists():
        raise FileNotFoundError(f"Candidate npz does not exist: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    if "candidate_masks" not in data or "candidate_ids" not in data:
        raise ValueError(f"Candidate npz must contain candidate_masks and candidate_ids: {npz_path}")
    masks = data["candidate_masks"]
    ids = data["candidate_ids"]
    if masks.ndim != 2 or masks.shape[1] != n_points:
        raise ValueError(f"Expected candidate_masks shape [K,{n_points}], got {masks.shape}: {npz_path}")
    if ids.ndim != 1 or ids.shape[0] != masks.shape[0]:
        raise ValueError(f"candidate_ids length does not match candidate_masks: {npz_path}")
    return manifest, npz_path


def write_initial_mask(path: Path, n_points: int, overwrite: bool, resume: bool, dry_run: bool) -> str:
    if path.exists() and resume and not overwrite:
        existing = np.load(path, allow_pickle=False)
        if existing.shape != (n_points, len(EXECUTOR_ORDER)):
            raise ValueError(f"Existing initial mask has wrong shape {existing.shape}: {path}")
        if np.any(existing):
            raise ValueError(f"Existing initial mask is not empty: {path}")
        return "reused"
    if path.exists() and not overwrite:
        raise FileExistsError(f"Initial mask exists. Use --overwrite or --resume: {path}")
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.zeros((n_points, len(EXECUTOR_ORDER)), dtype=np.uint8))
    return "written"


def empty_executor_metadata(reason: str) -> tuple[dict[str, bool], dict[str, str], dict[str, str]]:
    return (
        {executor: False for executor in EXECUTOR_ORDER},
        {executor: "unavailable" for executor in EXECUTOR_ORDER},
        {executor: reason for executor in EXECUTOR_ORDER},
    )


def build_review_row(
    *,
    object_row: dict[str, Any],
    task: str,
    executor: str,
    pilot_id: str,
    initial_mask_path: str,
    candidate_manifest_path: str,
    candidate_npz_path: str,
    split: str,
) -> dict[str, Any]:
    object_id = str(object_row["object_id"])
    sample_id = sanitize_id(f"{object_id}_{task}")
    review_id = sanitize_id(f"{sample_id}_{executor}")
    row_key = "|".join([review_id, sample_id, task, executor])
    reason = "partnet_link_proposal_requires_manual_review"
    feasibility, label_source, negative_reason = empty_executor_metadata(reason)
    update = {
        "pilot_id": pilot_id,
        "executor": executor,
        "source": "partnet_mobility_urdf_link_proposal",
        "candidate_manifest": candidate_manifest_path,
        "rule_filter_path": "",
        "selected_candidates": [],
        "positive_points": 0,
        "requires_human_review": True,
        "review_mode": "point_refine",
        "negative_reason": reason,
    }
    return {
        "row_key": row_key,
        "review_id": review_id,
        "pilot_id": pilot_id,
        "task_instance_id": review_id,
        "object_id": object_id,
        "raw_model_id": object_row.get("raw_model_id", ""),
        "source_dataset": "partnet_mobility",
        "object_category": object_row.get("object_category", ""),
        "sample_id": sample_id,
        "task": task,
        "target_task": task,
        "task_display": task_display(task),
        "task_instruction": task_instruction(task),
        "source_task": task,
        "source_sample_id": object_id,
        "task_taxonomy_version": TASK_TAXONOMY_VERSION,
        "task_split_source": "partnet_mobility_direct_5task",
        "point_cloud_path": object_row["point_cloud_path"],
        "multi_channel_mask_path": initial_mask_path,
        "source_mask_path": initial_mask_path,
        "checked_mask_path": initial_mask_path,
        "candidate_manifest": candidate_manifest_path,
        "candidate_npz": candidate_npz_path,
        "executor": executor,
        "target_executor": executor,
        "executor_order": list(EXECUTOR_ORDER),
        "feasibility": feasibility,
        "label_source": label_source,
        "negative_reason": negative_reason,
        "quality_flag": "weak",
        "split": split,
        "review_mode": "point_refine",
        "requires_human_review": True,
        "proposal_only": True,
        "v2_candidate_update": dict(update),
        "v3_candidate_update": dict(update),
        "notes": (
            "PartNet-Mobility URDF link masks are review proposals only. "
            "Reviewer must select relevant parts or refine points manually under the five-task definition."
        ),
    }


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    objects_manifest = resolve_path(root, args.objects_manifest)
    output_samples = resolve_path(root, args.output_samples)
    summary_path = resolve_path(root, args.summary_json)
    initial_mask_dir = resolve_path(root, args.initial_mask_dir)
    selected_tasks = parse_selection(args.tasks, list(NEW_TASKS), "tasks")
    selected_executors = parse_selection(args.executors, list(EXECUTOR_ORDER), "executors")
    object_rows = read_jsonl(objects_manifest)
    if args.max_objects is not None:
        object_rows = object_rows[: args.max_objects]

    review_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    counts_by_category: Counter[str] = Counter()
    counts_by_task: Counter[str] = Counter()
    counts_by_executor: Counter[str] = Counter()
    mask_actions: Counter[str] = Counter()

    for object_index, object_row in enumerate(object_rows, start=1):
        object_id = str(object_row.get("object_id") or "")
        category = str(object_row.get("object_category") or "")
        try:
            if not object_id:
                raise ValueError("Object row is missing object_id")
            point_value = str(object_row.get("point_cloud_path") or "")
            if not point_value:
                raise ValueError("Object row is missing point_cloud_path")
            points_path = resolve_path(root, point_value)
            points = load_points(points_path)
            candidate_manifest, candidate_npz = validate_candidate_files(root, object_row, int(points.shape[0]))
            tasks = tasks_for_category(category, args.task_policy, selected_tasks)
            if not tasks:
                raise ValueError(f"No tasks selected for category {category!r} under task policy {args.task_policy!r}")
            initial_mask = initial_mask_dir / f"{sanitize_id(object_id)}_initial_empty.npy"
            mask_actions[write_initial_mask(initial_mask, int(points.shape[0]), args.overwrite, args.resume, args.dry_run)] += 1
            initial_mask_rel = relative_to_dataset(root, initial_mask)
            candidate_manifest_rel = relative_to_dataset(root, resolve_path(root, str(object_row["candidate_manifest"])))
            candidate_npz_rel = relative_to_dataset(root, candidate_npz)
            for task in tasks:
                for executor in selected_executors:
                    pilot_id = f"{args.pilot_prefix}_{len(review_rows) + 1:06d}"
                    review_rows.append(
                        build_review_row(
                            object_row=object_row,
                            task=task,
                            executor=executor,
                            pilot_id=pilot_id,
                            initial_mask_path=initial_mask_rel,
                            candidate_manifest_path=candidate_manifest_rel,
                            candidate_npz_path=candidate_npz_rel,
                            split=args.default_split,
                        )
                    )
                    counts_by_task[task] += 1
                    counts_by_executor[executor] += 1
            counts_by_category[category] += 1
            if object_index == 1 or object_index == len(object_rows) or object_index % 100 == 0:
                print(f"[partnet-review] {object_index}/{len(object_rows)} objects", flush=True)
        except Exception as exc:
            skipped.append(
                {
                    "object_id": object_id,
                    "object_category": category,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if args.strict_files:
                raise

    summary = {
        "version": "v0_1",
        "pipeline": "partnet_mobility_direct_5task_review_samples",
        "proposal_only": True,
        "dataset_root": str(root),
        "objects_manifest": relative_to_dataset(root, objects_manifest),
        "output_samples": relative_to_dataset(root, output_samples),
        "initial_mask_dir": relative_to_dataset(root, initial_mask_dir),
        "task_taxonomy_version": TASK_TAXONOMY_VERSION,
        "task_policy": args.task_policy,
        "selected_tasks": selected_tasks,
        "selected_executors": selected_executors,
        "objects_read": len(object_rows),
        "objects_emitted": sum(counts_by_category.values()),
        "review_rows_written": len(review_rows),
        "initial_masks": dict(mask_actions),
        "counts_by_category": dict(sorted(counts_by_category.items())),
        "counts_by_task": dict(sorted(counts_by_task.items())),
        "counts_by_executor": dict(sorted(counts_by_executor.items())),
        "skipped_objects": len(skipped),
        "skipped": skipped[:200],
        "dry_run": args.dry_run,
        "notes": [
            "Rows are direct five-task human-review inputs; do not run legacy task expansion on this file.",
            "Initial [N,4] masks are empty. URDF-link candidate masks remain proposals only.",
            "Each selected object-task pair emits one review row per executor so reviewers can confirm valid positives or empty labels.",
        ],
    }
    if not args.dry_run:
        write_jsonl(output_samples, review_rows, args.overwrite)
        write_json(summary_path, summary, args.overwrite)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
