#!/usr/bin/env python3
"""Generate v3 3D part candidates without asking the VLM for coordinates.

This stage is the part-segmentation candidate backbone for the v3 path:

  original point cloud + optional weak masks / external PartSLIP++ segments /
  self-developed high-recall proposals
      -> 3D part candidates [K, N]
      -> VLM selects candidate IDs later

All candidates are binary masks over the original point cloud length N.
The default high_recall backend is model-free and intentionally favors recall
over precision. The hybrid_partslippp_high_recall backend uses PartSLIP++
segments for mapped categories and supplements or falls back to high_recall.
"""

from __future__ import annotations

import argparse
import csv
import json
import string
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

import generate_3d_candidate_regions as gen
from path_utils import relative_to_dataset, resolve_portable_path


LETTERS = list(string.ascii_uppercase)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate v3 part-level 3D candidate proposals.")
    parser.add_argument("--dataset-root", default=".", help="Dataset root directory.")
    parser.add_argument("--pilot-csv", default="processed/metadata/vlm_pilot_samples_v0_1.csv")
    parser.add_argument("--samples", default="processed/metadata/samples_checked_v0_1.jsonl")
    parser.add_argument("--output-root", default="processed/vlm_candidate_v3/3d_candidates")
    parser.add_argument("--semantic-plan-root", default="processed/vlm_candidate_v3/semantic_plans")
    parser.add_argument("--renders-root", default="processed/vlm_semantic_part/renders")
    parser.add_argument("--fallback-renders-root", default="processed/vlm_pilot/renders")
    parser.add_argument(
        "--partslippp-root",
        default="external/partslippp/outputs",
        help="Root containing external PartSLIP++ predictions.",
    )
    parser.add_argument(
        "--partslippp-path",
        default="",
        help=(
            "Optional explicit PartSLIP++ prediction path. Supports {pilot_id}, {sample_id}, "
            "{task}, {executor}, {object_category}, and {partslippp_category} tokens."
        ),
    )
    parser.add_argument(
        "--partslippp-category-map",
        default="configs/partslippp_category_map.json",
        help=(
            "JSON mapping from 3DAffordanceNet object_category to PartSLIP++ category/checkpoint. "
            "Used by hybrid_partslippp_high_recall."
        ),
    )
    parser.add_argument(
        "--partslippp-fallback",
        choices=["error", "geometry"],
        default="error",
        help="Fallback when PartSLIP++ output is missing or malformed.",
    )
    parser.add_argument(
        "--backend",
        choices=["high_recall", "geometry", "partslippp", "hybrid_partslippp_high_recall"],
        default="high_recall",
        help=(
            "Candidate backend. high_recall is the default self-developed model-free generator; "
            "geometry is the old fallback; partslippp reads optional external normalized predictions; "
            "hybrid_partslippp_high_recall routes mapped categories through PartSLIP++ and supplements "
            "or falls back with high_recall."
        ),
    )
    parser.add_argument(
        "--hybrid-min-high-recall-supplement",
        type=int,
        default=8,
        help="Minimum high_recall supplement candidates to reserve when hybrid PartSLIP++ candidates are available.",
    )
    parser.add_argument(
        "--hybrid-max-partslippp-primary",
        type=int,
        default=48,
        help="Maximum PartSLIP++ primary candidates kept before high_recall supplements in hybrid mode.",
    )
    parser.add_argument("--pilot-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--k-neighbors", type=int, default=24)
    parser.add_argument("--min-points", type=int, default=4)
    parser.add_argument("--max-candidate-fraction", type=float, default=0.70)
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--part-top-k", type=int, default=5, help="Maximum candidates kept per high-recall part group before global fill.")
    parser.add_argument("--dedupe-iou", type=float, default=0.985, help="Only remove near-duplicate candidates above this IoU.")
    parser.add_argument("--small-part-max-fraction", type=float, default=0.10, help="Maximum object fraction for small-part component proposals.")
    parser.add_argument("--seed-expand-hops", type=int, default=1)
    parser.add_argument("--component-max-candidates", type=int, default=8)
    parser.add_argument("--view-dilation-radius", type=int, default=3)
    parser.add_argument("--body-row-threshold-ratio", type=float, default=0.18)
    parser.add_argument("--body-top-margin", type=int, default=8)
    parser.add_argument(
        "--allow-empty-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write an empty candidate manifest instead of aborting when no geometry proposal survives.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(root: Path, value: str | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Pilot CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, data: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_category_key(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def load_partslippp_category_map(root: Path, args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    path = resolve_path(root, args.partslippp_category_map)
    if path is None or not path.exists():
        return {}
    payload = read_json(path)
    raw_categories = payload.get("categories", payload)
    if not isinstance(raw_categories, dict):
        raise ValueError(f"PartSLIP++ category map must be a JSON object: {path}")

    mapped: dict[str, dict[str, Any]] = {}
    for category, value in raw_categories.items():
        if isinstance(value, str):
            entry: dict[str, Any] = {"partslippp_category": value, "checkpoint": value, "enabled": True}
        elif isinstance(value, dict):
            entry = dict(value)
            entry.setdefault("partslippp_category", entry.get("category", category))
            entry.setdefault("checkpoint", entry.get("partslippp_category", category))
            entry.setdefault("enabled", True)
        else:
            continue
        if not bool(entry.get("enabled", True)):
            continue
        entry["object_category"] = str(category)
        entry["partslippp_category"] = str(entry.get("partslippp_category") or category)
        entry["checkpoint"] = str(entry.get("checkpoint") or entry["partslippp_category"])
        mapped[normalize_category_key(str(category))] = entry
    return mapped


def partslippp_mapping_for(
    root: Path,
    args: argparse.Namespace,
    row: dict[str, str],
) -> dict[str, Any] | None:
    category = str(row.get("object_category") or "")
    mapping = load_partslippp_category_map(root, args)
    return mapping.get(normalize_category_key(category))


def selected_rows(root: Path, args: argparse.Namespace) -> list[dict[str, str]]:
    rows = read_csv(resolve_path(root, args.pilot_csv))
    if args.pilot_id:
        rows = [row for row in rows if row.get("pilot_id") == args.pilot_id]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No pilot rows selected.")
    return rows


def progress_rows(rows: list[dict[str, str]], desc: str):
    try:
        from tqdm import tqdm

        return tqdm(rows, desc=desc, unit="row")
    except Exception:
        return rows


def semantic_plan_path(root: Path, args: argparse.Namespace, pilot_id: str) -> Path:
    return resolve_path(root, args.semantic_plan_root) / pilot_id / "combined_semantic_plan.json"


def load_optional_plan(root: Path, args: argparse.Namespace, pilot_id: str) -> dict[str, Any] | None:
    path = semantic_plan_path(root, args, pilot_id)
    if path.exists():
        return read_json(path)
    return None


def normalize_meta(meta: dict[str, Any]) -> dict[str, Any]:
    out = dict(meta)
    out["provenance"] = "v3_partseg_geometry_proposal"
    out.setdefault("quality_hint", "v3_partseg_candidate")
    return out


def partslippp_meta(meta: dict[str, Any]) -> dict[str, Any]:
    out = dict(meta)
    out["provenance"] = "v3_partseg_partslippp_adapter"
    out.setdefault("candidate_family", "partslippp_part")
    out.setdefault("quality_hint", "partslippp_candidate")
    out.setdefault("source", "partslippp")
    return out


def empty_candidates(n: int) -> tuple[np.ndarray, list[dict[str, Any]], str]:
    return np.zeros((0, n), dtype=np.uint8), [], "no_candidate_survived"


def coerce_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        if value.size == 1:
            return value.reshape(-1)[0].item()
    return value


def decode_name(value: Any) -> str:
    value = coerce_scalar(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def maybe_object_payload(value: Any) -> Any:
    value = coerce_scalar(value)
    if isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
        return value.item()
    return value


def names_from_payload(payload: Any, k: int, prefix: str) -> list[str]:
    payload = maybe_object_payload(payload)
    if payload is None:
        return [f"{prefix}_{i:02d}" for i in range(k)]
    if isinstance(payload, dict):
        return [decode_name(payload.get(str(i), payload.get(i, f"{prefix}_{i:02d}"))) for i in range(k)]
    if isinstance(payload, np.ndarray):
        payload = payload.tolist()
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        values = list(payload)
        return [decode_name(values[i]) if i < len(values) else f"{prefix}_{i:02d}" for i in range(k)]
    return [f"{prefix}_{i:02d}" for i in range(k)]


def scores_from_payload(payload: Any, k: int) -> list[float | None]:
    payload = maybe_object_payload(payload)
    if payload is None:
        return [None] * k
    if isinstance(payload, dict):
        out: list[float | None] = []
        for i in range(k):
            value = payload.get(str(i), payload.get(i))
            out.append(None if value is None else float(value))
        return out
    arr = np.asarray(payload).reshape(-1)
    return [float(arr[i]) if i < arr.size else None for i in range(k)]


def candidate_id_for(index: int) -> str:
    if index < len(LETTERS):
        return LETTERS[index]
    first = LETTERS[(index // len(LETTERS)) - 1]
    second = LETTERS[index % len(LETTERS)]
    return f"{first}{second}"


def labels_to_masks(labels: np.ndarray, n: int, point_indices: np.ndarray | None = None) -> tuple[np.ndarray, list[Any]]:
    labels = np.asarray(labels).reshape(-1)
    if point_indices is not None:
        point_indices = np.asarray(point_indices, dtype=np.int64).reshape(-1)
        if labels.shape[0] != point_indices.shape[0]:
            raise ValueError("PartSLIP++ labels length does not match point_indices length.")
        full = np.full((n,), -1, dtype=labels.dtype)
        valid = (point_indices >= 0) & (point_indices < n)
        full[point_indices[valid]] = labels[valid]
        labels = full
    if labels.shape[0] != n:
        raise ValueError(f"PartSLIP++ label length {labels.shape[0]} does not match point count {n}.")
    unique = [item for item in np.unique(labels) if item != -1 and str(item) != "-1"]
    masks = np.stack([(labels == item).astype(np.uint8) for item in unique], axis=0) if unique else np.zeros((0, n), dtype=np.uint8)
    return masks, unique


def semantic_instance_to_masks(
    semantic: np.ndarray,
    instance: np.ndarray | None,
    n: int,
    point_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, list[tuple[Any, Any | None]]]:
    semantic = np.asarray(semantic).reshape(-1)
    instance_arr = np.asarray(instance).reshape(-1) if instance is not None else None
    if instance_arr is not None and instance_arr.shape[0] != semantic.shape[0]:
        raise ValueError("PartSLIP++ semantic and instance labels have different lengths.")
    if point_indices is not None:
        point_indices = np.asarray(point_indices, dtype=np.int64).reshape(-1)
        if semantic.shape[0] != point_indices.shape[0]:
            raise ValueError("PartSLIP++ semantic labels length does not match point_indices length.")
        full_sem = np.full((n,), -1, dtype=semantic.dtype)
        valid = (point_indices >= 0) & (point_indices < n)
        full_sem[point_indices[valid]] = semantic[valid]
        semantic = full_sem
        if instance_arr is not None:
            full_inst = np.full((n,), -1, dtype=instance_arr.dtype)
            full_inst[point_indices[valid]] = instance_arr[valid]
            instance_arr = full_inst
    if semantic.shape[0] != n:
        raise ValueError(f"PartSLIP++ semantic label length {semantic.shape[0]} does not match point count {n}.")
    if instance_arr is not None and instance_arr.shape[0] != n:
        raise ValueError(f"PartSLIP++ instance label length {instance_arr.shape[0]} does not match point count {n}.")

    keys: list[tuple[Any, Any | None]] = []
    for sem in np.unique(semantic):
        if sem == -1 or str(sem) == "-1":
            continue
        if instance_arr is None:
            keys.append((sem, None))
            continue
        for inst in np.unique(instance_arr[semantic == sem]):
            if inst == -1 or str(inst) == "-1":
                continue
            keys.append((sem, inst))
    masks = []
    for sem, inst in keys:
        mask = semantic == sem
        if inst is not None and instance_arr is not None:
            mask = mask & (instance_arr == inst)
        masks.append(mask.astype(np.uint8))
    if not masks:
        return np.zeros((0, n), dtype=np.uint8), []
    return np.stack(masks, axis=0), keys


def normalize_masks(raw_masks: Any, n: int, point_indices: np.ndarray | None = None) -> np.ndarray:
    masks = np.asarray(raw_masks)
    if masks.ndim == 1:
        masks, _ = labels_to_masks(masks, n, point_indices)
        return masks
    if masks.ndim != 2:
        raise ValueError(f"PartSLIP++ masks must be [K,N] or [N,K], got shape {masks.shape}.")
    if point_indices is not None:
        point_indices = np.asarray(point_indices, dtype=np.int64).reshape(-1)
        if masks.shape[1] == point_indices.shape[0]:
            full = np.zeros((masks.shape[0], n), dtype=np.uint8)
            valid = (point_indices >= 0) & (point_indices < n)
            full[:, point_indices[valid]] = masks[:, valid].astype(np.uint8)
            return full
        if masks.shape[0] == point_indices.shape[0]:
            full = np.zeros((masks.shape[1], n), dtype=np.uint8)
            valid = (point_indices >= 0) & (point_indices < n)
            full[:, point_indices[valid]] = masks[valid, :].T.astype(np.uint8)
            return full
    if masks.shape[1] == n:
        return masks.astype(np.uint8)
    if masks.shape[0] == n:
        return masks.T.astype(np.uint8)
    raise ValueError(f"PartSLIP++ mask shape {masks.shape} does not match point count {n}.")


def find_partslippp_path(root: Path, args: argparse.Namespace, row: dict[str, str]) -> Path:
    tokens = {key: str(value) for key, value in row.items()}
    for key in ("pilot_id", "sample_id", "task", "executor", "object_category"):
        tokens.setdefault(key, "")
    tokens.setdefault("partslippp_category", tokens.get("object_category", ""))
    tokens.setdefault("partslippp_checkpoint", tokens.get("partslippp_category", ""))
    tokens["object_category_lower"] = tokens.get("object_category", "").lower()
    tokens["partslippp_category_lower"] = tokens.get("partslippp_category", "").lower()
    if args.partslippp_path:
        formatted = args.partslippp_path.format(**tokens)
        path = resolve_portable_path(root, formatted)
        if path.exists():
            return path
        raise FileNotFoundError(f"PartSLIP++ prediction not found: {path}")

    base = resolve_path(root, args.partslippp_root)
    pilot_id = tokens["pilot_id"]
    sample_id = tokens["sample_id"]
    part_category = tokens.get("partslippp_category", "")
    candidates = [
        base / pilot_id / "partslippp_candidates.npz",
        base / pilot_id / "segments.npz",
        base / pilot_id / "prediction.npz",
        base / pilot_id / "result.npz",
        base / pilot_id / "partslippp_candidates.json",
        base / pilot_id / "segments.json",
        base / sample_id / f"{pilot_id}.npz",
        base / sample_id / "segments.npz",
        base / sample_id / "partslippp_candidates.npz",
        base / f"{pilot_id}.npz",
        base / f"{sample_id}_{tokens['task']}_{tokens['executor']}.npz",
        base / f"{sample_id}.npz",
        base / f"{pilot_id}.json",
        base / f"{sample_id}_{tokens['task']}_{tokens['executor']}.json",
        base / f"{sample_id}.json",
    ]
    if part_category:
        category_base = base / part_category
        candidates = [
            category_base / pilot_id / "partslippp_candidates.npz",
            category_base / pilot_id / "segments.npz",
            category_base / pilot_id / "prediction.npz",
            category_base / pilot_id / "result.npz",
            category_base / pilot_id / "partslippp_candidates.json",
            category_base / pilot_id / "segments.json",
            category_base / sample_id / "segments.npz",
            category_base / sample_id / "partslippp_candidates.npz",
            category_base / sample_id / f"{pilot_id}.npz",
            category_base / f"{sample_id}_{tokens['task']}_{tokens['executor']}.npz",
            category_base / f"{sample_id}.npz",
            category_base / f"{pilot_id}.npz",
            category_base / f"{sample_id}_{tokens['task']}_{tokens['executor']}.json",
            category_base / f"{sample_id}.json",
            category_base / f"{pilot_id}.json",
        ] + candidates
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No PartSLIP++ prediction found. Expected one of: "
        + ", ".join(str(path) for path in candidates[:6])
        + ", ..."
    )


def build_part_candidates_from_masks(
    masks: np.ndarray,
    names: list[str],
    scores: list[float | None],
    args: argparse.Namespace,
    n: int,
    source_path: Path,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    masks = (np.asarray(masks) > 0).astype(np.uint8)
    if masks.ndim != 2 or masks.shape[1] != n:
        raise ValueError(f"Expected PartSLIP++ masks shape [K,{n}], got {masks.shape}.")

    items: list[tuple[np.ndarray, dict[str, Any], float]] = []
    for idx, mask in enumerate(masks):
        point_count = int(mask.sum())
        if point_count < int(args.min_points):
            continue
        point_fraction = float(point_count / max(n, 1))
        if point_fraction > float(args.max_candidate_fraction):
            continue
        score = scores[idx] if idx < len(scores) else None
        name = names[idx] if idx < len(names) else f"part_{idx:02d}"
        meta = partslippp_meta(
            {
                "candidate_id": "",
                "candidate_name": name,
                "candidate_family": "partslippp_part",
                "point_count": point_count,
                "point_fraction": point_fraction,
                "score": score,
                "source_path": str(source_path),
            }
        )
        sort_score = float(score) if score is not None else 0.0
        items.append((mask, meta, sort_score))

    items.sort(key=lambda item: (item[2], item[1]["point_count"]), reverse=True)
    items = items[: int(args.max_candidates)]
    if not items:
        return np.zeros((0, n), dtype=np.uint8), []

    out_masks = []
    out_meta = []
    for idx, (mask, meta, _score) in enumerate(items):
        meta["candidate_id"] = candidate_id_for(idx)
        out_masks.append(mask)
        out_meta.append(meta)
    return np.stack(out_masks, axis=0).astype(np.uint8), out_meta


def load_partslippp_npz(path: Path, n: int, args: argparse.Namespace) -> tuple[np.ndarray, list[dict[str, Any]]]:
    data = np.load(path, allow_pickle=True)
    point_indices = data["point_indices"] if "point_indices" in data.files else None
    raw_masks = None
    for key in ("candidate_masks", "masks", "part_masks", "instance_masks"):
        if key in data.files:
            raw_masks = data[key]
            break
    names_payload = None
    for key in ("candidate_names", "part_names", "label_names", "labels"):
        if key in data.files:
            names_payload = data[key]
            break
    scores_payload = data["scores"] if "scores" in data.files else None

    if raw_masks is not None:
        masks = normalize_masks(raw_masks, n, point_indices)
        names = names_from_payload(names_payload, masks.shape[0], "partslippp_part")
        scores = scores_from_payload(scores_payload, masks.shape[0])
        return build_part_candidates_from_masks(masks, names, scores, args, n, path)

    semantic = None
    for key in ("semantic_seg", "semantic_labels", "part_labels", "labels"):
        if key in data.files:
            semantic = data[key]
            break
    if semantic is None:
        raise ValueError(f"PartSLIP++ npz has no masks or labels: {path}")
    instance = None
    for key in ("instance_seg", "instance_labels", "instances"):
        if key in data.files:
            instance = data[key]
            break
    masks, keys = semantic_instance_to_masks(semantic, instance, n, point_indices)
    label_lookup = maybe_object_payload(names_payload)
    if isinstance(label_lookup, dict):
        label_names = []
        for sem, inst in keys:
            base = label_lookup.get(str(sem), label_lookup.get(sem))
            base_name = decode_name(base) if base is not None else f"part_{sem}"
            label_names.append(f"{base_name}_inst_{inst}" if inst is not None else base_name)
    else:
        label_names = names_from_payload(names_payload, len(keys), "partslippp_part")
    scores = scores_from_payload(scores_payload, masks.shape[0])
    return build_part_candidates_from_masks(masks, label_names, scores, args, n, path)


def load_partslippp_json(path: Path, n: int, args: argparse.Namespace) -> tuple[np.ndarray, list[dict[str, Any]]]:
    payload = read_json(path)
    parts = payload.get("parts") or payload.get("candidates")
    if parts:
        masks = []
        names = []
        scores = []
        for idx, part in enumerate(parts):
            indices = part.get("indices") or part.get("point_indices") or []
            mask = np.zeros((n,), dtype=np.uint8)
            arr = np.asarray(indices, dtype=np.int64).reshape(-1)
            valid = (arr >= 0) & (arr < n)
            mask[arr[valid]] = 1
            masks.append(mask)
            names.append(str(part.get("name") or part.get("label") or f"partslippp_part_{idx:02d}"))
            scores.append(part.get("score"))
        stacked = np.stack(masks, axis=0) if masks else np.zeros((0, n), dtype=np.uint8)
        return build_part_candidates_from_masks(stacked, names, scores, args, n, path)

    point_indices = np.asarray(payload["point_indices"], dtype=np.int64) if "point_indices" in payload else None
    if "candidate_masks" in payload or "masks" in payload:
        masks = normalize_masks(payload.get("candidate_masks", payload.get("masks")), n, point_indices)
        names = names_from_payload(payload.get("candidate_names") or payload.get("part_names"), masks.shape[0], "partslippp_part")
        scores = scores_from_payload(payload.get("scores"), masks.shape[0])
        return build_part_candidates_from_masks(masks, names, scores, args, n, path)
    semantic = payload.get("semantic_seg") or payload.get("semantic_labels") or payload.get("part_labels") or payload.get("labels")
    if semantic is None:
        raise ValueError(f"PartSLIP++ json has no parts, masks, or labels: {path}")
    instance = payload.get("instance_seg") or payload.get("instance_labels") or payload.get("instances")
    masks, keys = semantic_instance_to_masks(np.asarray(semantic), np.asarray(instance) if instance is not None else None, n, point_indices)
    label_names = []
    label_lookup = payload.get("label_names") or payload.get("part_names") or {}
    for sem, inst in keys:
        base = label_lookup.get(str(sem), label_lookup.get(sem)) if isinstance(label_lookup, dict) else None
        base_name = decode_name(base) if base is not None else f"part_{sem}"
        label_names.append(f"{base_name}_inst_{inst}" if inst is not None else base_name)
    scores = scores_from_payload(payload.get("scores"), masks.shape[0])
    return build_part_candidates_from_masks(masks, label_names, scores, args, n, path)


def load_partslippp_candidates(path: Path, n: int, args: argparse.Namespace) -> tuple[np.ndarray, list[dict[str, Any]]]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return load_partslippp_npz(path, n, args)
    if suffix == ".json":
        return load_partslippp_json(path, n, args)
    raise ValueError(f"Unsupported PartSLIP++ prediction format: {path}")


def annotate_candidates(
    candidates: list[dict[str, Any]],
    *,
    candidate_source: str,
    hybrid_role: str,
    mapping: dict[str, Any] | None = None,
    prediction_path: Path | None = None,
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for item in candidates:
        meta = dict(item)
        meta["candidate_source"] = candidate_source
        meta["hybrid_role"] = hybrid_role
        meta.setdefault("need_review", True)
        meta.setdefault("allows_overlap", True)
        if mapping:
            meta["partslippp_category"] = mapping.get("partslippp_category", "")
            meta["partslippp_checkpoint"] = mapping.get("checkpoint", "")
        if prediction_path is not None:
            meta["partslippp_prediction_path"] = str(prediction_path)
        annotated.append(meta)
    return annotated


def merge_hybrid_candidates(
    primary_masks: np.ndarray,
    primary_candidates: list[dict[str, Any]],
    supplement_masks: np.ndarray,
    supplement_candidates: list[dict[str, Any]],
    args: argparse.Namespace,
    n: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    max_total = max(1, int(args.max_candidates))
    min_supplement = max(0, int(args.hybrid_min_high_recall_supplement))
    max_primary = max(0, int(args.hybrid_max_partslippp_primary))

    primary_limit = min(primary_masks.shape[0], max_total, max_primary or max_total)
    if supplement_masks.shape[0] > 0 and min_supplement > 0:
        reserve = min(min_supplement, supplement_masks.shape[0], max_total)
        primary_limit = min(primary_limit, max(0, max_total - reserve))

    selected: list[tuple[np.ndarray, dict[str, Any]]] = []

    def append_candidate(mask: np.ndarray, meta: dict[str, Any]) -> None:
        if len(selected) >= max_total:
            return
        for existing, _existing_meta in selected:
            if candidate_iou(mask, existing) >= float(args.dedupe_iou):
                return
        selected.append((mask.astype(np.uint8), dict(meta)))

    for idx in range(primary_limit):
        append_candidate(primary_masks[idx], primary_candidates[idx])
    for idx in range(min(supplement_masks.shape[0], len(supplement_candidates))):
        append_candidate(supplement_masks[idx], supplement_candidates[idx])
        if len(selected) >= max_total:
            break
    for idx in range(primary_limit, min(primary_masks.shape[0], len(primary_candidates))):
        append_candidate(primary_masks[idx], primary_candidates[idx])
        if len(selected) >= max_total:
            break

    if not selected:
        return np.zeros((0, n), dtype=np.uint8), []

    out_masks = np.stack([mask for mask, _meta in selected], axis=0).astype(np.uint8)
    out_candidates: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    for idx, (_mask, meta) in enumerate(selected):
        source = str(meta.get("candidate_source") or meta.get("source") or "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
        meta["candidate_id"] = candidate_id_for(idx)
        meta["rank_in_candidate_source"] = source_counts[source]
        meta["retention_policy"] = "hybrid_partslippp_primary_high_recall_supplement"
        out_candidates.append(meta)
    return out_masks, out_candidates


def candidate_iou(a: np.ndarray, b: np.ndarray) -> float:
    a_bool = a.astype(bool)
    b_bool = b.astype(bool)
    union = np.logical_or(a_bool, b_bool).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a_bool, b_bool).sum() / union)


def high_recall_record(
    name: str,
    mask: np.ndarray,
    family: str,
    part_group: str,
    description: str,
    recommended_executors: list[str],
    recommended_tasks: list[str],
    xyz: np.ndarray,
    priority: float,
    quality_hint: str,
    confidence_hint: str = "low",
    need_review: bool = True,
    partial: bool = False,
) -> dict[str, Any]:
    point_count = int(mask.sum())
    return {
        "candidate_id": "",
        "candidate_name": name,
        "candidate_family": family,
        "part_group": part_group,
        "description": description,
        "recommended_executors": recommended_executors,
        "recommended_tasks": recommended_tasks,
        "point_count": point_count,
        "point_fraction": float(point_count / max(1, mask.shape[0])),
        "bbox_extent_ratio": gen.bbox_extent_ratio(xyz, mask.astype(bool)),
        "quality_hint": quality_hint,
        "confidence_hint": confidence_hint,
        "need_review": bool(need_review),
        "is_partial_candidate": bool(partial),
        "allows_overlap": True,
        "priority": float(priority),
        "provenance": "v3_high_recall_3d_candidate_generator",
        "source": "self_high_recall_3d",
    }


def add_high_recall_candidate(
    items: list[tuple[np.ndarray, dict[str, Any]]],
    mask: np.ndarray,
    *,
    name: str,
    family: str,
    part_group: str,
    description: str,
    recommended_executors: list[str],
    recommended_tasks: list[str],
    xyz: np.ndarray,
    min_points: int,
    max_fraction: float,
    priority: float,
    quality_hint: str,
    confidence_hint: str = "low",
    need_review: bool = True,
    partial: bool = False,
    dedupe_iou: float = 0.985,
) -> None:
    mask = (np.asarray(mask).reshape(-1) > 0).astype(np.uint8)
    n = mask.shape[0]
    count = int(mask.sum())
    if count < int(min_points):
        return
    if count > int(n * max_fraction):
        return
    # Keep overlap by design. Only collapse almost identical masks within the
    # same semantic group to keep the review UI readable.
    for old_mask, old_meta in items:
        if old_meta.get("part_group") != part_group:
            continue
        if candidate_iou(mask, old_mask) >= float(dedupe_iou):
            return
    items.append(
        (
            mask,
            high_recall_record(
                name=name,
                mask=mask,
                family=family,
                part_group=part_group,
                description=description,
                recommended_executors=list(dict.fromkeys(recommended_executors)),
                recommended_tasks=list(dict.fromkeys(recommended_tasks)),
                xyz=xyz,
                priority=priority,
                quality_hint=quality_hint,
                confidence_hint=confidence_hint,
                need_review=need_review,
                partial=partial,
            ),
        )
    )


def expanded(mask: np.ndarray, knn: np.ndarray, hops: int) -> np.ndarray:
    return gen.knn_expand_mask(mask.astype(bool), knn, hops).astype(np.uint8)


def component_masks(
    seed: np.ndarray,
    knn: np.ndarray,
    n: int,
    min_points: int,
    max_fraction: float,
    limit: int,
) -> list[np.ndarray]:
    comps = gen.graph_components_from_seed(
        seed.astype(bool),
        knn,
        min_points=max(1, int(min_points)),
        max_points=max(int(min_points), int(n * max_fraction)),
    )
    return [gen.mask_from_ids(n, ids) for ids in comps[: max(0, int(limit))]]


def add_component_family(
    items: list[tuple[np.ndarray, dict[str, Any]]],
    *,
    seed: np.ndarray,
    knn: np.ndarray,
    xyz: np.ndarray,
    args: argparse.Namespace,
    name_prefix: str,
    family: str,
    part_group: str,
    description: str,
    recommended_executors: list[str],
    recommended_tasks: list[str],
    max_fraction: float,
    priority: float,
    quality_hint: str,
    confidence_hint: str,
    add_seed: bool = True,
    partial: bool = False,
) -> None:
    n = xyz.shape[0]
    comps = component_masks(seed, knn, n, args.min_points, max_fraction, args.part_top_k)
    for idx, comp in enumerate(comps, start=1):
        comp_expanded = expanded(comp, knn, args.seed_expand_hops)
        add_high_recall_candidate(
            items,
            comp_expanded,
            name=f"{name_prefix}_{idx:02d}_expanded",
            family=f"expanded_{family}",
            part_group=part_group,
            description=f"kNN-expanded {description}",
            recommended_executors=recommended_executors,
            recommended_tasks=recommended_tasks,
            xyz=xyz,
            min_points=args.min_points,
            max_fraction=max_fraction,
            priority=priority + idx * 0.01,
            quality_hint=f"expanded_{quality_hint}",
            confidence_hint=confidence_hint,
            need_review=True,
            partial=partial,
            dedupe_iou=args.dedupe_iou,
        )
        if add_seed:
            add_high_recall_candidate(
                items,
                comp,
                name=f"{name_prefix}_{idx:02d}_seed",
                family=family,
                part_group=f"{part_group}_seed",
                description=description,
                recommended_executors=recommended_executors,
                recommended_tasks=recommended_tasks,
                xyz=xyz,
                min_points=args.min_points,
                max_fraction=max_fraction,
                priority=priority + 12.0 + idx * 0.01,
                quality_hint=quality_hint,
                confidence_hint="low",
                need_review=True,
                partial=True,
                dedupe_iou=args.dedupe_iou,
            )


def finalize_high_recall(
    items: list[tuple[np.ndarray, dict[str, Any]]],
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if not items:
        return np.zeros((0, 0), dtype=np.uint8), []
    grouped: dict[str, list[tuple[np.ndarray, dict[str, Any]]]] = {}
    for mask, meta in items:
        grouped.setdefault(str(meta.get("part_group", "misc")), []).append((mask, meta))
    for values in grouped.values():
        values.sort(key=lambda item: (float(item[1].get("priority", 50.0)), -int(item[1].get("point_count", 0))))

    selected: list[tuple[np.ndarray, dict[str, Any]]] = []
    selected_keys: set[int] = set()
    max_per_group = max(1, int(args.part_top_k))
    max_candidates = max(1, int(args.max_candidates))

    group_names = sorted(
        grouped,
        key=lambda group: min(float(meta.get("priority", 50.0)) for _, meta in grouped[group]),
    )
    for rank in range(max_per_group):
        for group in group_names:
            values = grouped[group]
            if rank >= len(values):
                continue
            item = values[rank]
            item_key = id(item[1])
            if item_key in selected_keys:
                continue
            selected.append(item)
            selected_keys.add(item_key)
            if len(selected) >= max_candidates:
                break
        if len(selected) >= max_candidates:
            break

    if len(selected) < max_candidates:
        remaining = [
            item
            for values in grouped.values()
            for item in values
            if id(item[1]) not in selected_keys
        ]
        remaining.sort(key=lambda item: (float(item[1].get("priority", 50.0)), -int(item[1].get("point_count", 0))))
        for item in remaining:
            selected.append(item)
            if len(selected) >= max_candidates:
                break

    masks = np.stack([mask for mask, _ in selected], axis=0).astype(np.uint8)
    metas = [dict(meta) for _, meta in selected]
    per_group_seen: dict[str, int] = {}
    for idx, meta in enumerate(metas):
        group = str(meta.get("part_group", "misc"))
        per_group_seen[group] = per_group_seen.get(group, 0) + 1
        meta["candidate_id"] = candidate_id_for(idx)
        meta["rank_in_group"] = per_group_seen[group]
        meta["retention_policy"] = "high_recall_top_k_per_group"
    return masks, metas


def generate_with_high_recall(
    root: Path,
    args: argparse.Namespace,
    row: dict[str, str],
    sample_by_id: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]], str, Path, Path | None]:
    enriched = gen.enrich_row(row, sample_by_id)
    point_path = resolve_portable_path(root, enriched.get("point_cloud_path", ""))
    mask_value = enriched.get("checked_mask_path") or enriched.get("multi_channel_mask_path") or enriched.get("source_mask_path")
    mask_path = resolve_portable_path(root, mask_value) if mask_value else None
    xyz, normals = gen.load_points(point_path)
    weak_mask = gen.load_mask(mask_path, xyz.shape[0])
    n = xyz.shape[0]
    task = str(enriched.get("task", ""))
    executor = str(enriched.get("executor", ""))
    all_tasks = ["pick_up", "lift_carry", "open_pull", "press_push"]
    dexterous_like = ["gripper", "hook", "dexterous_hand"]
    items: list[tuple[np.ndarray, dict[str, Any]]] = []

    # Keep the previous geometry generator as a baseline proposal source, but
    # re-label its outputs as review candidates rather than final selections.
    try:
        base_masks, base_candidates = gen.generate_candidates(xyz, normals, weak_mask, enriched, args)
        for mask, meta in zip(base_masks, base_candidates):
            add_high_recall_candidate(
                items,
                mask,
                name=f"baseline_{meta.get('candidate_name', 'candidate')}",
                family=str(meta.get("candidate_family", "baseline_geometry")),
                part_group=f"baseline_{meta.get('candidate_family', 'geometry')}",
                description=str(meta.get("description", "Baseline geometry proposal from the previous generator.")),
                recommended_executors=list(meta.get("recommended_executors", dexterous_like)),
                recommended_tasks=list(meta.get("recommended_tasks", all_tasks)),
                xyz=xyz,
                min_points=args.min_points,
                max_fraction=max(float(args.max_candidate_fraction), 0.82),
                priority=float(meta.get("priority", 60.0)),
                quality_hint=str(meta.get("quality_hint", "baseline_geometry")),
                confidence_hint="medium" if meta.get("quality_hint") in {"source_prior", "targeted_visual_proposal"} else "low",
                need_review=True,
                partial=str(meta.get("candidate_family", "")).endswith("boundary"),
                dedupe_iou=args.dedupe_iou,
            )
    except Exception:
        # The high-recall path should still produce candidates when the older
        # generator fails on missing renders or sparse geometry.
        pass

    knn = gen.pairwise_knn(xyz, args.k_neighbors)
    features = gen.local_pca_features(xyz, knn)
    curvature = features["curvature"]
    linearity = features["linearity"]
    planarity = features["planarity"]

    high_curv_loose = curvature >= np.quantile(curvature, 0.70)
    high_curv_strict = curvature >= np.quantile(curvature, 0.88)
    linear_loose = linearity >= np.quantile(linearity, 0.70)
    linear_strict = linearity >= np.quantile(linearity, 0.88)
    smooth_loose = curvature <= np.quantile(curvature, 0.45)
    planar = planarity >= np.quantile(planarity, 0.55)
    edge_linear = high_curv_loose | linear_loose

    # 1. Coarse partitions. These improve recall for ordinary bodies, panels,
    # broad handles, and partial visible parts. They deliberately cover the
    # object without requiring high confidence.
    for axis in range(3):
        values = xyz[:, axis]
        qs = np.quantile(values, [0.0, 0.34, 0.67, 1.0])
        for band_idx, (lo, hi) in enumerate(zip(qs[:-1], qs[1:]), start=1):
            band = (values >= lo) & (values <= hi)
            add_high_recall_candidate(
                items,
                band,
                name=f"coarse_axis_{gen.axis_name(axis)}_band_{band_idx}",
                family="coarse_axis_partition",
                part_group=f"coarse_partition_{gen.axis_name(axis)}",
                description="Coarse axis partition retained for high recall and complete object decomposition.",
                recommended_executors=["gripper", "suction", "hook", "dexterous_hand"],
                recommended_tasks=all_tasks,
                xyz=xyz,
                min_points=args.min_points,
                max_fraction=0.70,
                priority=70.0 + axis + band_idx * 0.01,
                quality_hint="coarse_review_partition",
                confidence_hint="low",
                need_review=True,
                partial=True,
                dedupe_iou=args.dedupe_iou,
            )

    # 2. View-derived partial/upper/small components. These catch handles,
    # knobs, buttons, and partially visible structures that pure 3D thresholds
    # often miss.
    view_groups = gen.render_component_candidate_ids(root, args, str(enriched.get("sample_id", "")), n)
    for group_name, ids in view_groups.items():
        if ids.size == 0:
            continue
        seed = gen.mask_from_ids(n, ids)
        add_high_recall_candidate(
            items,
            expanded(seed, knn, args.seed_expand_hops),
            name=f"{group_name}_expanded",
            family="visual_component_expanded",
            part_group="visual_partial_component",
            description="Expanded multi-view foreground component; retained even when partially visible.",
            recommended_executors=dexterous_like,
            recommended_tasks=all_tasks,
            xyz=xyz,
            min_points=args.min_points,
            max_fraction=0.55,
            priority=12.0,
            quality_hint="visual_high_recall",
            confidence_hint="medium",
            need_review=True,
            partial=True,
            dedupe_iou=args.dedupe_iou,
        )
        add_high_recall_candidate(
            items,
            seed,
            name=group_name,
            family="visual_component_seed",
            part_group="visual_partial_component_seed",
            description="Raw multi-view foreground component seed for precise review.",
            recommended_executors=dexterous_like,
            recommended_tasks=all_tasks,
            xyz=xyz,
            min_points=args.min_points,
            max_fraction=0.45,
            priority=28.0,
            quality_hint="visual_seed",
            confidence_hint="low",
            need_review=True,
            partial=True,
            dedupe_iou=args.dedupe_iou,
        )

    # 3. Weak mask priors and expanded variants. A point can belong to both
    # weak-prior and geometry candidates.
    if weak_mask is not None:
        for ch, source_executor in enumerate(gen.EXECUTOR_ORDER):
            source = weak_mask[:, ch] > 0
            if not np.any(source):
                continue
            recommended = [source_executor]
            if source_executor in {"gripper", "dexterous_hand"}:
                recommended.extend(["hook", "dexterous_hand", "gripper"])
            add_high_recall_candidate(
                items,
                expanded(source, knn, max(1, args.seed_expand_hops)),
                name=f"weak_{source_executor}_prior_expanded",
                family="expanded_existing_weak_mask",
                part_group="weak_mask_prior",
                description="Expanded existing weak mask prior; useful but not trusted as final GT.",
                recommended_executors=recommended,
                recommended_tasks=[task] if task else all_tasks,
                xyz=xyz,
                min_points=args.min_points,
                max_fraction=max(float(args.max_candidate_fraction), 0.82),
                priority=18.0,
                quality_hint="expanded_source_prior",
                confidence_hint="medium",
                need_review=True,
                partial=False,
                dedupe_iou=args.dedupe_iou,
            )
            add_high_recall_candidate(
                items,
                source,
                name=f"weak_{source_executor}_prior_seed",
                family="existing_weak_mask",
                part_group="weak_mask_prior_seed",
                description="Raw existing weak mask prior retained for traceability.",
                recommended_executors=recommended,
                recommended_tasks=[task] if task else all_tasks,
                xyz=xyz,
                min_points=args.min_points,
                max_fraction=max(float(args.max_candidate_fraction), 0.82),
                priority=34.0,
                quality_hint="source_prior_seed",
                confidence_hint="low",
                need_review=True,
                partial=True,
                dedupe_iou=args.dedupe_iou,
            )

    # 4. Top-k per geometric part family. Loose thresholds favor recall; strict
    # seeds keep more precise candidates for review.
    add_component_family(
        items,
        seed=edge_linear,
        knn=knn,
        xyz=xyz,
        args=args,
        name_prefix="loop_handle_or_lip_loose",
        family="loop_handle_lip_seed",
        part_group="loop_handle_lip",
        description="connected high-curvature or linear component for handles, rings, holes, lips, and thin parts.",
        recommended_executors=dexterous_like,
        recommended_tasks=all_tasks,
        max_fraction=0.50,
        priority=20.0,
        quality_hint="loose_functional_component",
        confidence_hint="medium",
        add_seed=True,
        partial=False,
    )
    add_component_family(
        items,
        seed=high_curv_strict | linear_strict,
        knn=knn,
        xyz=xyz,
        args=args,
        name_prefix="loop_handle_or_lip_strict",
        family="strict_loop_handle_lip_seed",
        part_group="loop_handle_lip_strict",
        description="strict high-curvature/linear component for more precise functional boundaries.",
        recommended_executors=dexterous_like,
        recommended_tasks=all_tasks,
        max_fraction=0.38,
        priority=24.0,
        quality_hint="strict_functional_component",
        confidence_hint="medium",
        add_seed=True,
        partial=True,
    )
    add_component_family(
        items,
        seed=smooth_loose & planar,
        knn=knn,
        xyz=xyz,
        args=args,
        name_prefix="smooth_panel_or_surface",
        family="smooth_surface_component",
        part_group="smooth_panel_surface",
        description="smooth/planar component for suction, press, panel, door, or broad contact regions.",
        recommended_executors=["suction", "dexterous_hand"],
        recommended_tasks=["pick_up", "open_pull", "press_push"],
        max_fraction=0.65,
        priority=38.0,
        quality_hint="surface_component",
        confidence_hint="medium",
        add_seed=False,
        partial=False,
    )

    # 5. Small and extreme parts. These are intentionally kept as need_review
    # candidates instead of being discarded as low confidence.
    small_max = max(float(args.small_part_max_fraction), 0.03)
    compact_extreme = np.zeros((n,), dtype=bool)
    for axis in range(3):
        values = xyz[:, axis]
        compact_extreme |= (values <= np.quantile(values, 0.10)) & high_curv_loose
        compact_extreme |= (values >= np.quantile(values, 0.90)) & high_curv_loose
    add_component_family(
        items,
        seed=compact_extreme | high_curv_strict,
        knn=knn,
        xyz=xyz,
        args=args,
        name_prefix="small_button_knob_joint",
        family="small_part_seed",
        part_group="small_part",
        description="small compact or high-curvature component for buttons, knobs, switches, handle joints, and thin legs.",
        recommended_executors=["gripper", "dexterous_hand", "suction", "hook"],
        recommended_tasks=all_tasks,
        max_fraction=small_max,
        priority=16.0,
        quality_hint="small_part_high_recall",
        confidence_hint="low",
        add_seed=True,
        partial=True,
    )

    # 6. Symmetric/paired handles or holes. Keep unions of top components so
    # scissors/faucet/bag-like paired structures are not split away too early.
    top_loop_components = component_masks(edge_linear, knn, n, args.min_points, 0.40, max(args.part_top_k, 4))
    if len(top_loop_components) >= 2:
        pair = np.zeros((n,), dtype=np.uint8)
        for comp in top_loop_components[:2]:
            pair |= expanded(comp, knn, args.seed_expand_hops)
        add_high_recall_candidate(
            items,
            pair,
            name="paired_loop_handle_or_ring_components",
            family="paired_loop_handle_candidate",
            part_group="paired_parts",
            description="Union of two strong functional components for paired handles, rings, scissor holes, or symmetric structures.",
            recommended_executors=dexterous_like,
            recommended_tasks=["pick_up", "lift_carry", "open_pull"],
            xyz=xyz,
            min_points=args.min_points,
            max_fraction=0.60,
            priority=14.0,
            quality_hint="paired_high_recall",
            confidence_hint="medium",
            need_review=True,
            partial=False,
            dedupe_iou=args.dedupe_iou,
        )

    candidate_masks, candidates = finalize_high_recall(items, args)
    if candidate_masks.size == 0 or not candidates:
        raise ValueError("No high-recall candidates generated; inspect point cloud or relax min-points.")
    warning = ""
    if len(candidates) >= int(args.max_candidates):
        warning = "high_recall_candidate_cap_reached_review_top_k"
    return candidate_masks, candidates, warning, point_path, mask_path


def generate_with_geometry(
    root: Path,
    args: argparse.Namespace,
    row: dict[str, str],
    sample_by_id: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]], str, Path, Path | None]:
    enriched = gen.enrich_row(row, sample_by_id)
    point_path = resolve_portable_path(root, enriched.get("point_cloud_path", ""))
    mask_value = (
        enriched.get("checked_mask_path")
        or enriched.get("multi_channel_mask_path")
        or enriched.get("source_mask_path")
    )
    mask_path = resolve_portable_path(root, mask_value) if mask_value else None
    xyz, normals = gen.load_points(point_path)
    weak_mask = gen.load_mask(mask_path, xyz.shape[0])
    candidate_masks, candidates = gen.generate_candidates(xyz, normals, weak_mask, enriched, args)
    return candidate_masks, [normalize_meta(item) for item in candidates], "", point_path, mask_path


def generate_with_hybrid_partslippp_high_recall(
    root: Path,
    args: argparse.Namespace,
    row: dict[str, str],
    sample_by_id: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]], str, Path, Path | None, dict[str, Any]]:
    enriched = gen.enrich_row(row, sample_by_id)
    point_path = resolve_portable_path(root, enriched.get("point_cloud_path", ""))
    mask_value = enriched.get("checked_mask_path") or enriched.get("multi_channel_mask_path") or enriched.get("source_mask_path")
    mask_path = resolve_portable_path(root, mask_value) if mask_value else None
    xyz, _normals = gen.load_points(point_path)
    n = xyz.shape[0]

    route_info: dict[str, Any] = {
        "backend_route": "hybrid_partslippp_high_recall",
        "partslippp_status": "not_attempted",
        "fallback_backend": "high_recall",
        "fallback_reason": "",
    }
    warnings: list[str] = []
    mapping = partslippp_mapping_for(root, args, enriched)

    part_masks = np.zeros((0, n), dtype=np.uint8)
    part_candidates: list[dict[str, Any]] = []
    if mapping is None:
        route_info["partslippp_status"] = "unmapped_category"
        route_info["fallback_reason"] = f"category_not_mapped:{enriched.get('object_category', '')}"
        warnings.append(route_info["fallback_reason"])
    else:
        route_row = dict(row)
        for key in ("sample_id", "task", "executor", "object_category"):
            if enriched.get(key):
                route_row[key] = str(enriched.get(key))
        route_row["partslippp_category"] = str(mapping.get("partslippp_category", ""))
        route_row["partslippp_checkpoint"] = str(mapping.get("checkpoint", ""))
        route_info["partslippp_category"] = route_row["partslippp_category"]
        route_info["partslippp_checkpoint"] = route_row["partslippp_checkpoint"]
        try:
            prediction_path = find_partslippp_path(root, args, route_row)
            part_masks, loaded_part_candidates = load_partslippp_candidates(prediction_path, n, args)
            part_candidates = annotate_candidates(
                loaded_part_candidates,
                candidate_source="partslippp",
                hybrid_role="primary",
                mapping=mapping,
                prediction_path=prediction_path,
            )
            route_info["partslippp_status"] = "loaded"
            route_info["partslippp_prediction_path"] = relative_to_dataset(root, prediction_path)
        except Exception as exc:
            route_info["partslippp_status"] = "failed"
            route_info["fallback_reason"] = f"partslippp_failed:{type(exc).__name__}:{exc}"
            warnings.append(route_info["fallback_reason"])

    try:
        high_masks, high_candidates, high_warning, high_point_path, high_mask_path = generate_with_high_recall(
            root, args, row, sample_by_id
        )
        point_path = high_point_path
        mask_path = high_mask_path
        high_candidates = annotate_candidates(
            high_candidates,
            candidate_source="self_high_recall_3d",
            hybrid_role="supplement" if part_candidates else "fallback",
        )
        if high_warning:
            warnings.append(high_warning)
    except Exception as exc:
        if part_candidates:
            high_masks = np.zeros((0, n), dtype=np.uint8)
            high_candidates = []
            warnings.append(f"high_recall_supplement_failed:{type(exc).__name__}:{exc}")
        elif args.allow_empty_candidates:
            candidate_masks, candidates, status = empty_candidates(n)
            route_info["fallback_reason"] = f"{route_info.get('fallback_reason', '')}; high_recall_failed:{type(exc).__name__}:{exc}"
            warnings.append(status)
            return candidate_masks, candidates, "; ".join(item for item in warnings if item), point_path, mask_path, route_info
        else:
            raise

    if part_candidates:
        candidate_masks, candidates = merge_hybrid_candidates(part_masks, part_candidates, high_masks, high_candidates, args, n)
    else:
        candidate_masks, candidates = high_masks, high_candidates

    route_info["candidate_count_by_source"] = {}
    for item in candidates:
        source = str(item.get("candidate_source") or item.get("source") or "unknown")
        route_info["candidate_count_by_source"][source] = route_info["candidate_count_by_source"].get(source, 0) + 1

    if candidate_masks.size == 0 or not candidates:
        if not args.allow_empty_candidates:
            raise ValueError("Hybrid backend produced no candidates.")
        candidate_masks, candidates, status = empty_candidates(n)
        warnings.append(status)
    return candidate_masks, candidates, "; ".join(item for item in warnings if item), point_path, mask_path, route_info


def generate_with_backend(
    root: Path,
    args: argparse.Namespace,
    row: dict[str, str],
    sample_by_id: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]], str, Path, Path | None, dict[str, Any]]:
    if args.backend == "high_recall":
        try:
            candidate_masks, candidates, warning, point_path, mask_path = generate_with_high_recall(root, args, row, sample_by_id)
            return candidate_masks, candidates, warning, point_path, mask_path, {
                "backend_route": "high_recall",
                "partslippp_status": "not_used",
            }
        except Exception as exc:
            if not args.allow_empty_candidates:
                raise
            enriched = gen.enrich_row(row, sample_by_id)
            point_path = resolve_portable_path(root, enriched.get("point_cloud_path", ""))
            mask_value = enriched.get("checked_mask_path") or enriched.get("multi_channel_mask_path") or enriched.get("source_mask_path")
            mask_path = resolve_portable_path(root, mask_value) if mask_value else None
            xyz, _normals = gen.load_points(point_path)
            candidate_masks, candidates, status = empty_candidates(xyz.shape[0])
            return candidate_masks, candidates, f"{status}: high_recall_failed: {type(exc).__name__}: {exc}", point_path, mask_path, {
                "backend_route": "high_recall",
                "partslippp_status": "not_used",
                "fallback_reason": f"high_recall_failed:{type(exc).__name__}:{exc}",
            }

    if args.backend == "partslippp":
        enriched = gen.enrich_row(row, sample_by_id)
        point_path = resolve_portable_path(root, enriched.get("point_cloud_path", ""))
        xyz, _normals = gen.load_points(point_path)
        mask_value = enriched.get("checked_mask_path") or enriched.get("multi_channel_mask_path") or enriched.get("source_mask_path")
        mask_path = resolve_portable_path(root, mask_value) if mask_value else None
        try:
            prediction_path = find_partslippp_path(root, args, row)
            candidate_masks, candidates = load_partslippp_candidates(prediction_path, xyz.shape[0], args)
            candidates = annotate_candidates(candidates, candidate_source="partslippp", hybrid_role="primary", prediction_path=prediction_path)
            return candidate_masks, candidates, "", point_path, mask_path, {
                "backend_route": "partslippp",
                "partslippp_status": "loaded",
                "partslippp_prediction_path": relative_to_dataset(root, prediction_path),
            }
        except Exception as exc:
            if args.partslippp_fallback == "geometry":
                candidate_masks, candidates, warning, point_path, mask_path = generate_with_geometry(root, args, row, sample_by_id)
                fallback_warning = f"partslippp_failed_using_geometry: {type(exc).__name__}: {exc}"
                return candidate_masks, candidates, fallback_warning if not warning else f"{fallback_warning}; {warning}", point_path, mask_path, {
                    "backend_route": "partslippp_geometry_fallback",
                    "partslippp_status": "failed",
                    "fallback_backend": "geometry",
                    "fallback_reason": f"{type(exc).__name__}: {exc}",
                }
            raise

    if args.backend == "hybrid_partslippp_high_recall":
        return generate_with_hybrid_partslippp_high_recall(root, args, row, sample_by_id)

    if args.backend != "geometry":
        raise ValueError(f"Unsupported backend: {args.backend}")

    try:
        candidate_masks, candidates, warning, point_path, mask_path = generate_with_geometry(root, args, row, sample_by_id)
        return candidate_masks, candidates, warning, point_path, mask_path, {
            "backend_route": "geometry",
            "partslippp_status": "not_used",
        }
    except Exception as exc:
        if not args.allow_empty_candidates:
            raise
        enriched = gen.enrich_row(row, sample_by_id)
        point_path = resolve_portable_path(root, enriched.get("point_cloud_path", ""))
        mask_value = enriched.get("checked_mask_path") or enriched.get("multi_channel_mask_path") or enriched.get("source_mask_path")
        mask_path = resolve_portable_path(root, mask_value) if mask_value else None
        xyz, _normals = gen.load_points(point_path)
        candidate_masks, candidates, status = empty_candidates(xyz.shape[0])
        return candidate_masks, candidates, f"{status}: {type(exc).__name__}: {exc}", point_path, mask_path, {
            "backend_route": "geometry",
            "partslippp_status": "not_used",
            "fallback_reason": f"geometry_failed:{type(exc).__name__}:{exc}",
        }


def write_candidate_outputs(
    root: Path,
    args: argparse.Namespace,
    row: dict[str, str],
    candidate_masks: np.ndarray,
    candidates: list[dict[str, Any]],
    generation_warning: str,
    point_path: Path,
    mask_path: Path | None,
    route_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pilot_id = row["pilot_id"]
    sample_id = row["sample_id"]
    output_dir = resolve_path(root, args.output_root) / pilot_id
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "candidates.npz"
    if npz_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {npz_path}")

    np.savez_compressed(
        npz_path,
        pilot_id=pilot_id,
        sample_id=sample_id,
        executor=row.get("executor", ""),
        candidate_ids=np.asarray([item["candidate_id"] for item in candidates], dtype=object),
        candidate_names=np.asarray([item["candidate_name"] for item in candidates], dtype=object),
        candidate_families=np.asarray([item["candidate_family"] for item in candidates], dtype=object),
        candidate_masks=candidate_masks.astype(np.uint8),
    )

    plan_path = semantic_plan_path(root, args, pilot_id)
    manifest = {
        "version": "v3",
        "pipeline": "part_segmentation_candidate_proposal",
        "candidate_source": "partseg",
        "backend": args.backend,
        "backend_route": (route_info or {}).get("backend_route", args.backend),
        "partslippp_status": (route_info or {}).get("partslippp_status", "not_used"),
        "partslippp_category": (route_info or {}).get("partslippp_category", ""),
        "partslippp_checkpoint": (route_info or {}).get("partslippp_checkpoint", ""),
        "partslippp_prediction_path": (route_info or {}).get("partslippp_prediction_path", ""),
        "fallback_backend": (route_info or {}).get("fallback_backend", ""),
        "fallback_reason": (route_info or {}).get("fallback_reason", ""),
        "candidate_count_by_source": (route_info or {}).get("candidate_count_by_source", {}),
        "pilot_id": pilot_id,
        "sample_id": sample_id,
        "object_category": row.get("object_category", ""),
        "task": row.get("task", ""),
        "executor": row.get("executor", ""),
        "point_cloud_path": relative_to_dataset(root, point_path),
        "weak_mask_path": relative_to_dataset(root, mask_path) if mask_path and mask_path.exists() else None,
        "semantic_plan": relative_to_dataset(root, plan_path) if plan_path.exists() else "",
        "renders_root": args.renders_root,
        "candidate_npz": relative_to_dataset(root, npz_path),
        "candidate_count": int(candidate_masks.shape[0]),
        "default_selected_candidates": [],
        "candidates": candidates,
        "generation_warning": generation_warning,
        "parameters": {
            "partslippp_root": args.partslippp_root,
            "partslippp_path": args.partslippp_path,
            "partslippp_category_map": args.partslippp_category_map,
            "partslippp_fallback": args.partslippp_fallback,
            "hybrid_min_high_recall_supplement": int(args.hybrid_min_high_recall_supplement),
            "hybrid_max_partslippp_primary": int(args.hybrid_max_partslippp_primary),
            "k_neighbors": int(args.k_neighbors),
            "min_points": int(args.min_points),
            "max_candidate_fraction": float(args.max_candidate_fraction),
            "max_candidates": int(args.max_candidates),
            "part_top_k": int(args.part_top_k),
            "dedupe_iou": float(args.dedupe_iou),
            "small_part_max_fraction": float(args.small_part_max_fraction),
            "seed_expand_hops": int(args.seed_expand_hops),
            "component_max_candidates": int(args.component_max_candidates),
            "view_dilation_radius": int(args.view_dilation_radius),
            "body_row_threshold_ratio": float(args.body_row_threshold_ratio),
            "body_top_margin": int(args.body_top_margin),
        },
        "notes": (
            "Part candidates are high-recall proposals over original point indices only. "
            "The default high_recall backend favors recall over precision, permits overlapping candidates, "
            "keeps top-k candidates per part group, and marks uncertain regions for human review."
        ),
    }
    manifest_path = output_dir / "candidate_manifest.json"
    write_json(manifest_path, manifest, args.overwrite)
    return {
        "pilot_id": pilot_id,
        "sample_id": sample_id,
        "executor": row.get("executor", ""),
        "candidate_count": int(candidate_masks.shape[0]),
        "candidate_point_counts": {item["candidate_id"]: int(item["point_count"]) for item in candidates},
        "candidate_manifest": relative_to_dataset(root, manifest_path),
        "generation_warning": generation_warning,
    }


def generate_for_row(
    root: Path,
    args: argparse.Namespace,
    row: dict[str, str],
    sample_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    candidate_masks, candidates, warning, point_path, mask_path, route_info = generate_with_backend(root, args, row, sample_by_id)
    return write_candidate_outputs(root, args, row, candidate_masks, candidates, warning, point_path, mask_path, route_info)


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    rows = selected_rows(root, args)
    checked_samples = gen.read_jsonl(resolve_path(root, args.samples))
    sample_by_id = {str(row.get("sample_id")): row for row in checked_samples}
    outputs = [generate_for_row(root, args, row, sample_by_id) for row in progress_rows(rows, "part propose")]
    print(json.dumps({"version": "v3", "generated_rows": len(outputs), "rows": outputs}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
