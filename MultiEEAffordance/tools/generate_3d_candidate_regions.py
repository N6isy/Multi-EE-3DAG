#!/usr/bin/env python3
"""Generate general 3D candidate regions for VLM-guided affordance labeling.

This is the first stage of the v2 pipeline:

  point cloud + optional weak masks
      -> geometry-first candidate masks
      -> VLM candidate selection
      -> executor-rule filtering
      -> human review

The generator is intentionally high-recall. It proposes interpretable regions
such as smooth patches, extreme bands, thin structures, high-curvature points,
small protrusions, and existing weak-label channels. These candidates are not
ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import string
import sys
from pathlib import Path
from typing import Any

import numpy as np

from path_utils import relative_to_dataset, resolve_portable_path


EXECUTOR_ORDER = ["gripper", "suction", "hook", "dexterous_hand"]
LETTERS = list(string.ascii_uppercase)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate v2 general 3D candidate regions.")
    parser.add_argument("--dataset-root", default=".", help="Dataset root directory.")
    parser.add_argument(
        "--pilot-csv",
        default="processed/metadata/vlm_pilot_samples_v0_1.csv",
        help="Channel-level pilot CSV relative to dataset root.",
    )
    parser.add_argument(
        "--samples",
        default="processed/metadata/samples_checked_v0_1.jsonl",
        help="Checked sample JSONL used as metadata fallback.",
    )
    parser.add_argument(
        "--output-root",
        default="processed/vlm_candidate_v2/3d_candidates",
        help="Output root relative to dataset root.",
    )
    parser.add_argument("--pilot-id", default=None, help="Generate only one pilot row.")
    parser.add_argument("--limit", type=int, default=None, help="Limit selected pilot rows.")
    parser.add_argument("--k-neighbors", type=int, default=24, help="kNN size for local geometric features.")
    parser.add_argument("--min-points", type=int, default=4, help="Minimum points for a candidate.")
    parser.add_argument(
        "--max-candidate-fraction",
        type=float,
        default=0.70,
        help="Drop candidates larger than this fraction of the object.",
    )
    parser.add_argument("--max-candidates", type=int, default=18, help="Maximum candidates kept per pilot row.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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


def write_json(path: Path, data: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_points(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    if not path.exists():
        raise FileNotFoundError(f"Point cloud not found: {path}")
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] not in (3, 6):
        raise ValueError(f"Expected point cloud shape [N,3] or [N,6], got {arr.shape}: {path}")
    xyz = arr[:, :3].astype(np.float32)
    normals = arr[:, 3:6].astype(np.float32) if arr.shape[1] >= 6 else None
    return xyz, normals


def load_mask(path: Path | None, n: int) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    mask = np.load(path)
    if mask.ndim != 2 or mask.shape[0] != n or mask.shape[1] != len(EXECUTOR_ORDER):
        raise ValueError(f"Expected mask shape [N,4] matching N={n}, got {mask.shape}: {path}")
    return mask.astype(np.uint8)


def pairwise_knn(xyz: np.ndarray, k: int) -> np.ndarray:
    n = xyz.shape[0]
    if n <= 1:
        return np.zeros((n, 0), dtype=np.int64)
    k = max(1, min(int(k), n - 1))
    diff = xyz[:, None, :] - xyz[None, :, :]
    dist2 = np.einsum("ijk,ijk->ij", diff, diff)
    np.fill_diagonal(dist2, np.inf)
    return np.argpartition(dist2, kth=k - 1, axis=1)[:, :k].astype(np.int64)


def local_pca_features(xyz: np.ndarray, knn: np.ndarray) -> dict[str, np.ndarray]:
    n = xyz.shape[0]
    curvature = np.zeros((n,), dtype=np.float32)
    linearity = np.zeros((n,), dtype=np.float32)
    planarity = np.zeros((n,), dtype=np.float32)
    normals = np.zeros((n, 3), dtype=np.float32)
    eps = 1e-8
    for idx in range(n):
        ids = knn[idx]
        pts = xyz[ids] if ids.size else xyz[[idx]]
        pts = pts - pts.mean(axis=0, keepdims=True)
        cov = (pts.T @ pts) / max(1, pts.shape[0] - 1)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)
        vals = vals[order]
        vecs = vecs[:, order]
        l0, l1, l2 = [float(v) for v in vals]
        total = l0 + l1 + l2 + eps
        curvature[idx] = l0 / total
        linearity[idx] = (l2 - l1) / (l2 + eps)
        planarity[idx] = (l1 - l0) / (l2 + eps)
        normal = vecs[:, 0]
        normal = normal / (np.linalg.norm(normal) + eps)
        normals[idx] = normal.astype(np.float32)
    return {
        "curvature": curvature,
        "linearity": linearity,
        "planarity": planarity,
        "estimated_normals": normals,
    }


def quantile_mask(values: np.ndarray, q: float, side: str) -> np.ndarray:
    q = max(0.0, min(1.0, float(q)))
    threshold = float(np.quantile(values, q))
    if side == "le":
        return values <= threshold
    if side == "ge":
        return values >= threshold
    raise ValueError(f"Unknown side: {side}")


def bbox_extent_ratio(xyz: np.ndarray, mask: np.ndarray) -> list[float]:
    ids = np.where(mask)[0]
    if ids.size == 0:
        return [0.0, 0.0, 0.0]
    overall = np.ptp(xyz, axis=0) + 1e-8
    local = np.ptp(xyz[ids], axis=0)
    return (local / overall).astype(float).tolist()


def candidate_record(
    candidate_id: str,
    name: str,
    mask: np.ndarray,
    family: str,
    description: str,
    recommended_executors: list[str],
    recommended_tasks: list[str],
    xyz: np.ndarray,
    quality_hint: str = "weak",
) -> dict[str, Any]:
    point_count = int(mask.sum())
    return {
        "candidate_id": candidate_id,
        "candidate_name": name,
        "candidate_family": family,
        "description": description,
        "recommended_executors": recommended_executors,
        "recommended_tasks": recommended_tasks,
        "point_count": point_count,
        "point_fraction": float(point_count / max(1, mask.shape[0])),
        "bbox_extent_ratio": bbox_extent_ratio(xyz, mask.astype(bool)),
        "quality_hint": quality_hint,
        "provenance": "geometry_proposal_v2",
    }


def add_candidate(
    items: list[tuple[np.ndarray, dict[str, Any]]],
    mask: np.ndarray,
    name: str,
    family: str,
    description: str,
    recommended_executors: list[str],
    recommended_tasks: list[str],
    xyz: np.ndarray,
    min_points: int,
    max_fraction: float,
    quality_hint: str = "weak",
) -> None:
    mask = mask.astype(np.uint8)
    n = mask.shape[0]
    count = int(mask.sum())
    if count < min_points or count > int(n * max_fraction):
        return
    new_bool = mask.astype(bool)
    for old_mask, _ in items:
        old_bool = old_mask.astype(bool)
        union = np.logical_or(new_bool, old_bool).sum()
        if union == 0:
            continue
        iou = np.logical_and(new_bool, old_bool).sum() / union
        if iou >= 0.88:
            return
    candidate_id = LETTERS[len(items)] if len(items) < len(LETTERS) else f"C{len(items) + 1}"
    meta = candidate_record(
        candidate_id=candidate_id,
        name=name,
        mask=mask,
        family=family,
        description=description,
        recommended_executors=recommended_executors,
        recommended_tasks=recommended_tasks,
        xyz=xyz,
        quality_hint=quality_hint,
    )
    items.append((mask, meta))


def axis_name(axis: int) -> str:
    return ["x", "y", "z"][axis]


def generate_candidates(
    xyz: np.ndarray,
    normals: np.ndarray | None,
    weak_mask: np.ndarray | None,
    row: dict[str, str],
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    n = xyz.shape[0]
    knn = pairwise_knn(xyz, args.k_neighbors)
    features = local_pca_features(xyz, knn)
    curvature = features["curvature"]
    linearity = features["linearity"]
    planarity = features["planarity"]
    items: list[tuple[np.ndarray, dict[str, Any]]] = []
    task = row.get("task", "")

    if weak_mask is not None:
        for ch, executor in enumerate(EXECUTOR_ORDER):
            source = weak_mask[:, ch] > 0
            add_candidate(
                items,
                source,
                f"existing_{executor}_weak_mask",
                "existing_weak_mask",
                f"Existing checked/weak mask channel for {executor}; useful as a prior, not final truth.",
                [executor],
                [task] if task else [],
                xyz,
                args.min_points,
                args.max_candidate_fraction,
                quality_hint="source_prior",
            )

    smooth = curvature <= np.quantile(curvature, 0.35)
    planar = planarity >= np.quantile(planarity, 0.60)
    high_curv = curvature >= np.quantile(curvature, 0.82)
    linear = linearity >= np.quantile(linearity, 0.82)

    add_candidate(
        items,
        smooth & planar,
        "smooth_low_curvature_surface",
        "smooth_surface",
        "Low-curvature and planar points; potential suction or push/panel candidate.",
        ["suction"],
        ["pick_up", "lift_carry", "open_pull", "press_push"],
        xyz,
        args.min_points,
        args.max_candidate_fraction,
    )
    add_candidate(
        items,
        high_curv,
        "high_curvature_boundary",
        "edge_or_boundary",
        "High-curvature boundary points; may indicate edges, lips, holes, knobs, or handles.",
        ["gripper", "hook", "dexterous_hand"],
        ["pick_up", "lift_carry", "open_pull", "press_push"],
        xyz,
        args.min_points,
        args.max_candidate_fraction,
    )
    add_candidate(
        items,
        linear,
        "thin_or_linear_structure",
        "thin_structure",
        "Locally linear points; may correspond to stems, handles, rods, rings, or thin graspable structures.",
        ["gripper", "hook", "dexterous_hand"],
        ["pick_up", "lift_carry", "open_pull"],
        xyz,
        args.min_points,
        args.max_candidate_fraction,
    )

    for axis in range(3):
        values = xyz[:, axis]
        low_extreme = quantile_mask(values, 0.12, "le")
        high_extreme = quantile_mask(values, 0.88, "ge")
        for side, extreme in [("low", low_extreme), ("high", high_extreme)]:
            axis_label = f"{axis_name(axis)}_{side}"
            add_candidate(
                items,
                extreme & smooth,
                f"smooth_extreme_{axis_label}",
                "smooth_extreme_patch",
                f"Smooth patch near the {side} extreme of the {axis_name(axis)} axis.",
                ["suction"],
                ["pick_up", "lift_carry", "open_pull", "press_push"],
                xyz,
                args.min_points,
                args.max_candidate_fraction,
            )
            add_candidate(
                items,
                extreme & high_curv,
                f"edge_extreme_{axis_label}",
                "extreme_edge_or_lip",
                f"High-curvature points near the {side} extreme of the {axis_name(axis)} axis.",
                ["gripper", "hook", "dexterous_hand"],
                ["pick_up", "lift_carry", "open_pull"],
                xyz,
                args.min_points,
                args.max_candidate_fraction,
            )
            add_candidate(
                items,
                extreme & linear,
                f"linear_extreme_{axis_label}",
                "protruding_or_thin_part",
                f"Linear structure near the {side} extreme of the {axis_name(axis)} axis.",
                ["gripper", "hook", "dexterous_hand"],
                ["pick_up", "lift_carry", "open_pull"],
                xyz,
                args.min_points,
                args.max_candidate_fraction,
            )

    # Button/knob-like heuristic: compact high-curvature or protruding points.
    compact = np.zeros((n,), dtype=bool)
    for axis in range(3):
        values = xyz[:, axis]
        compact |= quantile_mask(values, 0.93, "ge") & high_curv
        compact |= quantile_mask(values, 0.07, "le") & high_curv
    add_candidate(
        items,
        compact,
        "compact_protrusion_or_button",
        "small_protrusion",
        "Compact protruding/high-curvature points; may be buttons, knobs, small handles, or switch-like regions.",
        ["dexterous_hand", "gripper"],
        ["press_push", "open_pull", "pick_up"],
        xyz,
        args.min_points,
        args.max_candidate_fraction,
    )

    # A conservative central-body candidate for dexterous wrapping; rules will
    # later reject it unless the task/category makes it functional.
    centered = np.ones((n,), dtype=bool)
    for axis in range(3):
        values = xyz[:, axis]
        centered &= values >= np.quantile(values, 0.20)
        centered &= values <= np.quantile(values, 0.80)
    add_candidate(
        items,
        centered,
        "central_body_wrapping_region",
        "central_body",
        "Central object body; only useful for dexterous wrapping or grasping if task semantics support it.",
        ["dexterous_hand"],
        ["pick_up", "lift_carry"],
        xyz,
        args.min_points,
        args.max_candidate_fraction,
        quality_hint="needs_strict_review",
    )

    if not items:
        raise ValueError("No candidate regions generated; relax thresholds or inspect point cloud.")
    items = items[: max(1, int(args.max_candidates))]
    masks = np.stack([item[0] for item in items], axis=0).astype(np.uint8)
    metas = [item[1] for item in items]
    # Reassign ids after max-candidate truncation to keep labels compact.
    for idx, meta in enumerate(metas):
        meta["candidate_id"] = LETTERS[idx] if idx < len(LETTERS) else f"C{idx + 1}"
    return masks, metas


def select_rows(args: argparse.Namespace, root: Path) -> list[dict[str, str]]:
    rows = read_csv(resolve_path(root, args.pilot_csv))
    if args.pilot_id:
        rows = [row for row in rows if row.get("pilot_id") == args.pilot_id]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No pilot rows selected.")
    return rows


def enrich_row(row: dict[str, str], sample_by_id: dict[str, dict[str, Any]]) -> dict[str, str]:
    sample = sample_by_id.get(row.get("sample_id", ""), {})
    merged = dict(sample)
    merged.update({k: v for k, v in row.items() if v not in (None, "")})
    return merged


def generate_for_row(
    root: Path,
    args: argparse.Namespace,
    row: dict[str, str],
    sample_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    row = enrich_row(row, sample_by_id)
    pilot_id = row["pilot_id"]
    sample_id = row["sample_id"]
    point_path = resolve_portable_path(root, row.get("point_cloud_path", ""))
    mask_value = row.get("checked_mask_path") or row.get("multi_channel_mask_path") or row.get("source_mask_path")
    mask_path = resolve_portable_path(root, mask_value) if mask_value else None
    xyz, normals = load_points(point_path)
    weak_mask = load_mask(mask_path, xyz.shape[0])
    candidate_masks, candidates = generate_candidates(xyz, normals, weak_mask, row, args)

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
        candidate_masks=candidate_masks,
    )

    manifest = {
        "version": "v2",
        "pipeline": "vlm_guided_candidate_selection",
        "pilot_id": pilot_id,
        "sample_id": sample_id,
        "object_category": row.get("object_category", ""),
        "task": row.get("task", ""),
        "executor": row.get("executor", ""),
        "issue_type": row.get("issue_type", ""),
        "point_cloud_path": relative_to_dataset(root, point_path),
        "weak_mask_path": relative_to_dataset(root, mask_path) if mask_path and mask_path.exists() else None,
        "candidate_npz": relative_to_dataset(root, npz_path),
        "candidate_count": int(candidate_masks.shape[0]),
        "candidates": candidates,
        "parameters": {
            "k_neighbors": int(args.k_neighbors),
            "min_points": int(args.min_points),
            "max_candidate_fraction": float(args.max_candidate_fraction),
            "max_candidates": int(args.max_candidates),
        },
        "notes": (
            "General 3D geometry proposals only. They are high-recall candidates, "
            "not final affordance labels."
        ),
    }
    manifest_path = output_dir / "candidate_manifest.json"
    write_json(manifest_path, manifest, args.overwrite)
    return {
        "pilot_id": pilot_id,
        "sample_id": sample_id,
        "executor": row.get("executor", ""),
        "candidate_count": int(candidate_masks.shape[0]),
        "candidate_point_counts": {
            item["candidate_id"]: int(item["point_count"]) for item in candidates
        },
        "candidate_manifest": relative_to_dataset(root, manifest_path),
    }


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    rows = select_rows(args, root)
    checked_samples = read_jsonl(resolve_path(root, args.samples))
    sample_by_id = {str(row.get("sample_id")): row for row in checked_samples}
    outputs = [generate_for_row(root, args, row, sample_by_id) for row in rows]
    print(json.dumps({"generated_rows": len(outputs), "rows": outputs}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
