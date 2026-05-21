#!/usr/bin/env python3
"""Grow v3 3D candidates from target seeds while respecting reject veto masks."""

from __future__ import annotations

import argparse
import csv
import json
import string
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from path_utils import relative_to_dataset, resolve_portable_path


LETTERS = list(string.ascii_uppercase)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grow v3 reject-aware 3D candidate regions.")
    parser.add_argument("--dataset-root", default=".", help="Dataset root directory.")
    parser.add_argument("--pilot-csv", default="processed/metadata/vlm_pilot_samples_v0_1.csv")
    parser.add_argument("--semantic-plan-root", default="processed/vlm_candidate_v3/semantic_plans")
    parser.add_argument("--projected-root", default="processed/vlm_candidate_v3/projected_3d")
    parser.add_argument("--output-root", default="processed/vlm_candidate_v3/3d_candidates")
    parser.add_argument("--pilot-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--k-neighbors", type=int, default=24)
    parser.add_argument("--target-score-threshold", type=float, default=0.20)
    parser.add_argument("--reject-score-threshold", type=float, default=0.10)
    parser.add_argument("--min-target-votes", type=float, default=1.0)
    parser.add_argument("--min-reject-votes", type=float, default=1.0)
    parser.add_argument("--expand-hops", type=int, default=1)
    parser.add_argument("--min-points", type=int, default=4)
    parser.add_argument("--max-candidate-fraction", type=float, default=0.45)
    parser.add_argument("--max-components", type=int, default=6)
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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def selected_rows(root: Path, args: argparse.Namespace) -> list[dict[str, str]]:
    rows = read_csv(resolve_path(root, args.pilot_csv))
    if args.pilot_id:
        rows = [row for row in rows if row.get("pilot_id") == args.pilot_id]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No pilot rows selected.")
    return rows


def load_points(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Point cloud not found: {path}")
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] not in (3, 6):
        raise ValueError(f"Expected point cloud shape [N,3] or [N,6], got {arr.shape}: {path}")
    return arr[:, :3].astype(np.float32)


def pairwise_knn(xyz: np.ndarray, k: int) -> np.ndarray:
    n = xyz.shape[0]
    if n <= 1:
        return np.zeros((n, 0), dtype=np.int64)
    k = max(1, min(int(k), n - 1))
    diff = xyz[:, None, :] - xyz[None, :, :]
    dist2 = np.einsum("ijk,ijk->ij", diff, diff)
    np.fill_diagonal(dist2, np.inf)
    return np.argpartition(dist2, kth=k - 1, axis=1)[:, :k].astype(np.int64)


def connected_components(mask: np.ndarray, knn: np.ndarray, max_components: int) -> list[np.ndarray]:
    active = mask.astype(bool)
    visited = np.zeros(active.shape[0], dtype=bool)
    components: list[np.ndarray] = []
    active_ids = np.where(active)[0]
    for start in active_ids:
        if visited[start]:
            continue
        q: deque[int] = deque([int(start)])
        visited[start] = True
        ids: list[int] = []
        while q:
            cur = q.popleft()
            ids.append(cur)
            for nb in knn[cur]:
                nb = int(nb)
                if active[nb] and not visited[nb]:
                    visited[nb] = True
                    q.append(nb)
        comp = np.zeros(active.shape[0], dtype=np.uint8)
        comp[ids] = 1
        components.append(comp)
    components.sort(key=lambda m: int(m.sum()), reverse=True)
    return components[: max(1, int(max_components))]


def expand_without_veto(seed: np.ndarray, knn: np.ndarray, allowed: np.ndarray, hops: int) -> np.ndarray:
    out = seed.astype(bool).copy() & allowed.astype(bool)
    for _ in range(max(0, int(hops))):
        ids = np.where(out)[0]
        if ids.size == 0:
            break
        expanded = out.copy()
        expanded[knn[ids].reshape(-1)] = True
        out = expanded & allowed.astype(bool)
    return out.astype(np.uint8)


def bbox_extent_ratio(xyz: np.ndarray, mask: np.ndarray) -> list[float]:
    ids = np.where(mask.astype(bool))[0]
    if ids.size == 0:
        return [0.0, 0.0, 0.0]
    overall = np.ptp(xyz, axis=0) + 1e-8
    local = np.ptp(xyz[ids], axis=0)
    return (local / overall).astype(float).tolist()


def candidate_record(
    candidate_id: str,
    name: str,
    family: str,
    mask: np.ndarray,
    xyz: np.ndarray,
    row: dict[str, str],
    description: str,
    priority: float,
    target_scores: np.ndarray,
    reject_scores: np.ndarray,
) -> dict[str, Any]:
    ids = np.where(mask.astype(bool))[0]
    point_count = int(ids.size)
    mean_target = float(target_scores[ids].mean()) if ids.size else 0.0
    max_reject = float(reject_scores[ids].max()) if ids.size else 0.0
    return {
        "candidate_id": candidate_id,
        "candidate_name": name,
        "candidate_family": family,
        "description": description,
        "recommended_executors": [row.get("executor", "")],
        "recommended_tasks": [row.get("task", "")],
        "point_count": point_count,
        "point_fraction": float(point_count / max(1, mask.shape[0])),
        "bbox_extent_ratio": bbox_extent_ratio(xyz, mask),
        "quality_hint": "v3_reject_aware_candidate",
        "priority": float(priority),
        "mean_target_score": mean_target,
        "max_reject_score_after_veto": max_reject,
        "provenance": "v3_target_seed_growth_with_reject_veto",
    }


def add_candidate(
    items: list[tuple[np.ndarray, dict[str, Any]]],
    mask: np.ndarray,
    name: str,
    family: str,
    xyz: np.ndarray,
    row: dict[str, str],
    description: str,
    priority: float,
    target_scores: np.ndarray,
    reject_scores: np.ndarray,
    min_points: int,
    max_fraction: float,
) -> None:
    mask = mask.astype(np.uint8)
    n = mask.shape[0]
    count = int(mask.sum())
    if count < int(min_points) or count > int(n * float(max_fraction)):
        return
    new_bool = mask.astype(bool)
    for old_mask, _ in items:
        old_bool = old_mask.astype(bool)
        union = np.logical_or(new_bool, old_bool).sum()
        if union == 0:
            continue
        iou = np.logical_and(new_bool, old_bool).sum() / union
        if iou >= 0.90:
            return
    cid = LETTERS[len(items)] if len(items) < len(LETTERS) else f"C{len(items) + 1}"
    items.append(
        (
            mask,
            candidate_record(
                cid,
                name,
                family,
                mask,
                xyz,
                row,
                description,
                priority,
                target_scores,
                reject_scores,
            ),
        )
    )


def generate_candidates(row: dict[str, str], xyz: np.ndarray, projected: Any, plan: dict[str, Any], args: argparse.Namespace) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray, np.ndarray, list[str]]:
    target_votes = projected["target_votes"].astype(np.float32)
    reject_votes = projected["reject_votes"].astype(np.float32)
    target_scores = projected["target_scores"].astype(np.float32)
    reject_scores = projected["reject_scores"].astype(np.float32)
    n = target_votes.shape[0]
    if xyz.shape[0] != n:
        raise ValueError(f"Point count {xyz.shape[0]} does not match projected votes {n}")

    target_seed = (target_votes >= float(args.min_target_votes)) | (target_scores >= float(args.target_score_threshold))
    reject_veto = (reject_votes >= float(args.min_reject_votes)) | (reject_scores >= float(args.reject_score_threshold))
    clean_seed = target_seed & ~reject_veto
    allowed = ~reject_veto
    knn = pairwise_knn(xyz, args.k_neighbors)
    components = connected_components(clean_seed, knn, args.max_components) if clean_seed.any() else []
    items: list[tuple[np.ndarray, dict[str, Any]]] = []
    default_selected: list[str] = []

    if clean_seed.any():
        expanded_union = expand_without_veto(clean_seed.astype(np.uint8), knn, allowed, args.expand_hops)
        add_candidate(
            items,
            expanded_union,
            "target_seed_union_reject_aware_expanded",
            "semantic_target_union",
            xyz,
            row,
            "Union of all VLM target seeds after removing reject-veto points and applying small kNN expansion.",
            20.0,
            target_scores,
            reject_scores,
            args.min_points,
            args.max_candidate_fraction,
        )
        add_candidate(
            items,
            clean_seed.astype(np.uint8),
            "target_seed_core_without_reject",
            "semantic_target_core",
            xyz,
            row,
            "Raw 3D target seeds after hard reject-veto removal.",
            30.0,
            target_scores,
            reject_scores,
            args.min_points,
            args.max_candidate_fraction,
        )

    expanded_component_ids: list[str] = []
    for idx, comp in enumerate(components, start=1):
        expanded = expand_without_veto(comp, knn, allowed, args.expand_hops)
        before = len(items)
        add_candidate(
            items,
            expanded,
            f"target_component_{idx:02d}_expanded",
            "semantic_target_component_expanded",
            xyz,
            row,
            "Connected target component grown locally without crossing reject-veto points.",
            8.0 + idx * 0.1,
            target_scores,
            reject_scores,
            args.min_points,
            args.max_candidate_fraction,
        )
        if len(items) > before:
            expanded_component_ids.append(items[-1][1]["candidate_id"])
        add_candidate(
            items,
            comp,
            f"target_component_{idx:02d}_core",
            "semantic_target_component_core",
            xyz,
            row,
            "Connected core target seed component before local expansion.",
            18.0 + idx * 0.1,
            target_scores,
            reject_scores,
            args.min_points,
            args.max_candidate_fraction,
        )

    if len(components) >= 2 and row.get("executor") in {"hook", "gripper", "dexterous_hand"}:
        paired = np.zeros((n,), dtype=np.uint8)
        for comp in components[:2]:
            paired |= expand_without_veto(comp, knn, allowed, args.expand_hops)
        before = len(items)
        add_candidate(
            items,
            paired,
            "paired_target_components_reject_aware",
            "semantic_paired_target_components",
            xyz,
            row,
            "Union of two strongest semantic target components; useful for paired handles, rings, or two-finger structures.",
            6.0,
            target_scores,
            reject_scores,
            args.min_points,
            min(float(args.max_candidate_fraction), 0.60),
        )
        if len(items) > before:
            default_selected = [items[-1][1]["candidate_id"]]

    items.sort(key=lambda item: (float(item[1].get("priority", 50.0)), -int(item[1].get("point_count", 0))))
    for idx, (_, meta) in enumerate(items):
        meta["candidate_id"] = LETTERS[idx] if idx < len(LETTERS) else f"C{idx + 1}"

    if default_selected:
        # Recompute default after id reassignment using candidate name.
        default_selected = [meta["candidate_id"] for _, meta in items if meta["candidate_name"] == "paired_target_components_reject_aware"]
    if not default_selected:
        default_selected = [meta["candidate_id"] for _, meta in items if meta["candidate_family"] == "semantic_target_component_expanded"][:3]
    if not default_selected and items:
        default_selected = [items[0][1]["candidate_id"]]

    candidate_masks = np.stack([mask for mask, _ in items], axis=0).astype(np.uint8) if items else np.zeros((0, n), dtype=np.uint8)
    candidates = [meta for _, meta in items]
    return candidate_masks, candidates, clean_seed.astype(np.uint8), reject_veto.astype(np.uint8), default_selected


def grow_one(root: Path, args: argparse.Namespace, row: dict[str, str]) -> dict[str, Any]:
    pilot_id = row["pilot_id"]
    projected_path = resolve_path(root, args.projected_root) / f"{pilot_id}_target_reject_votes.npz"
    if not projected_path.exists():
        raise FileNotFoundError(f"Projected v3 votes not found: {projected_path}")
    projected = np.load(projected_path, allow_pickle=True)
    plan_path = resolve_path(root, args.semantic_plan_root) / pilot_id / "combined_semantic_plan.json"
    plan = read_json(plan_path)
    point_path = resolve_portable_path(root, row.get("point_cloud_path", ""))
    xyz = load_points(point_path)
    candidate_masks, candidates, target_seed, reject_veto, default_selected = generate_candidates(row, xyz, projected, plan, args)

    output_dir = resolve_path(root, args.output_root) / pilot_id
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "candidates.npz"
    if npz_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {npz_path}")
    np.savez_compressed(
        npz_path,
        pilot_id=pilot_id,
        sample_id=row["sample_id"],
        executor=row.get("executor", ""),
        candidate_ids=np.asarray([item["candidate_id"] for item in candidates], dtype=object),
        candidate_names=np.asarray([item["candidate_name"] for item in candidates], dtype=object),
        candidate_families=np.asarray([item["candidate_family"] for item in candidates], dtype=object),
        candidate_masks=candidate_masks,
        target_seed_mask=target_seed,
        reject_veto_mask=reject_veto,
        target_scores=projected["target_scores"],
        reject_scores=projected["reject_scores"],
    )
    manifest = {
        "version": "v3",
        "pipeline": "reject_aware_semantic_candidate_growth",
        "pilot_id": pilot_id,
        "sample_id": row["sample_id"],
        "object_category": row.get("object_category", ""),
        "task": row.get("task", ""),
        "executor": row.get("executor", ""),
        "point_cloud_path": relative_to_dataset(root, point_path),
        "semantic_plan": relative_to_dataset(root, plan_path),
        "projected_votes": relative_to_dataset(root, projected_path),
        "candidate_npz": relative_to_dataset(root, npz_path),
        "candidate_count": int(candidate_masks.shape[0]),
        "default_selected_candidates": default_selected,
        "target_seed_points": int(target_seed.sum()),
        "reject_veto_points": int(reject_veto.sum()),
        "candidates": candidates,
        "parameters": {
            "target_score_threshold": float(args.target_score_threshold),
            "reject_score_threshold": float(args.reject_score_threshold),
            "min_target_votes": float(args.min_target_votes),
            "min_reject_votes": float(args.min_reject_votes),
            "expand_hops": int(args.expand_hops),
            "k_neighbors": int(args.k_neighbors),
        },
        "notes": (
            "v3 candidates grow only from semantic target seeds and are clipped by reject-veto points. "
            "They remain proposals requiring human review."
        ),
    }
    manifest_path = output_dir / "candidate_manifest.json"
    write_json(manifest_path, manifest, args.overwrite)
    return {
        "pilot_id": pilot_id,
        "sample_id": row["sample_id"],
        "executor": row.get("executor", ""),
        "candidate_count": int(candidate_masks.shape[0]),
        "default_selected_candidates": default_selected,
        "target_seed_points": int(target_seed.sum()),
        "reject_veto_points": int(reject_veto.sum()),
        "candidate_manifest": relative_to_dataset(root, manifest_path),
    }


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    rows = selected_rows(root, args)
    outputs = [grow_one(root, args, row) for row in rows]
    print(json.dumps({"rows": len(outputs), "outputs": outputs}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
