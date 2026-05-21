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
small protrusions, loop/handle components, expanded seed regions, and existing
weak-label channels. These candidates are not ground truth.
"""

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
    parser.add_argument(
        "--renders-root",
        default="processed/vlm_semantic_part/renders",
        help="Optional render root used for view-based component proposals.",
    )
    parser.add_argument(
        "--fallback-renders-root",
        default="processed/vlm_pilot/renders",
        help="Fallback render root for view-based component proposals.",
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
    parser.add_argument("--max-candidates", type=int, default=36, help="Maximum candidates kept per pilot row.")
    parser.add_argument(
        "--seed-expand-hops",
        type=int,
        default=1,
        help="kNN expansion hops used to turn sparse seed points into reviewable region candidates.",
    )
    parser.add_argument(
        "--component-max-candidates",
        type=int,
        default=8,
        help="Maximum connected seed components converted to separate part-level candidates.",
    )
    parser.add_argument("--view-dilation-radius", type=int, default=3, help="2D dilation radius for visual components.")
    parser.add_argument("--body-row-threshold-ratio", type=float, default=0.18, help="Dense-row threshold for body top.")
    parser.add_argument("--body-top-margin", type=int, default=8, help="Margin above estimated body top.")
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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def render_manifest_path(root: Path, args: argparse.Namespace, sample_id: str) -> Path | None:
    preferred = resolve_path(root, args.renders_root) / sample_id / "view_manifest.json"
    if preferred.exists():
        return preferred
    fallback = resolve_path(root, args.fallback_renders_root) / sample_id / "view_manifest.json"
    if fallback.exists():
        return fallback
    return None


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


def knn_expand_mask(mask: np.ndarray, knn: np.ndarray, hops: int) -> np.ndarray:
    """Expand sparse seed points on the point-cloud kNN graph."""
    out = mask.astype(bool).copy()
    frontier = out.copy()
    hops = max(0, int(hops))
    n = out.shape[0]
    for _ in range(hops):
        ids = np.where(frontier)[0]
        if ids.size == 0 or knn.size == 0:
            break
        nbrs = knn[ids].reshape(-1)
        nbrs = nbrs[(nbrs >= 0) & (nbrs < n)]
        new_frontier = np.zeros((n,), dtype=bool)
        new_frontier[nbrs] = True
        new_frontier &= ~out
        if not np.any(new_frontier):
            break
        out |= new_frontier
        frontier = new_frontier
    return out.astype(np.uint8)


def graph_components_from_seed(seed: np.ndarray, knn: np.ndarray, min_points: int, max_points: int) -> list[np.ndarray]:
    """Split a seed mask into connected components on the kNN graph."""
    seed = seed.astype(bool)
    ids = np.where(seed)[0]
    if ids.size == 0:
        return []
    visited = np.zeros(seed.shape[0], dtype=bool)
    components: list[np.ndarray] = []
    for start in ids.tolist():
        if visited[start]:
            continue
        q: deque[int] = deque([start])
        visited[start] = True
        comp: list[int] = []
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nbr in knn[cur].tolist():
                if nbr < 0 or nbr >= seed.shape[0] or visited[nbr] or not seed[nbr]:
                    continue
                visited[nbr] = True
                q.append(int(nbr))
        if min_points <= len(comp) <= max_points:
            components.append(np.asarray(comp, dtype=np.int64))
    components.sort(key=lambda arr: int(arr.size), reverse=True)
    return components


def mask_from_ids(n: int, ids: np.ndarray) -> np.ndarray:
    mask = np.zeros((n,), dtype=np.uint8)
    safe = ids[(ids >= 0) & (ids < n)].astype(np.int64)
    mask[safe] = 1
    return mask


def dilate_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    if radius <= 0:
        return mask.astype(bool)
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    r2 = radius * radius
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > r2:
                continue
            src_y0 = max(0, -dy)
            src_y1 = min(h, h - dy)
            src_x0 = max(0, -dx)
            src_x1 = min(w, w - dx)
            dst_y0 = max(0, dy)
            dst_y1 = min(h, h + dy)
            dst_x0 = max(0, dx)
            dst_x1 = min(w, w + dx)
            out[dst_y0:dst_y1, dst_x0:dst_x1] |= mask[src_y0:src_y1, src_x0:src_x1]
    return out


def connected_components(mask: np.ndarray) -> list[dict[str, Any]]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    components: list[dict[str, Any]] = []
    ys, xs = np.where(mask)
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if visited[start_y, start_x]:
            continue
        q: deque[tuple[int, int]] = deque([(start_y, start_x)])
        visited[start_y, start_x] = True
        pixels: list[tuple[int, int]] = []
        while q:
            y, x = q.popleft()
            pixels.append((y, x))
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    q.append((ny, nx))
        py = np.asarray([p[0] for p in pixels], dtype=np.int32)
        px = np.asarray([p[1] for p in pixels], dtype=np.int32)
        components.append(
            {
                "pixels": pixels,
                "area": int(len(pixels)),
                "bbox": [int(px.min()), int(py.min()), int(px.max()), int(py.max())],
                "center": [float(px.mean()), float(py.mean())],
            }
        )
    return components


def ids_from_pixels(index_map: np.ndarray, pixels: list[tuple[int, int]]) -> np.ndarray:
    if not pixels:
        return np.asarray([], dtype=np.int64)
    ys = np.asarray([p[0] for p in pixels], dtype=np.int64)
    xs = np.asarray([p[1] for p in pixels], dtype=np.int64)
    ids = index_map[ys, xs]
    ids = ids[ids >= 0]
    if ids.size == 0:
        return np.asarray([], dtype=np.int64)
    return np.unique(ids.astype(np.int64))


def above_body_ids_from_view(index_map: np.ndarray, row_threshold_ratio: float, margin: int) -> np.ndarray:
    valid = index_map >= 0
    if not np.any(valid):
        return np.asarray([], dtype=np.int64)
    row_counts = valid.sum(axis=1)
    threshold = max(6, int(round(float(row_counts.max()) * float(row_threshold_ratio))))
    dense_rows = np.where(row_counts >= threshold)[0]
    if dense_rows.size == 0:
        return np.asarray([], dtype=np.int64)
    body_top = int(dense_rows.min())
    cutoff = max(0, body_top - max(0, int(margin)))
    ids = index_map[:cutoff, :]
    ids = ids[ids >= 0]
    if ids.size == 0:
        return np.asarray([], dtype=np.int64)
    return np.unique(ids.astype(np.int64))


def render_component_candidate_ids(
    root: Path,
    args: argparse.Namespace,
    sample_id: str,
    num_points: int,
) -> dict[str, np.ndarray]:
    manifest_path = render_manifest_path(root, args, sample_id)
    if manifest_path is None:
        return {}
    manifest = read_json(manifest_path)
    above_chunks: list[np.ndarray] = []
    detached_chunks: list[np.ndarray] = []
    small_component_chunks: list[np.ndarray] = []

    for entry in manifest.get("views", []):
        index_path = resolve_portable_path(root, entry["point_index_path"], manifest_path.parent)
        if not index_path.exists():
            continue
        index_map = np.load(index_path)
        above_ids = above_body_ids_from_view(index_map, args.body_row_threshold_ratio, args.body_top_margin)
        if above_ids.size:
            above_chunks.append(above_ids)

        foreground = dilate_bool(index_map >= 0, args.view_dilation_radius)
        comps = connected_components(foreground)
        if not comps:
            continue
        largest_area = max(int(comp["area"]) for comp in comps)
        h = index_map.shape[0]
        for comp in comps:
            ids = ids_from_pixels(index_map, comp["pixels"])
            if ids.size < args.min_points:
                continue
            if ids.size > max(1, int(num_points * 0.22)):
                continue
            _, y1, _, y2 = comp["bbox"]
            center_y = float(comp["center"][1])
            is_detached = int(comp["area"]) < largest_area * 0.45
            is_upper = center_y <= h * 0.58 or y1 <= h * 0.42
            if is_detached and is_upper:
                detached_chunks.append(ids)
            if int(comp["area"]) < largest_area * 0.18:
                small_component_chunks.append(ids)

    def union(chunks: list[np.ndarray]) -> np.ndarray:
        if not chunks:
            return np.asarray([], dtype=np.int64)
        return np.unique(np.concatenate(chunks).astype(np.int64))

    return {
        "above_main_body_structure": union(above_chunks),
        "detached_upper_or_side_component": union(detached_chunks),
        "small_visual_component": union(small_component_chunks),
    }


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
    priority: float = 50.0,
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
        "priority": float(priority),
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
    priority: float = 50.0,
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
        priority=priority,
    )
    items.append((mask, meta))


def add_expanded_candidate(
    items: list[tuple[np.ndarray, dict[str, Any]]],
    seed_mask: np.ndarray,
    knn: np.ndarray,
    hops: int,
    name: str,
    family: str,
    description: str,
    recommended_executors: list[str],
    recommended_tasks: list[str],
    xyz: np.ndarray,
    min_points: int,
    max_fraction: float,
    quality_hint: str = "expanded_seed",
    priority: float = 25.0,
) -> None:
    expanded = knn_expand_mask(seed_mask.astype(bool), knn, hops)
    add_candidate(
        items,
        expanded,
        name,
        family,
        description,
        recommended_executors,
        recommended_tasks,
        xyz,
        min_points,
        max_fraction,
        quality_hint=quality_hint,
        priority=priority,
    )


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
    expand_hops = max(0, int(args.seed_expand_hops))

    view_ids = render_component_candidate_ids(
        root=Path(args.dataset_root).resolve(),
        args=args,
        sample_id=row["sample_id"],
        num_points=n,
    )
    above_main = view_ids.get("above_main_body_structure", np.asarray([], dtype=np.int64))
    detached = view_ids.get("detached_upper_or_side_component", np.asarray([], dtype=np.int64))
    small_visual = view_ids.get("small_visual_component", np.asarray([], dtype=np.int64))
    if above_main.size:
        mask = mask_from_ids(n, above_main)
        add_expanded_candidate(
            items,
            mask,
            knn,
            expand_hops,
            "above_main_body_structure_expanded",
            "visual_component_expanded",
            "kNN-expanded upper structure proposal; intended to make sparse handles, loops, rings, or top protrusions reviewable.",
            ["hook", "gripper", "dexterous_hand"],
            ["pick_up", "lift_carry", "open_pull"],
            xyz,
            args.min_points,
            min(float(args.max_candidate_fraction), 0.36),
            quality_hint="expanded_visual_proposal",
            priority=8.0,
        )
        add_candidate(
            items,
            mask,
            "above_main_body_structure",
            "visual_detached_structure",
            "View-derived points above the dense main body; useful for handles, loops, rings, or top protrusions.",
            ["hook", "gripper", "dexterous_hand"],
            ["pick_up", "lift_carry", "open_pull"],
            xyz,
            args.min_points,
            min(float(args.max_candidate_fraction), 0.30),
            quality_hint="targeted_visual_proposal",
            priority=12.0,
        )
    if detached.size:
        mask = mask_from_ids(n, detached)
        add_expanded_candidate(
            items,
            mask,
            knn,
            expand_hops,
            "detached_upper_or_side_component_expanded",
            "visual_component_expanded",
            "kNN-expanded detached visual component; useful when a handle, ring, knob, or protruding part is sparse in point render.",
            ["hook", "gripper", "dexterous_hand"],
            ["pick_up", "lift_carry", "open_pull", "press_push"],
            xyz,
            args.min_points,
            min(float(args.max_candidate_fraction), 0.40),
            quality_hint="expanded_visual_proposal",
            priority=9.0,
        )
        add_candidate(
            items,
            mask,
            "detached_upper_or_side_component",
            "visual_detached_component",
            "Small detached visual component from multi-view point-index maps; may be a handle, ring, knob, or protruding part.",
            ["hook", "gripper", "dexterous_hand"],
            ["pick_up", "lift_carry", "open_pull", "press_push"],
            xyz,
            args.min_points,
            min(float(args.max_candidate_fraction), 0.35),
            quality_hint="targeted_visual_proposal",
            priority=13.0,
        )
    if small_visual.size:
        mask = mask_from_ids(n, small_visual)
        add_expanded_candidate(
            items,
            mask,
            knn,
            expand_hops,
            "small_visual_component_expanded",
            "visual_component_expanded",
            "kNN-expanded small visual component; high-recall proposal for small functional parts.",
            ["hook", "gripper", "dexterous_hand"],
            ["pick_up", "lift_carry", "open_pull", "press_push"],
            xyz,
            args.min_points,
            min(float(args.max_candidate_fraction), 0.40),
            quality_hint="expanded_visual_proposal",
            priority=14.0,
        )
        add_candidate(
            items,
            mask,
            "small_visual_component",
            "visual_small_component",
            "Small foreground components seen in rendered views; high-recall proposal for small functional parts.",
            ["hook", "gripper", "dexterous_hand"],
            ["pick_up", "lift_carry", "open_pull", "press_push"],
            xyz,
            args.min_points,
            min(float(args.max_candidate_fraction), 0.35),
            quality_hint="targeted_visual_proposal",
            priority=18.0,
        )

    if weak_mask is not None:
        for ch, executor in enumerate(EXECUTOR_ORDER):
            source = weak_mask[:, ch] > 0
            recommended = [executor]
            if executor in {"gripper", "dexterous_hand"}:
                recommended.extend(["hook", "dexterous_hand", "gripper"])
            recommended = list(dict.fromkeys(recommended))
            add_expanded_candidate(
                items,
                source,
                knn,
                expand_hops,
                f"existing_{executor}_weak_mask_expanded",
                "expanded_existing_weak_mask",
                (
                    f"kNN-expanded checked/weak mask channel for {executor}; useful when the existing prior is sparse "
                    "but spatially close to a functional part."
                ),
                recommended,
                [task] if task else [],
                xyz,
                args.min_points,
                args.max_candidate_fraction,
                quality_hint="expanded_source_prior",
                priority=28.0,
            )
            add_candidate(
                items,
                source,
                f"existing_{executor}_weak_mask",
                "existing_weak_mask",
                (
                    f"Existing checked/weak mask channel for {executor}; useful as a spatial prior, "
                    "not final truth. A gripper/hand prior may overlap handles that are relevant to hook."
                ),
                recommended,
                [task] if task else [],
                xyz,
                args.min_points,
                args.max_candidate_fraction,
                quality_hint="source_prior",
                priority=34.0,
            )

    smooth = curvature <= np.quantile(curvature, 0.35)
    planar = planarity >= np.quantile(planarity, 0.60)
    high_curv = curvature >= np.quantile(curvature, 0.82)
    linear = linearity >= np.quantile(linearity, 0.82)
    edge_linear_seed = high_curv | linear

    edge_linear_components = graph_components_from_seed(
        edge_linear_seed,
        knn,
        min_points=max(args.min_points, 8),
        max_points=max(args.min_points, int(n * 0.35)),
    )
    component_limit = max(0, int(args.component_max_candidates))
    component_records: list[tuple[np.ndarray, float]] = []
    for comp_idx, ids in enumerate(edge_linear_components[:component_limit], start=1):
        seed = mask_from_ids(n, ids)
        expanded = knn_expand_mask(seed, knn, expand_hops)
        ext = bbox_extent_ratio(xyz, expanded.astype(bool))
        sorted_ext = sorted([float(v) for v in ext], reverse=True)
        elongation = sorted_ext[0] / (sorted_ext[1] + 1e-6) if len(sorted_ext) > 1 else 999.0
        balance_bonus = max(0.0, 4.0 - min(4.0, elongation))
        if int(expanded.sum()) <= int(n * 0.35):
            component_records.append((expanded, float(ids.size) + balance_bonus * 20.0))
        add_candidate(
            items,
            expanded,
            f"loop_or_handle_component_{comp_idx:02d}_expanded",
            "expanded_loop_or_handle",
            (
                "kNN-expanded connected high-curvature/linear component; designed for complete handles, rings, "
                "holes, loops, lips, or similar functional structures."
            ),
            ["hook", "gripper", "dexterous_hand"],
            ["pick_up", "lift_carry", "open_pull", "press_push"],
            xyz,
            args.min_points,
            min(float(args.max_candidate_fraction), 0.42),
            quality_hint="part_level_expanded_candidate",
            priority=20.0 + comp_idx * 0.1,
        )
        add_candidate(
            items,
            seed,
            f"loop_or_handle_component_{comp_idx:02d}_seed",
            "loop_or_hole_boundary",
            "Connected high-curvature/linear seed component before expansion; useful for precise boundary review.",
            ["hook", "gripper", "dexterous_hand"],
            ["pick_up", "lift_carry", "open_pull", "press_push"],
            xyz,
            args.min_points,
            min(float(args.max_candidate_fraction), 0.35),
            quality_hint="part_level_seed_candidate",
            priority=44.0 + comp_idx * 0.1,
        )

    if len(component_records) >= 2:
        component_records.sort(key=lambda item: item[1], reverse=True)
        paired_mask = np.zeros((n,), dtype=np.uint8)
        for comp_mask, _ in component_records[:2]:
            paired_mask |= comp_mask.astype(np.uint8)
        add_candidate(
            items,
            paired_mask,
            "paired_loop_or_handle_components",
            "paired_loop_or_handle",
            (
                "Union of two strong loop/handle-like components. This is a high-recall proposal for paired handles, "
                "double rings, scissor finger holes, and symmetric grasp/hook structures."
            ),
            ["hook", "gripper", "dexterous_hand"],
            ["pick_up", "lift_carry", "open_pull"],
            xyz,
            args.min_points,
            min(float(args.max_candidate_fraction), 0.55),
            quality_hint="paired_part_candidate",
            priority=16.0,
        )

    for axis in range(3):
        values = xyz[:, axis]
        median = float(np.median(values))
        for side, spatial_mask in [("low", values <= median), ("high", values >= median)]:
            seed = edge_linear_seed & spatial_mask
            add_expanded_candidate(
                items,
                seed,
                knn,
                expand_hops,
                f"edge_linear_half_{axis_name(axis)}_{side}_expanded",
                "expanded_axis_part_component",
                (
                    f"kNN-expanded high-curvature/linear points in the {side} half of the {axis_name(axis)} axis. "
                    "This high-recall split helps expose paired handles, double rings, and symmetric functional parts "
                    "that may be connected through a central joint."
                ),
                ["hook", "gripper", "dexterous_hand"],
                ["pick_up", "lift_carry", "open_pull", "press_push"],
                xyz,
                args.min_points,
                min(float(args.max_candidate_fraction), 0.34),
                quality_hint="axis_split_expanded_candidate",
                priority=36.0 + axis * 0.2 + (0.0 if side == "low" else 0.1),
            )

    for axis in range(3):
        values = xyz[:, axis]
        for side, extreme in [("low", quantile_mask(values, 0.18, "le")), ("high", quantile_mask(values, 0.82, "ge"))]:
            seed = edge_linear_seed & extreme
            add_expanded_candidate(
                items,
                seed,
                knn,
                expand_hops,
                f"edge_linear_extreme_{axis_name(axis)}_{side}_expanded",
                "expanded_extreme_part_component",
                (
                    f"kNN-expanded high-curvature/linear points near the {side} extreme of the {axis_name(axis)} axis. "
                    "Useful for handles, loops, lips, knobs, and side/end functional parts."
                ),
                ["hook", "gripper", "dexterous_hand"],
                ["pick_up", "lift_carry", "open_pull", "press_push"],
                xyz,
                args.min_points,
                min(float(args.max_candidate_fraction), 0.32),
                quality_hint="extreme_expanded_candidate",
                priority=22.0 + axis * 0.2 + (0.0 if side == "low" else 0.1),
            )

    add_expanded_candidate(
        items,
        edge_linear_seed,
        knn,
        expand_hops,
        "edge_linear_seed_expanded",
        "expanded_functional_seed",
        (
            "Expanded union of high-curvature and linear seeds. This is a broad fallback for manual review when "
            "individual part components miss a functional handle, ring, lip, or boundary."
        ),
        ["hook", "gripper", "dexterous_hand"],
        ["pick_up", "lift_carry", "open_pull", "press_push"],
        xyz,
        args.min_points,
        min(float(args.max_candidate_fraction), 0.62),
        quality_hint="broad_review_fallback",
        priority=32.0,
    )

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
        priority=55.0,
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
        priority=48.0,
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
        priority=49.0,
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
                priority=60.0,
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
                priority=52.0,
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
                priority=53.0,
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
        priority=42.0,
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
        priority=80.0,
    )

    if not items:
        raise ValueError("No candidate regions generated; relax thresholds or inspect point cloud.")
    items.sort(key=lambda item: (float(item[1].get("priority", 50.0)), -int(item[1].get("point_count", 0))))
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
            "seed_expand_hops": int(args.seed_expand_hops),
            "component_max_candidates": int(args.component_max_candidates),
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
