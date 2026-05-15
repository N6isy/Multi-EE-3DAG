#!/usr/bin/env python3
"""Generate geometry-first hook candidates from rendered point-index maps.

This script does not assume that an existing dataset already contains hook
masks. It finds candidate hookable structures by looking for small upper
foreground components in multi-view point-cloud renders, then writes several
3D candidate masks for later VLM selection and human review.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from path_utils import relative_to_dataset, resolve_portable_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 3D geometry candidates for hook affordance.")
    parser.add_argument("--dataset-root", default=".", help="Dataset root directory.")
    parser.add_argument(
        "--pilot-csv",
        default="processed/metadata/vlm_pilot_samples_v0_1.csv",
        help="Pilot CSV relative to dataset root.",
    )
    parser.add_argument(
        "--renders-root",
        default="processed/vlm_pilot/renders",
        help="Render root relative to dataset root.",
    )
    parser.add_argument(
        "--output-root",
        default="processed/vlm_pilot/hook_candidates",
        help="Output root relative to dataset root.",
    )
    parser.add_argument("--pilot-id", default=None, help="Generate only one pilot row.")
    parser.add_argument("--limit", type=int, default=None, help="Limit selected hook pilot rows.")
    parser.add_argument("--executor", default="hook", help="Executor to process. Default: hook.")
    parser.add_argument("--dilation-radius", type=int, default=4, help="2D dilation radius before component search.")
    parser.add_argument("--upper-ratio", type=float, default=0.58, help="Component center-y must be in this upper image ratio.")
    parser.add_argument("--strict-upper-ratio", type=float, default=0.45, help="Strict upper threshold for candidate A.")
    parser.add_argument("--max-component-fraction", type=float, default=0.30, help="Reject huge body-like components.")
    parser.add_argument("--min-points", type=int, default=3, help="Minimum unique 3D points per component.")
    parser.add_argument("--max-components-per-view", type=int, default=3, help="Keep top components per view.")
    parser.add_argument("--top-band-ratio", type=float, default=0.20, help="Fallback upper-band ratio for candidate C.")
    parser.add_argument(
        "--body-row-threshold-ratio",
        type=float,
        default=0.18,
        help="Row-density ratio for estimating the top of the main object body.",
    )
    parser.add_argument(
        "--body-top-margin",
        type=int,
        default=8,
        help="Pixel margin above the estimated body top for above-body hook candidates.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite candidate outputs.")
    return parser.parse_args()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
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


def dilate_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    if radius <= 0:
        return mask.astype(bool)
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    radius2 = radius * radius
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius2:
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
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {mask.shape}")
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict[str, Any]] = []
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

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


def component_mask(shape: tuple[int, int], pixels: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if pixels:
        ys = [p[0] for p in pixels]
        xs = [p[1] for p in pixels]
        mask[ys, xs] = True
    return mask


def unique_points_for_component(index_map: np.ndarray, pixels: list[tuple[int, int]]) -> np.ndarray:
    comp = component_mask(index_map.shape, pixels)
    ids = index_map[comp & (index_map >= 0)]
    ids = ids[(ids >= 0)]
    if ids.size == 0:
        return np.asarray([], dtype=np.int64)
    return np.unique(ids.astype(np.int64))


def candidate_score(comp: dict[str, Any], h: int, point_count: int, num_points: int, largest_area: int) -> float:
    _, y1, _, y2 = comp["bbox"]
    center_y = float(comp["center"][1])
    upper_score = 1.0 - center_y / max(1.0, float(h - 1))
    small_score = 1.0 - min(1.0, point_count / max(1.0, num_points * 0.20))
    detached_bonus = 0.35 if int(comp["area"]) < largest_area else -0.75
    vertical_thin_bonus = 0.15 if (y2 - y1) < h * 0.22 else 0.0
    return 2.0 * upper_score + 0.5 * small_score + detached_bonus + vertical_thin_bonus


def select_view_components(
    index_map: np.ndarray,
    view: str,
    num_points: int,
    dilation_radius: int,
    upper_ratio: float,
    strict_upper_ratio: float,
    max_component_fraction: float,
    min_points: int,
    max_components: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    sparse = index_map >= 0
    foreground = dilate_bool(sparse, dilation_radius)
    components = connected_components(foreground)
    if not components:
        return [], np.asarray([], dtype=np.int64)

    h, w = index_map.shape
    largest_area = max(int(comp["area"]) for comp in components)
    selected: list[dict[str, Any]] = []
    upper_band_ids: list[np.ndarray] = []

    for comp_idx, comp in enumerate(components):
        point_ids = unique_points_for_component(index_map, comp["pixels"])
        point_count = int(point_ids.size)
        if point_count:
            _, y1, _, y2 = comp["bbox"]
            if y1 <= int(h * upper_ratio):
                upper_band_ids.append(point_ids)

        if point_count < min_points:
            continue
        if point_count > max(1, int(num_points * max_component_fraction)):
            continue
        center_y = float(comp["center"][1])
        _, y1, _, y2 = comp["bbox"]
        if center_y > h * upper_ratio and y1 > h * strict_upper_ratio:
            continue
        score = candidate_score(comp, h, point_count, num_points, largest_area)
        selected.append(
            {
                "view": view,
                "component_index": comp_idx,
                "bbox": comp["bbox"],
                "center": comp["center"],
                "area": int(comp["area"]),
                "point_count": point_count,
                "score": float(score),
                "strict_upper": bool(center_y <= h * strict_upper_ratio),
                "point_ids": point_ids,
            }
        )

    selected.sort(key=lambda item: item["score"], reverse=True)
    selected = selected[:max_components]
    if upper_band_ids:
        upper_ids = np.unique(np.concatenate(upper_band_ids).astype(np.int64))
    else:
        upper_ids = np.asarray([], dtype=np.int64)
    return selected, upper_ids


def mask_from_ids(num_points: int, ids: np.ndarray) -> np.ndarray:
    mask = np.zeros((num_points,), dtype=np.uint8)
    valid = ids[(ids >= 0) & (ids < num_points)].astype(np.int64)
    if valid.size:
        mask[np.unique(valid)] = 1
    return mask


def above_body_ids(index_map: np.ndarray, row_threshold_ratio: float, margin: int) -> tuple[np.ndarray, int | None]:
    """Find points above the first dense foreground row.

    In sparse point-cloud renders of bags, the handle often appears as a thin
    structure above the dense rectangular body. This heuristic captures such
    handle/loop points without requiring an existing hook mask.
    """
    valid = index_map >= 0
    if not np.any(valid):
        return np.asarray([], dtype=np.int64), None
    row_counts = valid.sum(axis=1)
    max_count = int(row_counts.max())
    threshold = max(6, int(round(max_count * float(row_threshold_ratio))))
    dense_rows = np.where(row_counts >= threshold)[0]
    if dense_rows.size == 0:
        return np.asarray([], dtype=np.int64), None
    body_top = int(dense_rows.min())
    cutoff = max(0, body_top - max(0, int(margin)))
    top_region = index_map[:cutoff, :]
    ids = top_region[top_region >= 0]
    if ids.size == 0:
        return np.asarray([], dtype=np.int64), body_top
    return np.unique(ids.astype(np.int64)), body_top


def ids_from_components(components: list[dict[str, Any]], strict_only: bool = False) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for comp in components:
        if strict_only and not bool(comp.get("strict_upper")):
            continue
        ids = comp.get("point_ids")
        if isinstance(ids, np.ndarray) and ids.size:
            chunks.append(ids.astype(np.int64))
    if not chunks:
        return np.asarray([], dtype=np.int64)
    return np.unique(np.concatenate(chunks))


def load_point_cloud_point_count(root: Path, row: dict[str, str], fallback: int) -> int:
    path_value = row.get("point_cloud_path", "")
    if not path_value:
        return fallback
    path = resolve_portable_path(root, path_value)
    if not path.exists():
        return fallback
    points = np.load(path)
    if points.ndim != 2:
        raise ValueError(f"Invalid point cloud shape {points.shape}: {path}")
    return int(points.shape[0])


def generate_for_row(root: Path, args: argparse.Namespace, row: dict[str, str]) -> dict[str, Any]:
    sample_id = row["sample_id"]
    pilot_id = row["pilot_id"]
    renders_root = resolve_path(root, args.renders_root)
    manifest_path = renders_root / sample_id / "view_manifest.json"
    manifest = read_json(manifest_path)
    num_points = load_point_cloud_point_count(root, row, int(manifest["num_points"]))

    selected_components: list[dict[str, Any]] = []
    above_body_chunks: list[np.ndarray] = []
    upper_band_chunks: list[np.ndarray] = []
    per_view: list[dict[str, Any]] = []

    for entry in manifest.get("views", []):
        view = entry["view"]
        index_path = resolve_portable_path(root, entry["point_index_path"], manifest_path.parent)
        if not index_path.exists():
            raise FileNotFoundError(f"Point-index map not found: {index_path}")
        index_map = np.load(index_path)
        above_ids, body_top_y = above_body_ids(index_map, args.body_row_threshold_ratio, args.body_top_margin)
        if above_ids.size:
            above_body_chunks.append(above_ids)
        components, component_upper_ids = select_view_components(
            index_map=index_map,
            view=view,
            num_points=num_points,
            dilation_radius=args.dilation_radius,
            upper_ratio=args.upper_ratio,
            strict_upper_ratio=args.strict_upper_ratio,
            max_component_fraction=args.max_component_fraction,
            min_points=args.min_points,
            max_components=args.max_components_per_view,
        )
        selected_components.extend(components)
        top_limit = max(1, int(index_map.shape[0] * float(args.top_band_ratio)))
        top_band_pixels = index_map[:top_limit, :]
        top_band_ids = np.unique(top_band_pixels[top_band_pixels >= 0].astype(np.int64))
        if top_band_ids.size:
            upper_band_chunks.append(top_band_ids)
        per_view.append(
            {
                "view": view,
                "selected_components": [
                    {
                        key: value
                        for key, value in comp.items()
                        if key not in {"point_ids"}
                    }
                    for comp in components
                ],
                "component_upper_points": int(component_upper_ids.size),
                "top_band_points": int(top_band_ids.size),
                "above_body_points": int(above_ids.size),
                "estimated_body_top_y": body_top_y,
            }
        )

    loose_ids = ids_from_components(selected_components, strict_only=False)
    above_body_union_ids = (
        np.unique(np.concatenate(above_body_chunks).astype(np.int64)) if above_body_chunks else np.asarray([], dtype=np.int64)
    )
    upper_ids = np.unique(np.concatenate(upper_band_chunks).astype(np.int64)) if upper_band_chunks else np.asarray([], dtype=np.int64)

    candidate_ids = np.asarray(["A", "B", "C"], dtype=object)
    candidate_names = np.asarray(
        [
            "above_main_body_structure",
            "loose_upper_components",
            "upper_band_fallback",
        ],
        dtype=object,
    )
    candidate_descriptions = np.asarray(
        [
            "Points above the dense main object body; intended for bag handles, loops, or rings.",
            "Looser upper components that may include handle attachments.",
            "Fallback upper object band; broad and should be rejected unless it is truly hookable.",
        ],
        dtype=object,
    )
    candidate_masks = np.stack(
        [
            mask_from_ids(num_points, above_body_union_ids),
            mask_from_ids(num_points, loose_ids),
            mask_from_ids(num_points, upper_ids),
        ],
        axis=0,
    )

    output_dir = resolve_path(root, args.output_root) / pilot_id
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "candidates.npz"
    if npz_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {npz_path}")
    np.savez_compressed(
        npz_path,
        pilot_id=pilot_id,
        sample_id=sample_id,
        executor=row.get("executor", args.executor),
        candidate_ids=candidate_ids,
        candidate_names=candidate_names,
        candidate_descriptions=candidate_descriptions,
        candidate_masks=candidate_masks,
    )

    manifest_out = {
        "pilot_id": pilot_id,
        "sample_id": sample_id,
        "object_category": row.get("object_category", ""),
        "task": row.get("task", ""),
        "executor": row.get("executor", args.executor),
        "source": "geometry_hook_candidates_from_multiview_point_index",
        "candidate_npz": relative_to_dataset(root, npz_path),
        "candidate_ids": candidate_ids.tolist(),
        "candidate_names": candidate_names.tolist(),
        "candidate_descriptions": candidate_descriptions.tolist(),
        "candidate_point_counts": {
            str(candidate_ids[i]): int(candidate_masks[i].sum()) for i in range(len(candidate_ids))
        },
        "parameters": {
            "dilation_radius": int(args.dilation_radius),
            "upper_ratio": float(args.upper_ratio),
            "strict_upper_ratio": float(args.strict_upper_ratio),
            "max_component_fraction": float(args.max_component_fraction),
            "min_points": int(args.min_points),
            "max_components_per_view": int(args.max_components_per_view),
            "top_band_ratio": float(args.top_band_ratio),
            "body_row_threshold_ratio": float(args.body_row_threshold_ratio),
            "body_top_margin": int(args.body_top_margin),
        },
        "per_view": per_view,
        "notes": (
            "These masks are geometry proposals only. They are not ground truth and must be "
            "selected by VLM/rules and verified by a human reviewer."
        ),
    }
    manifest_json = output_dir / "candidate_manifest.json"
    write_json(manifest_json, manifest_out, args.overwrite)
    return manifest_out


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    rows = read_csv(resolve_path(root, args.pilot_csv))
    rows = [row for row in rows if row.get("executor") == args.executor]
    if args.pilot_id:
        rows = [row for row in rows if row.get("pilot_id") == args.pilot_id]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No hook pilot rows selected.")

    generated = [generate_for_row(root, args, row) for row in rows]
    summary = {
        "generated_rows": len(generated),
        "rows": [
            {
                "pilot_id": item["pilot_id"],
                "sample_id": item["sample_id"],
                "candidate_point_counts": item["candidate_point_counts"],
                "candidate_npz": item["candidate_npz"],
            }
            for item in generated
        ],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
