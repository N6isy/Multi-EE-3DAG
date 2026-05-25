#!/usr/bin/env python3
"""Generate v3 3D part candidates without asking the VLM for coordinates.

This stage is the part-segmentation candidate backbone for the v3 path:

  original point cloud + optional weak masks / part-segmentation adapters
      -> high-recall 3D part candidates [K, N]
      -> VLM selects candidate IDs later

All candidates are binary masks over the original point cloud length N.
PartSLIP++ can be plugged in later through the reserved adapter backend.
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
    parser.add_argument(
        "--backend",
        choices=["geometry", "partslippp"],
        default="geometry",
        help="Candidate backend. geometry is the local fallback; partslippp is a reserved external adapter.",
    )
    parser.add_argument("--pilot-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--k-neighbors", type=int, default=24)
    parser.add_argument("--min-points", type=int, default=4)
    parser.add_argument("--max-candidate-fraction", type=float, default=0.70)
    parser.add_argument("--max-candidates", type=int, default=18)
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


def empty_candidates(n: int) -> tuple[np.ndarray, list[dict[str, Any]], str]:
    return np.zeros((0, n), dtype=np.uint8), [], "no_candidate_survived"


def generate_with_backend(
    root: Path,
    args: argparse.Namespace,
    row: dict[str, str],
    sample_by_id: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]], str, Path, Path | None]:
    if args.backend == "partslippp":
        raise NotImplementedError(
            "Backend 'partslippp' is reserved for the external PartSLIP++ adapter. "
            "Use --backend geometry until the runtime integration is configured."
        )
    if args.backend != "geometry":
        raise ValueError(f"Unsupported backend: {args.backend}")

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
    try:
        candidate_masks, candidates = gen.generate_candidates(xyz, normals, weak_mask, enriched, args)
        return candidate_masks, [normalize_meta(item) for item in candidates], "", point_path, mask_path
    except Exception as exc:
        if not args.allow_empty_candidates:
            raise
        candidate_masks, candidates, status = empty_candidates(xyz.shape[0])
        return candidate_masks, candidates, f"{status}: {type(exc).__name__}: {exc}", point_path, mask_path


def write_candidate_outputs(
    root: Path,
    args: argparse.Namespace,
    row: dict[str, str],
    candidate_masks: np.ndarray,
    candidates: list[dict[str, Any]],
    generation_warning: str,
    point_path: Path,
    mask_path: Path | None,
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
            "k_neighbors": int(args.k_neighbors),
            "min_points": int(args.min_points),
            "max_candidate_fraction": float(args.max_candidate_fraction),
            "max_candidates": int(args.max_candidates),
            "seed_expand_hops": int(args.seed_expand_hops),
            "component_max_candidates": int(args.component_max_candidates),
            "view_dilation_radius": int(args.view_dilation_radius),
            "body_row_threshold_ratio": float(args.body_row_threshold_ratio),
            "body_top_margin": int(args.body_top_margin),
        },
        "notes": (
            "Part candidates are high-recall proposals over original point indices only. "
            "The default geometry backend is a fallback; PartSLIP++ can replace it once configured."
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
    candidate_masks, candidates, warning, point_path, mask_path = generate_with_backend(root, args, row, sample_by_id)
    return write_candidate_outputs(root, args, row, candidate_masks, candidates, warning, point_path, mask_path)


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
