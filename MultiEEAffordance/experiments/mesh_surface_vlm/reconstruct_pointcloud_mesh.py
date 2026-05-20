#!/usr/bin/env python3
"""Reconstruct mesh candidates from a processed point cloud.

This is an isolated experiment for VLM-friendly rendering. The reconstructed
mesh must not be treated as ground truth geometry for affordance labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct point-cloud mesh for VLM visual experiments.")
    parser.add_argument("--dataset-root", default=".", help="Dataset root directory.")
    parser.add_argument("--points", default="", help="Input points.npy path relative to dataset root or absolute.")
    parser.add_argument("--pilot-id", default="", help="Optional pilot id used to resolve point path from metadata.")
    parser.add_argument(
        "--pilot-csv",
        default="processed/metadata/vlm_pilot_samples_v0_1.csv",
        help="Pilot CSV relative to dataset root.",
    )
    parser.add_argument(
        "--samples",
        default="processed/metadata/samples_checked_v0_1.jsonl",
        help="Samples JSONL relative to dataset root.",
    )
    parser.add_argument(
        "--candidate-root",
        default="processed/vlm_candidate_v2/3d_candidates",
        help="Candidate root relative to dataset root. Used by --preserve-candidates.",
    )
    parser.add_argument(
        "--preserve-candidates",
        default="",
        help="Comma-separated candidate ids to exclude from mesh reconstruction and save as preserved point components.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory relative to dataset root or absolute.")
    parser.add_argument(
        "--method",
        default="all",
        choices=["all", "poisson", "ball_pivoting", "alpha_shape"],
        help="Mesh reconstruction method.",
    )
    parser.add_argument("--normal-radius-ratio", type=float, default=0.05, help="Normal radius as bbox diagonal ratio.")
    parser.add_argument("--normal-max-nn", type=int, default=30, help="Max neighbors for normal estimation.")
    parser.add_argument("--orient-k", type=int, default=24, help="Neighbors for normal orientation.")
    parser.add_argument("--poisson-depth", type=int, default=8, help="Poisson reconstruction depth.")
    parser.add_argument("--poisson-density-quantile", type=float, default=0.03, help="Remove low-density Poisson vertices.")
    parser.add_argument("--alpha-ratio", type=float, default=0.08, help="Alpha shape alpha as bbox diagonal ratio.")
    parser.add_argument(
        "--bpa-radii-ratios",
        default="0.012,0.024,0.048",
        help="Comma-separated Ball Pivoting radii as bbox diagonal ratios.",
    )
    parser.add_argument("--sample-points", type=int, default=0, help="Optional number of mesh surface points to sample.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite outputs.")
    return parser.parse_args()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def write_json(path: Path, data: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def relative_to_root(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def load_points(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    if not path.exists():
        raise FileNotFoundError(f"Point cloud not found: {path}")
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] not in (3, 6):
        raise ValueError(f"Expected [N,3] or [N,6], got {arr.shape}: {path}")
    xyz = arr[:, :3].astype(np.float64)
    normals = arr[:, 3:6].astype(np.float64) if arr.shape[1] >= 6 else None
    if not np.isfinite(xyz).all():
        raise ValueError(f"Point cloud contains NaN/Inf: {path}")
    return xyz, normals


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
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


def candidate_point_path(row: dict[str, Any]) -> str:
    for key in ("point_cloud_path", "points_path", "point_path", "pointcloud_path"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def resolve_points_path(root: Path, args: argparse.Namespace) -> Path:
    if args.points:
        return resolve_path(root, args.points)
    if not args.pilot_id:
        raise ValueError("Provide either --points or --pilot-id.")
    pilot_rows = read_csv(resolve_path(root, args.pilot_csv))
    pilot_row = next((row for row in pilot_rows if row.get("pilot_id") == args.pilot_id), None)
    if pilot_row is None:
        raise ValueError(f"Pilot id not found in {args.pilot_csv}: {args.pilot_id}")
    direct = candidate_point_path(pilot_row)
    if direct:
        return resolve_path(root, direct)
    sample_id = pilot_row.get("sample_id", "")
    sample_rows = read_jsonl(resolve_path(root, args.samples))
    sample_row = next((row for row in sample_rows if str(row.get("sample_id", "")) == sample_id), None)
    if sample_row is None:
        raise ValueError(f"Sample id not found in {args.samples}: {sample_id}")
    sample_points = candidate_point_path(sample_row)
    if not sample_points:
        raise ValueError(f"No point path field found for sample: {sample_id}")
    return resolve_path(root, sample_points)


def bbox_diag(xyz: np.ndarray) -> float:
    diag = float(np.linalg.norm(np.ptp(xyz, axis=0)))
    if diag <= 0:
        raise ValueError("Point cloud has zero spatial extent.")
    return diag


def parse_ratios(value: str) -> list[float]:
    ratios: list[float] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        ratios.append(float(item))
    if not ratios:
        raise ValueError("At least one ratio is required.")
    return ratios


def parse_id_list(value: str) -> list[str]:
    out: list[str] = []
    for item in str(value or "").split(","):
        cid = item.strip().upper()
        if cid and cid not in out:
            out.append(cid)
    return out


def load_preserve_mask(root: Path, args: argparse.Namespace, n: int) -> tuple[np.ndarray, list[str], Path | None]:
    ids = parse_id_list(args.preserve_candidates)
    preserve_mask = np.zeros((n,), dtype=bool)
    if not ids:
        return preserve_mask, [], None
    if not args.pilot_id:
        raise ValueError("--preserve-candidates requires --pilot-id so the candidate manifest can be resolved.")
    manifest_path = resolve_path(root, args.candidate_root) / args.pilot_id / "candidate_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Candidate manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate_npz_path = resolve_path(root, manifest["candidate_npz"])
    if not candidate_npz_path.exists():
        candidate_npz_path = manifest_path.parent / Path(manifest["candidate_npz"]).name
    if not candidate_npz_path.exists():
        raise FileNotFoundError(f"Candidate NPZ not found: {candidate_npz_path}")
    data = np.load(candidate_npz_path, allow_pickle=True)
    candidate_ids = [str(item).upper() for item in data["candidate_ids"].tolist()]
    candidate_masks = data["candidate_masks"].astype(bool)
    missing = [cid for cid in ids if cid not in candidate_ids]
    if missing:
        raise ValueError(f"Preserve candidates not found in manifest: {missing}")
    if candidate_masks.ndim != 2 or candidate_masks.shape[1] != n:
        raise ValueError(f"Candidate mask shape {candidate_masks.shape} does not match point count {n}")
    for cid in ids:
        preserve_mask |= candidate_masks[candidate_ids.index(cid)]
    return preserve_mask, ids, manifest_path


def save_component_arrays(
    root: Path,
    output_dir: Path,
    xyz: np.ndarray,
    preserve_mask: np.ndarray,
    preserve_ids: list[str],
    overwrite: bool,
) -> dict[str, Any]:
    if not preserve_ids:
        return {}
    preserved_points_path = output_dir / "preserved_candidate_points.npy"
    preserved_mask_path = output_dir / "preserved_candidate_mask.npy"
    body_points_path = output_dir / "body_points_for_mesh.npy"
    for path in (preserved_points_path, preserved_mask_path, body_points_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output exists. Use --overwrite: {path}")
    np.save(preserved_points_path, xyz[preserve_mask].astype(np.float32))
    np.save(preserved_mask_path, preserve_mask.astype(np.uint8))
    np.save(body_points_path, xyz[~preserve_mask].astype(np.float32))
    return {
        "preserve_candidates": preserve_ids,
        "preserved_point_count": int(preserve_mask.sum()),
        "body_point_count": int((~preserve_mask).sum()),
        "preserved_points_path": relative_to_root(root, preserved_points_path),
        "preserved_mask_path": relative_to_root(root, preserved_mask_path),
        "body_points_path": relative_to_root(root, body_points_path),
        "notes": "Preserved candidates are not meshed; render them as original points over the body mesh.",
    }


def make_point_cloud(xyz: np.ndarray, normals: np.ndarray | None, args: argparse.Namespace):
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ValueError("open3d is required for mesh reconstruction. Install open3d in this environment.") from exc

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    diag = bbox_diag(xyz)
    if normals is not None and normals.shape == xyz.shape and np.isfinite(normals).all():
        pcd.normals = o3d.utility.Vector3dVector(normals)
    else:
        radius = max(1e-8, float(args.normal_radius_ratio) * diag)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max(3, int(args.normal_max_nn)))
        )
    try:
        pcd.orient_normals_consistent_tangent_plane(max(3, int(args.orient_k)))
    except RuntimeError:
        # Sparse or disconnected point clouds can fail orientation. The mesh
        # is still useful as an experiment, so continue with estimated normals.
        pass
    return pcd, o3d


def clean_mesh(mesh: Any) -> Any:
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def write_mesh(o3d: Any, mesh: Any, path: Path, overwrite: bool) -> dict[str, Any]:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = clean_mesh(mesh)
    ok = o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False)
    if not ok:
        raise ValueError(f"Failed to write mesh: {path}")
    return {
        "path": str(path),
        "vertices": int(np.asarray(mesh.vertices).shape[0]),
        "triangles": int(np.asarray(mesh.triangles).shape[0]),
    }


def reconstruct_poisson(pcd: Any, o3d: Any, args: argparse.Namespace) -> Any:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=int(args.poisson_depth))
    densities_np = np.asarray(densities)
    if densities_np.size:
        q = min(0.95, max(0.0, float(args.poisson_density_quantile)))
        threshold = float(np.quantile(densities_np, q))
        mesh.remove_vertices_by_mask(densities_np < threshold)
    return clean_mesh(mesh)


def reconstruct_ball_pivoting(pcd: Any, o3d: Any, diag: float, args: argparse.Namespace) -> Any:
    radii = [max(1e-8, ratio * diag) for ratio in parse_ratios(args.bpa_radii_ratios)]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
    return clean_mesh(mesh)


def reconstruct_alpha_shape(pcd: Any, o3d: Any, diag: float, args: argparse.Namespace) -> Any:
    alpha = max(1e-8, float(args.alpha_ratio) * diag)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    return clean_mesh(mesh)


def sample_mesh_points(o3d: Any, mesh: Any, path: Path, n: int, overwrite: bool) -> dict[str, Any] | None:
    if n <= 0:
        return None
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {path}")
    if len(mesh.triangles) == 0:
        return None
    pcd = mesh.sample_points_poisson_disk(number_of_points=int(n), init_factor=5)
    arr = np.asarray(pcd.points, dtype=np.float32)
    np.save(path, arr)
    return {"path": str(path), "shape": list(arr.shape)}


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    points_path = resolve_points_path(root, args)
    output_dir = resolve_path(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    xyz, normals = load_points(points_path)
    preserve_mask, preserve_ids, candidate_manifest_path = load_preserve_mask(root, args, xyz.shape[0])
    component_summary = save_component_arrays(root, output_dir, xyz, preserve_mask, preserve_ids, args.overwrite)
    mesh_xyz = xyz[~preserve_mask] if preserve_ids else xyz
    mesh_normals = normals[~preserve_mask] if normals is not None and preserve_ids else normals
    if mesh_xyz.shape[0] < 32:
        raise ValueError(f"Too few points left for mesh reconstruction after preserving candidates: {mesh_xyz.shape[0]}")
    diag = bbox_diag(mesh_xyz)
    pcd, o3d = make_point_cloud(mesh_xyz, mesh_normals, args)

    methods = ["poisson", "ball_pivoting", "alpha_shape"] if args.method == "all" else [args.method]
    outputs: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for method in methods:
        try:
            if method == "poisson":
                mesh = reconstruct_poisson(pcd, o3d, args)
                mesh_path = output_dir / ("poisson_body_mesh.ply" if preserve_ids else "poisson_mesh.ply")
            elif method == "ball_pivoting":
                mesh = reconstruct_ball_pivoting(pcd, o3d, diag, args)
                mesh_path = output_dir / ("ball_pivoting_body_mesh.ply" if preserve_ids else "ball_pivoting_mesh.ply")
            elif method == "alpha_shape":
                mesh = reconstruct_alpha_shape(pcd, o3d, diag, args)
                mesh_path = output_dir / ("alpha_shape_body_mesh.ply" if preserve_ids else "alpha_shape_mesh.ply")
            else:
                raise ValueError(f"Unsupported method: {method}")
            record = write_mesh(o3d, mesh, mesh_path, args.overwrite)
            record["method"] = method
            sample = sample_mesh_points(
                o3d,
                mesh,
                output_dir / f"{method}_sampled_points.npy",
                int(args.sample_points),
                args.overwrite,
            )
            if sample:
                record["sampled_points"] = sample
            outputs.append(record)
        except Exception as exc:  # keep all-method experiments running
            errors.append({"method": method, "error": str(exc)})

    summary = {
        "experiment": "mesh_surface_vlm",
        "points_path": relative_to_root(root, points_path),
        "num_points": int(xyz.shape[0]),
        "mesh_input_points": int(mesh_xyz.shape[0]),
        "component_aware": bool(preserve_ids),
        "preserved_components": component_summary,
        "candidate_manifest": relative_to_root(root, candidate_manifest_path) if candidate_manifest_path else None,
        "bbox_diag": float(diag),
        "methods_requested": methods,
        "outputs": [
            {
                **item,
                "path": relative_to_root(root, Path(item["path"])),
                **(
                    {
                        "sampled_points": {
                            **item["sampled_points"],
                            "path": relative_to_root(root, Path(item["sampled_points"]["path"])),
                        }
                    }
                    if "sampled_points" in item
                    else {}
                ),
            }
            for item in outputs
        ],
        "errors": errors,
        "notes": "Meshes are visual aids for VLM experiments, not ground-truth affordance geometry.",
    }
    write_json(output_dir / "mesh_reconstruction_summary.json", summary, args.overwrite)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
