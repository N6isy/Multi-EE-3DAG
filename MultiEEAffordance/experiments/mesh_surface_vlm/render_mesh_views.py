#!/usr/bin/env python3
"""Render reconstructed meshes into VLM-friendly multi-view PNGs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_VIEWS = [
    "yaw000_elev20",
    "yaw045_elev20",
    "yaw090_elev20",
    "yaw135_elev20",
    "yaw180_elev20",
    "yaw225_elev20",
    "yaw270_elev20",
    "yaw315_elev20",
]

CANDIDATE_COLORS = [
    "#ff4848",
    "#32dc78",
    "#ffd240",
    "#78a0ff",
    "#ff70d2",
    "#46e6eb",
    "#ff9150",
    "#be7dff",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render mesh views for VLM visual experiments.")
    parser.add_argument("--dataset-root", default=".", help="Dataset root directory.")
    parser.add_argument("--mesh", required=True, help="Input mesh path relative to dataset root or absolute.")
    parser.add_argument("--output-dir", required=True, help="Output directory relative to dataset root or absolute.")
    parser.add_argument("--points", default="", help="Optional original points.npy path for overlay.")
    parser.add_argument("--pilot-id", default="", help="Optional pilot id used to resolve points and candidates.")
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
        help="Candidate root relative to dataset root.",
    )
    parser.add_argument(
        "--overlay-candidates",
        default="",
        help="Comma-separated candidate ids to overlay as highlighted original points.",
    )
    parser.add_argument("--overlay-points", action="store_true", help="Overlay all original points as context.")
    parser.add_argument("--views", default=",".join(DEFAULT_VIEWS), help="Comma-separated yaw/elev view names.")
    parser.add_argument("--image-size", type=int, default=768, help="Square output image size.")
    parser.add_argument("--face-color", default="#d9e1ea", help="Mesh face color.")
    parser.add_argument("--mesh-alpha", type=float, default=0.68, help="Mesh face alpha. Lower values reveal point overlays.")
    parser.add_argument("--edge-color", default="#5a6575", help="Mesh edge color.")
    parser.add_argument("--edge-alpha", type=float, default=0.08, help="Mesh edge alpha, 0 disables edges.")
    parser.add_argument("--background", default="#0e1117", help="Background color.")
    parser.add_argument("--point-color", default="#2f80ed", help="Original point overlay color.")
    parser.add_argument("--point-alpha", type=float, default=0.32, help="Original point overlay alpha.")
    parser.add_argument("--point-size", type=float, default=3.0, help="Original point scatter size.")
    parser.add_argument("--candidate-point-size", type=float, default=26.0, help="Candidate point scatter size.")
    parser.add_argument("--dpi", type=int, default=120, help="Matplotlib render DPI.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite outputs.")
    return parser.parse_args()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def relative_to_root(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def write_json(path: Path, data: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


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


def resolve_points_path(root: Path, args: argparse.Namespace) -> Path | None:
    if args.points:
        return resolve_path(root, args.points)
    if not args.pilot_id:
        return None
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


def load_points(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Point cloud not found: {path}")
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] not in (3, 6):
        raise ValueError(f"Expected [N,3] or [N,6], got {arr.shape}: {path}")
    return arr[:, :3].astype(np.float64)


def parse_id_list(value: str) -> list[str]:
    out: list[str] = []
    for item in str(value or "").split(","):
        cid = item.strip().upper()
        if cid and cid not in out:
            out.append(cid)
    return out


def load_candidate_points(root: Path, args: argparse.Namespace, xyz: np.ndarray | None) -> tuple[dict[str, np.ndarray], Path | None]:
    ids = parse_id_list(args.overlay_candidates)
    if not ids:
        return {}, None
    if xyz is None:
        raise ValueError("--overlay-candidates requires --points or --pilot-id so original points can be loaded.")
    if not args.pilot_id:
        raise ValueError("--overlay-candidates requires --pilot-id so candidate masks can be resolved.")
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
    if candidate_masks.shape[1] != xyz.shape[0]:
        raise ValueError(f"Candidate masks length {candidate_masks.shape[1]} does not match points {xyz.shape[0]}")
    missing = [cid for cid in ids if cid not in candidate_ids]
    if missing:
        raise ValueError(f"Overlay candidates not found in manifest: {missing}")
    out: dict[str, np.ndarray] = {}
    for cid in ids:
        out[cid] = xyz[candidate_masks[candidate_ids.index(cid)]]
    return out, manifest_path


def parse_views(value: str) -> list[tuple[str, float, float]]:
    views: list[tuple[str, float, float]] = []
    for raw in str(value).split(","):
        name = raw.strip()
        if not name:
            continue
        yaw = 0.0
        elev = 20.0
        for part in name.split("_"):
            if part.startswith("yaw"):
                yaw = float(part[3:])
            elif part.startswith("elev"):
                elev = float(part[4:])
        views.append((name, yaw, elev))
    if not views:
        raise ValueError("No views requested.")
    return views


def load_mesh(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ValueError("open3d is required to read mesh files for this experiment.") from exc
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
        raise ValueError(f"Mesh has no vertices: {mesh_path}")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.shape[0] == 0:
        raise ValueError(f"Mesh has no triangles: {mesh_path}")
    return vertices, triangles


def set_equal_axes(ax: Any, vertices: np.ndarray, extra_points: list[np.ndarray] | None = None) -> None:
    arrays = [vertices]
    if extra_points:
        arrays.extend([pts for pts in extra_points if pts is not None and pts.size])
    all_points = np.concatenate(arrays, axis=0)
    center = all_points.mean(axis=0)
    radius = float(np.max(np.linalg.norm(all_points - center, axis=1)))
    radius = max(radius, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def render_view(
    vertices: np.ndarray,
    triangles: np.ndarray,
    original_points: np.ndarray | None,
    candidate_points: dict[str, np.ndarray],
    output_path: Path,
    yaw: float,
    elev: float,
    args: argparse.Namespace,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    size = max(128, int(args.image_size))
    fig = plt.figure(figsize=(size / args.dpi, size / args.dpi), dpi=args.dpi)
    fig.patch.set_facecolor(args.background)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(args.background)
    tris = vertices[triangles]
    edge_color = args.edge_color if float(args.edge_alpha) > 0 else "none"
    collection = Poly3DCollection(
        tris,
        facecolors=args.face_color,
        edgecolors=edge_color,
        linewidths=0.18,
        alpha=max(0.05, min(1.0, float(args.mesh_alpha))),
    )
    if float(args.edge_alpha) > 0:
        collection.set_edgecolor(args.edge_color)
        collection.set_alpha(max(0.05, min(1.0, float(args.mesh_alpha))))
    ax.add_collection3d(collection)
    extra_for_axes: list[np.ndarray] = []
    if original_points is not None and (args.overlay_points or candidate_points):
        extra_for_axes.append(original_points)
    extra_for_axes.extend(candidate_points.values())
    set_equal_axes(ax, vertices, extra_for_axes)
    if original_points is not None and args.overlay_points:
        ax.scatter(
            original_points[:, 0],
            original_points[:, 1],
            original_points[:, 2],
            s=float(args.point_size),
            c=args.point_color,
            alpha=max(0.0, min(1.0, float(args.point_alpha))),
            depthshade=False,
        )
    for idx, (cid, pts) in enumerate(candidate_points.items()):
        if pts.size == 0:
            continue
        color = CANDIDATE_COLORS[idx % len(CANDIDATE_COLORS)]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            s=float(args.candidate_point_size),
            c=color,
            alpha=0.98,
            depthshade=False,
            label=cid,
        )
    ax.view_init(elev=float(elev), azim=float(yaw))
    if candidate_points:
        label = "candidates: " + ",".join(candidate_points.keys())
        ax.text2D(0.02, 0.96, label, transform=ax.transAxes, color="#f6f8fb", fontsize=8)
    ax.set_axis_off()
    ax.grid(False)
    try:
        ax.set_proj_type("persp")
    except Exception:
        pass
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(output_path, facecolor=args.background, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    mesh_path = resolve_path(root, args.mesh)
    output_dir = resolve_path(root, args.output_dir)
    vertices, triangles = load_mesh(mesh_path)
    points_path = resolve_points_path(root, args)
    original_points = load_points(points_path)
    candidate_points, candidate_manifest_path = load_candidate_points(root, args, original_points)
    views = parse_views(args.views)

    manifest_views: list[dict[str, Any]] = []
    for name, yaw, elev in views:
        output_path = output_dir / f"{name}_mesh.png"
        render_view(vertices, triangles, original_points, candidate_points, output_path, yaw, elev, args)
        manifest_views.append(
            {
                "view": name,
                "yaw": float(yaw),
                "elev": float(elev),
                "render_path": relative_to_root(root, output_path),
            }
        )

    manifest = {
        "experiment": "mesh_surface_vlm",
        "mesh_path": relative_to_root(root, mesh_path),
        "num_vertices": int(vertices.shape[0]),
        "num_triangles": int(triangles.shape[0]),
        "points_path": relative_to_root(root, points_path) if points_path else None,
        "candidate_manifest": relative_to_root(root, candidate_manifest_path) if candidate_manifest_path else None,
        "overlay_points": bool(args.overlay_points),
        "overlay_candidates": list(candidate_points.keys()),
        "views": manifest_views,
        "notes": "Mesh renders are visual aids for VLM inspection and should not be treated as ground-truth masks.",
    }
    write_json(output_dir / "mesh_render_manifest.json", manifest, args.overwrite)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
