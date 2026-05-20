#!/usr/bin/env python3
"""Render reconstructed meshes into VLM-friendly multi-view PNGs."""

from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render mesh views for VLM visual experiments.")
    parser.add_argument("--dataset-root", default=".", help="Dataset root directory.")
    parser.add_argument("--mesh", required=True, help="Input mesh path relative to dataset root or absolute.")
    parser.add_argument("--output-dir", required=True, help="Output directory relative to dataset root or absolute.")
    parser.add_argument("--views", default=",".join(DEFAULT_VIEWS), help="Comma-separated yaw/elev view names.")
    parser.add_argument("--image-size", type=int, default=768, help="Square output image size.")
    parser.add_argument("--face-color", default="#d9e1ea", help="Mesh face color.")
    parser.add_argument("--edge-color", default="#5a6575", help="Mesh edge color.")
    parser.add_argument("--edge-alpha", type=float, default=0.08, help="Mesh edge alpha, 0 disables edges.")
    parser.add_argument("--background", default="#0e1117", help="Background color.")
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


def set_equal_axes(ax: Any, vertices: np.ndarray) -> None:
    center = vertices.mean(axis=0)
    radius = float(np.max(np.linalg.norm(vertices - center, axis=1)))
    radius = max(radius, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def render_view(
    vertices: np.ndarray,
    triangles: np.ndarray,
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
        alpha=1.0,
    )
    if float(args.edge_alpha) > 0:
        collection.set_edgecolor(args.edge_color)
        collection.set_alpha(1.0)
    ax.add_collection3d(collection)
    set_equal_axes(ax, vertices)
    ax.view_init(elev=float(elev), azim=float(yaw))
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
    views = parse_views(args.views)

    manifest_views: list[dict[str, Any]] = []
    for name, yaw, elev in views:
        output_path = output_dir / f"{name}_mesh.png"
        render_view(vertices, triangles, output_path, yaw, elev, args)
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
