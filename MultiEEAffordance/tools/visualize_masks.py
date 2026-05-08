#!/usr/bin/env python3
"""Visualize raw point clouds and Multi-EE affordance mask channels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


EXECUTOR_ORDER = ["gripper", "suction", "hook", "dexterous_hand"]
CHANNELS = ["raw"] + EXECUTOR_ORDER + ["all"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize point clouds and [N,4] Multi-EE masks.")
    parser.add_argument("--points", required=True, help="Input point cloud .npy with shape [N,3] or [N,6].")
    parser.add_argument("--masks", help="Input mask .npy with shape [N,4]. Required unless --channel raw.")
    parser.add_argument("--channel", default="all", choices=CHANNELS, help="Channel to visualize.")
    parser.add_argument("--backend", default="matplotlib", choices=["matplotlib", "open3d"], help="Visualization backend.")
    parser.add_argument("--output", help="Optional PNG output path for matplotlib backend.")
    parser.add_argument("--max-points", type=int, default=50000, help="Randomly downsample to this many points for display.")
    parser.add_argument("--point-size", type=float, default=2.0, help="Point size for matplotlib scatter.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used for visualization downsampling.")
    return parser.parse_args()


def error(message: str) -> None:
    raise ValueError(message)


def load_points(path: Path) -> np.ndarray:
    if not path.exists():
        error(f"Point cloud file does not exist: {path}")
    points = np.load(path)
    if points.ndim != 2 or points.shape[1] not in (3, 6):
        error(f"Point cloud must have shape [N,3] or [N,6], got {points.shape} from {path}")
    if points.shape[0] == 0:
        error(f"Point cloud is empty: {path}")
    return points


def load_masks(path: Path, n_points: int) -> np.ndarray:
    if not path.exists():
        error(f"Mask file does not exist: {path}")
    masks = np.load(path)
    if masks.ndim != 2 or masks.shape != (n_points, len(EXECUTOR_ORDER)):
        error(f"Mask must have shape [{n_points}, 4], got {masks.shape} from {path}")
    return masks > 0


def sample_indices(n_points: int, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0:
        error("--max-points must be positive")
    if n_points <= max_points:
        return np.arange(n_points)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_points, size=max_points, replace=False))


def set_equal_axes(ax: object, xyz: np.ndarray) -> None:
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    ranges = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()])
    max_range = float(ranges.max())
    if max_range <= 0:
        max_range = 1.0
    centers = np.array([(x.max() + x.min()) / 2, (y.max() + y.min()) / 2, (z.max() + z.min()) / 2])
    ax.set_xlim(centers[0] - max_range / 2, centers[0] + max_range / 2)
    ax.set_ylim(centers[1] - max_range / 2, centers[1] + max_range / 2)
    ax.set_zlim(centers[2] - max_range / 2, centers[2] + max_range / 2)
    ax.set_box_aspect([1, 1, 1])


def matplotlib_visualize(points: np.ndarray, masks: np.ndarray | None, channel: str, output: str | None, point_size: float) -> None:
    try:
        if output:
            import matplotlib

            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ValueError("matplotlib is not installed. Use --backend open3d or install matplotlib.") from exc

    xyz = points[:, :3]
    channels = ["raw"] + EXECUTOR_ORDER if channel == "all" else [channel]
    fig = plt.figure(figsize=(4.2 * len(channels), 4.2))

    for index, name in enumerate(channels, start=1):
        ax = fig.add_subplot(1, len(channels), index, projection="3d")
        if name == "raw":
            colors = np.full((xyz.shape[0], 3), [0.50, 0.53, 0.58])
            title = "raw"
        else:
            assert masks is not None
            positive = masks[:, EXECUTOR_ORDER.index(name)]
            colors = np.full((xyz.shape[0], 3), [0.78, 0.80, 0.84])
            colors[positive] = [0.90, 0.12, 0.12]
            title = f"{name} ({int(positive.sum())})"
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=point_size, depthshade=False)
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_axis_off()
        set_equal_axes(ax, xyz)

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.88, wspace=0.03)
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180)
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()


def open3d_visualize(points: np.ndarray, masks: np.ndarray | None, channel: str) -> None:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ValueError("open3d is not installed. Use --backend matplotlib or install open3d.") from exc

    xyz = points[:, :3]
    channels = ["raw"] + EXECUTOR_ORDER if channel == "all" else [channel]
    for name in channels:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        if name == "raw":
            colors = np.full((xyz.shape[0], 3), [0.50, 0.53, 0.58])
        else:
            assert masks is not None
            positive = masks[:, EXECUTOR_ORDER.index(name)]
            colors = np.full((xyz.shape[0], 3), [0.78, 0.80, 0.84])
            colors[positive] = [0.90, 0.12, 0.12]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd], window_name=f"Multi-EE mask: {name}")


def main() -> int:
    args = parse_args()
    try:
        points = load_points(Path(args.points))
        masks = None
        if args.channel != "raw":
            if not args.masks:
                error("--masks is required unless --channel raw")
            masks = load_masks(Path(args.masks), points.shape[0])

        idx = sample_indices(points.shape[0], args.max_points, args.seed)
        points = points[idx]
        if masks is not None:
            masks = masks[idx]

        if args.backend == "matplotlib":
            matplotlib_visualize(points, masks, args.channel, args.output, args.point_size)
        else:
            if args.output:
                print("WARNING: --output is ignored by open3d backend.", file=sys.stderr)
            open3d_visualize(points, masks, args.channel)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
