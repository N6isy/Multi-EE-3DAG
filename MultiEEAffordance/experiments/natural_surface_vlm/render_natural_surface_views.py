#!/usr/bin/env python3
"""Render natural-like point-cloud surface views with 2D-to-3D index maps.

This experiment is intentionally isolated from the v2/v3 dataset pipeline.
It creates VLM-friendly images that look more like continuous object surfaces,
while preserving per-pixel links back to the original point indices.
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
from PIL import Image, ImageDraw, ImageFilter, ImageFont

DATASET_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = DATASET_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from path_utils import relative_to_dataset, resolve_portable_path  # noqa: E402
from render_multiview import DEFAULT_VIEWS, find_sample, normalize_points, read_jsonl, view_rotation  # noqa: E402


BACKGROUND = np.array([14, 16, 20], dtype=np.uint8)
MATERIAL = np.array([212, 220, 230], dtype=np.float32)
EDGE = np.array([58, 66, 78], dtype=np.uint8)
CONFIDENCE_LOW = np.array([42, 58, 92], dtype=np.float32)
CONFIDENCE_HIGH = np.array([220, 238, 255], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render natural-like VLM views with point-index maps for 2D-to-3D backprojection."
    )
    parser.add_argument("--dataset-root", default=".", help="Dataset root directory.")
    parser.add_argument(
        "--pilot-csv",
        default="processed/metadata/vlm_pilot_samples_v0_1.csv",
        help="Pilot CSV relative to dataset root.",
    )
    parser.add_argument(
        "--samples",
        default="processed/metadata/samples_checked_v0_1.jsonl",
        help="Checked samples JSONL relative to dataset root.",
    )
    parser.add_argument(
        "--output-root",
        default="processed/natural_surface_vlm/renders",
        help="Output root relative to dataset root.",
    )
    parser.add_argument("--sample-id", default="", help="Render a specific sample id.")
    parser.add_argument("--pilot-id", default="", help="Render sample(s) referenced by one pilot id.")
    parser.add_argument("--limit", type=int, default=None, help="Limit selected pilot rows before sample de-duplication.")
    parser.add_argument("--views", default=",".join(DEFAULT_VIEWS), help="Comma-separated view names.")
    parser.add_argument("--image-size", type=int, default=768, help="Square output image size.")
    parser.add_argument("--splat-radius", type=int, default=8, help="Visual splat radius for surface-like rendering.")
    parser.add_argument("--index-radius", type=int, default=1, help="Exact sparse point-index radius for diagnostics.")
    parser.add_argument("--fill-radius", type=int, default=5, help="Image-space nearest-point fill radius for small holes.")
    parser.add_argument("--padding", type=float, default=0.08, help="Projection padding ratio.")
    parser.add_argument("--smooth", action="store_true", help="Apply mild RGB smoothing to the natural render.")
    parser.add_argument("--no-panel", action="store_true", help="Do not write diagnostic panel images.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def resolve_path(root: Path, value: str | Path) -> Path:
    return resolve_portable_path(root, value)


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


def load_points_with_normals(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    if not path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {path}")
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] not in (3, 6):
        raise ValueError(f"Expected point cloud shape [N,3] or [N,6], got {arr.shape}: {path}")
    if arr.shape[0] == 0:
        raise ValueError(f"Point cloud is empty: {path}")
    xyz = arr[:, :3].astype(np.float32)
    normals = arr[:, 3:6].astype(np.float32) if arr.shape[1] == 6 else None
    return xyz, normals


def select_sample_ids(root: Path, args: argparse.Namespace) -> list[str]:
    if args.sample_id:
        return [args.sample_id]
    pilot_rows = read_csv(resolve_path(root, args.pilot_csv))
    if args.pilot_id:
        pilot_rows = [row for row in pilot_rows if row.get("pilot_id") == args.pilot_id]
    if args.limit is not None:
        pilot_rows = pilot_rows[: args.limit]
    if not pilot_rows:
        raise ValueError("No sample selected. Provide --sample-id or a valid --pilot-id.")
    selected: list[str] = []
    seen: set[str] = set()
    for row in pilot_rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if sample_id and sample_id not in seen:
            seen.add(sample_id)
            selected.append(sample_id)
    if not selected:
        raise ValueError("No sample_id found in selected pilot rows.")
    return selected


def parse_views(value: str) -> list[str]:
    views = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not views:
        raise ValueError("At least one view is required.")
    return views


def project_points(
    points_xyz: np.ndarray,
    normals: np.ndarray | None,
    view: str,
    image_size: int,
    padding: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    rotation = view_rotation(view)
    rotated = points_xyz @ rotation.T
    rotated_normals = None
    if normals is not None:
        rotated_normals = normals @ rotation.T
        denom = np.linalg.norm(rotated_normals, axis=1, keepdims=True)
        rotated_normals = rotated_normals / np.maximum(denom, 1e-8)

    xy = rotated[:, :2]
    xy_min = xy.min(axis=0)
    xy_max = xy.max(axis=0)
    span = np.maximum(xy_max - xy_min, 1e-6)
    xy_norm = (xy - xy_min) / span
    xy_norm = xy_norm * (1 - 2 * padding) + padding
    px = np.clip((xy_norm[:, 0] * (image_size - 1)).round().astype(np.int32), 0, image_size - 1)
    py = np.clip(((1.0 - xy_norm[:, 1]) * (image_size - 1)).round().astype(np.int32), 0, image_size - 1)
    return px, py, rotated[:, 2].astype(np.float32), rotated_normals


def splat_index_map(
    px: np.ndarray,
    py: np.ndarray,
    depth: np.ndarray,
    normals: np.ndarray | None,
    image_size: int,
    radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    index_map = np.full((image_size, image_size), -1, dtype=np.int64)
    depth_map = np.full((image_size, image_size), -np.inf, dtype=np.float32)
    dist_map = np.full((image_size, image_size), np.inf, dtype=np.float32)
    normal_map = np.zeros((image_size, image_size, 3), dtype=np.float32)
    radius = max(0, int(radius))
    radius2 = max(0, radius * radius)

    # Back-to-front splatting with a z-buffer. A larger radius creates a
    # continuous visual surface, but every pixel still stores a source point id.
    for idx in np.argsort(depth):
        x = int(px[idx])
        y = int(py[idx])
        z = float(depth[idx])
        y0, y1 = max(0, y - radius), min(image_size - 1, y + radius)
        x0, x1 = max(0, x - radius), min(image_size - 1, x + radius)
        for yy in range(y0, y1 + 1):
            dy = yy - y
            for xx in range(x0, x1 + 1):
                dx = xx - x
                d2 = dx * dx + dy * dy
                if radius > 0 and d2 > radius2:
                    continue
                if z > depth_map[yy, xx]:
                    index_map[yy, xx] = int(idx)
                    depth_map[yy, xx] = z
                    dist_map[yy, xx] = float(np.sqrt(d2)) / max(1, radius)
                    if normals is not None:
                        normal_map[yy, xx] = normals[idx]
    depth_map[index_map < 0] = np.nan
    return index_map, depth_map, dist_map, normal_map


def fill_small_holes(
    index_map: np.ndarray,
    depth_map: np.ndarray,
    dist_map: np.ndarray,
    normal_map: np.ndarray,
    fill_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_map = np.zeros(index_map.shape, dtype=np.uint8)
    source_map[index_map >= 0] = 1
    fill_dist = np.full(index_map.shape, 32767, dtype=np.int16)
    queue: deque[tuple[int, int]] = deque()
    ys, xs = np.where(index_map >= 0)
    for y, x in zip(ys.tolist(), xs.tolist(), strict=False):
        fill_dist[y, x] = 0
        queue.append((y, x))

    max_dist = max(0, int(fill_radius))
    h, w = index_map.shape
    while queue:
        y, x = queue.popleft()
        current = int(fill_dist[y, x])
        if current >= max_dist:
            continue
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            yy = y + dy
            xx = x + dx
            if yy < 0 or yy >= h or xx < 0 or xx >= w or index_map[yy, xx] >= 0:
                continue
            index_map[yy, xx] = index_map[y, x]
            depth_map[yy, xx] = depth_map[y, x]
            dist_map[yy, xx] = dist_map[y, x]
            normal_map[yy, xx] = normal_map[y, x]
            fill_dist[yy, xx] = current + 1
            source_map[yy, xx] = 2
            queue.append((yy, xx))

    confidence = np.zeros(index_map.shape, dtype=np.float32)
    direct = source_map == 1
    filled = source_map == 2
    confidence[direct] = np.clip(1.0 - 0.22 * np.nan_to_num(dist_map[direct], nan=0.0), 0.72, 1.0)
    if np.any(filled):
        confidence[filled] = np.clip(0.68 * (1.0 - fill_dist[filled].astype(np.float32) / (max_dist + 1.0)), 0.18, 0.68)
    return index_map, depth_map, dist_map, normal_map, source_map, confidence


def edge_mask(valid: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
    edge = np.zeros(valid.shape, dtype=bool)
    edge[:-1, :] |= valid[:-1, :] != valid[1:, :]
    edge[:, :-1] |= valid[:, :-1] != valid[:, 1:]
    finite = np.isfinite(depth_map)
    if np.any(finite):
        values = depth_map[finite]
        threshold = max(1e-6, float(values.max() - values.min()) * 0.09)
        dz_y = np.zeros_like(valid, dtype=bool)
        dz_x = np.zeros_like(valid, dtype=bool)
        both_y = finite[:-1, :] & finite[1:, :]
        both_x = finite[:, :-1] & finite[:, 1:]
        dz_y[:-1, :] = both_y & (np.abs(depth_map[:-1, :] - depth_map[1:, :]) > threshold)
        dz_x[:, :-1] = both_x & (np.abs(depth_map[:, :-1] - depth_map[:, 1:]) > threshold)
        edge |= dz_y | dz_x
    return edge


def make_natural_rgb(
    depth_map: np.ndarray,
    normal_map: np.ndarray,
    source_map: np.ndarray,
    confidence: np.ndarray,
    smooth: bool,
) -> np.ndarray:
    valid = source_map > 0
    rgb = np.zeros((*source_map.shape, 3), dtype=np.uint8)
    rgb[..., :] = BACKGROUND
    if not np.any(valid):
        return rgb

    values = depth_map[valid]
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    depth_norm = np.zeros_like(depth_map, dtype=np.float32)
    depth_norm[valid] = 0.5 if hi - lo < 1e-6 else (depth_map[valid] - lo) / (hi - lo)

    light = np.array([0.35, -0.45, 0.82], dtype=np.float32)
    light = light / np.linalg.norm(light)
    normal_strength = np.abs((normal_map * light.reshape(1, 1, 3)).sum(axis=2))
    has_normals = np.linalg.norm(normal_map, axis=2) > 0
    lambert = np.where(has_normals, 0.70 + 0.30 * normal_strength, 1.0)
    shade = (0.58 + 0.38 * depth_norm) * lambert
    shade *= 0.92 + 0.08 * confidence

    material = MATERIAL.reshape(1, 1, 3) * shade[..., None]
    material[source_map == 2] = material[source_map == 2] * 0.96 + np.array([18, 22, 28], dtype=np.float32) * 0.04
    rgb[valid] = np.clip(material[valid], 0, 255).astype(np.uint8)

    if smooth:
        image = Image.fromarray(rgb, mode="RGB").filter(ImageFilter.SMOOTH_MORE)
        rgb = np.asarray(image).copy()
        rgb[~valid] = BACKGROUND

    rgb[edge_mask(valid, depth_map)] = EDGE
    return rgb


def confidence_rgb(confidence: np.ndarray) -> np.ndarray:
    valid = confidence > 0
    rgb = np.zeros((*confidence.shape, 3), dtype=np.uint8)
    rgb[..., :] = BACKGROUND
    blend = confidence[..., None]
    rgb[valid] = np.clip(CONFIDENCE_LOW * (1 - blend[valid]) + CONFIDENCE_HIGH * blend[valid], 0, 255).astype(np.uint8)
    return rgb


def crop_foreground(image: Image.Image, source_map: np.ndarray, padding: int = 48) -> Image.Image:
    ys, xs = np.where(source_map > 0)
    if len(xs) == 0:
        return image.copy()
    w, h = image.size
    x0 = max(0, int(xs.min()) - padding)
    x1 = min(w - 1, int(xs.max()) + padding)
    y0 = max(0, int(ys.min()) - padding)
    y1 = min(h - 1, int(ys.max()) + padding)
    return image.crop((x0, y0, x1 + 1, y1 + 1))


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    font = ImageFont.load_default()
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle([bbox[0] - 5, bbox[1] - 4, bbox[2] + 5, bbox[3] + 4], fill=(8, 10, 14), outline=(88, 96, 112))
    draw.text((x, y), text, fill=(235, 238, 245), font=font)


def make_panel(natural_rgb: np.ndarray, conf_rgb: np.ndarray, source_map: np.ndarray) -> Image.Image:
    natural = Image.fromarray(natural_rgb, mode="RGB")
    conf = Image.fromarray(conf_rgb, mode="RGB")
    zoom = crop_foreground(natural, source_map)
    w, h = natural.size
    resample = getattr(Image, "Resampling", Image).BICUBIC
    scale = min(h / max(1, zoom.width), h / max(1, zoom.height))
    zoom = zoom.resize((max(1, int(zoom.width * scale)), max(1, int(zoom.height * scale))), resample)
    zoom_canvas = Image.new("RGB", (h, h), tuple(BACKGROUND.tolist()))
    zoom_canvas.paste(zoom, ((h - zoom.width) // 2, (h - zoom.height) // 2))

    panel = Image.new("RGB", (w * 3, h), tuple(BACKGROUND.tolist()))
    panel.paste(natural, (0, 0))
    panel.paste(conf, (w, 0))
    panel.paste(zoom_canvas, (w * 2, 0))
    draw = ImageDraw.Draw(panel)
    draw.rectangle([w, 0, w * 2 - 1, h - 1], outline=(88, 96, 112), width=1)
    draw.rectangle([w * 2, 0, w * 3 - 1, h - 1], outline=(88, 96, 112), width=1)
    draw_label(draw, (12, h - 28), "Natural-like render")
    draw_label(draw, (w + 12, h - 28), "Backprojection confidence")
    draw_label(draw, (w * 2 + 12, h - 28), "Foreground zoom")
    return panel


def render_view(
    points_xyz: np.ndarray,
    normals: np.ndarray | None,
    view: str,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    px, py, depth, view_normals = project_points(points_xyz, normals, view, args.image_size, args.padding)
    exact_index, exact_depth, _, _ = splat_index_map(px, py, depth, view_normals, args.image_size, args.index_radius)
    index_map, depth_map, dist_map, normal_map = splat_index_map(
        px, py, depth, view_normals, args.image_size, args.splat_radius
    )
    index_map, depth_map, dist_map, normal_map, source_map, confidence = fill_small_holes(
        index_map, depth_map, dist_map, normal_map, args.fill_radius
    )
    natural_rgb = make_natural_rgb(depth_map, normal_map, source_map, confidence, args.smooth)
    conf_rgb = confidence_rgb(confidence)
    return {
        "natural_rgb": natural_rgb,
        "confidence_rgb": conf_rgb,
        "point_index": index_map,
        "exact_point_index": exact_index,
        "depth": depth_map,
        "exact_depth": exact_depth,
        "source": source_map,
        "confidence": confidence,
    }


def render_sample(
    root: Path,
    samples: list[dict[str, Any]],
    sample_id: str,
    views: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    sample = find_sample(samples, sample_id)
    point_path = resolve_path(root, sample["point_cloud_path"])
    xyz_raw, normals = load_points_with_normals(point_path)
    xyz = normalize_points(xyz_raw)
    sample_dir = resolve_path(root, args.output_root) / sample_id
    if sample_dir.exists() and any(sample_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory exists. Use --overwrite: {sample_dir}")
    sample_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "version": "natural_surface_vlm_v0",
        "sample_id": sample_id,
        "object_id": sample.get("object_id", ""),
        "object_category": sample.get("object_category", ""),
        "task": sample.get("task", ""),
        "point_cloud_path": sample.get("point_cloud_path", ""),
        "num_points": int(xyz.shape[0]),
        "image_size": int(args.image_size),
        "splat_radius": int(args.splat_radius),
        "index_radius": int(args.index_radius),
        "fill_radius": int(args.fill_radius),
        "notes": "Natural-like images are VLM inputs. point_index/confidence/source maps are used for 2D-to-3D projection.",
        "views": [],
    }

    for view in views:
        rendered = render_view(xyz, normals, view, args)
        natural_path = sample_dir / f"{view}_natural.png"
        conf_path = sample_dir / f"{view}_confidence.png"
        point_index_path = sample_dir / f"{view}_point_index.npy"
        exact_index_path = sample_dir / f"{view}_exact_point_index.npy"
        depth_path = sample_dir / f"{view}_depth.npy"
        source_path = sample_dir / f"{view}_source.npy"
        confidence_path = sample_dir / f"{view}_confidence.npy"
        Image.fromarray(rendered["natural_rgb"], mode="RGB").save(natural_path)
        Image.fromarray(rendered["confidence_rgb"], mode="RGB").save(conf_path)
        np.save(point_index_path, rendered["point_index"])
        np.save(exact_index_path, rendered["exact_point_index"])
        np.save(depth_path, rendered["depth"])
        np.save(source_path, rendered["source"])
        np.save(confidence_path, rendered["confidence"])

        panel_path = None
        if not args.no_panel:
            panel_path = sample_dir / f"{view}_panel.png"
            make_panel(rendered["natural_rgb"], rendered["confidence_rgb"], rendered["source"]).save(panel_path)

        manifest["views"].append(
            {
                "view": view,
                "natural_render_path": relative_to_dataset(root, natural_path),
                "confidence_image_path": relative_to_dataset(root, conf_path),
                "point_index_path": relative_to_dataset(root, point_index_path),
                "exact_point_index_path": relative_to_dataset(root, exact_index_path),
                "depth_path": relative_to_dataset(root, depth_path),
                "source_path": relative_to_dataset(root, source_path),
                "confidence_path": relative_to_dataset(root, confidence_path),
                "panel_path": relative_to_dataset(root, panel_path) if panel_path else "",
                "valid_pixel_fraction": float(np.mean(rendered["point_index"] >= 0)),
                "direct_pixel_fraction": float(np.mean(rendered["source"] == 1)),
                "filled_pixel_fraction": float(np.mean(rendered["source"] == 2)),
            }
        )

    manifest_path = sample_dir / "view_manifest.json"
    write_json(manifest_path, manifest, args.overwrite)
    return {"sample_id": sample_id, "manifest": relative_to_dataset(root, manifest_path), "views": views}


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    samples = read_jsonl(resolve_path(root, args.samples))
    sample_ids = select_sample_ids(root, args)
    views = parse_views(args.views)
    rendered = [render_sample(root, samples, sample_id, views, args) for sample_id in sample_ids]
    print(json.dumps({"rendered_samples": len(rendered), "rows": rendered}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
