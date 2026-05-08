#!/usr/bin/env python3
"""Export standalone HTML point-cloud mask visualizations.

This is a dependency-light fallback for environments where Open3D/matplotlib
cannot be installed yet. It writes interactive HTML files with a canvas-based
3D projection, channel buttons, mouse drag rotation, and wheel zoom.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


EXECUTOR_ORDER = ["gripper", "suction", "hook", "dexterous_hand"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export standalone HTML visualizations for Multi-EE masks.")
    parser.add_argument("--dataset-root", default=".", help="Dataset root used for metadata-relative paths.")
    parser.add_argument("--samples", help="Optional samples.jsonl path for batch export.")
    parser.add_argument("--sample-id", action="append", help="Sample id to export. Can be used multiple times.")
    parser.add_argument("--task", action="append", help="Filter batch export by task. Can be used multiple times.")
    parser.add_argument("--limit", type=int, default=12, help="Maximum batch samples to export.")
    parser.add_argument("--points", help="Single point cloud .npy with shape [N,3] or [N,6].")
    parser.add_argument("--masks", help="Single mask .npy with shape [N,4].")
    parser.add_argument("--title", help="Single export title.")
    parser.add_argument("--output", help="Single output HTML path.")
    parser.add_argument(
        "--output-dir",
        default="processed/visualizations/html",
        help="Batch output directory relative to dataset root unless absolute.",
    )
    parser.add_argument("--write-index", action="store_true", help="Write an index.html page for batch exports.")
    parser.add_argument("--max-points", type=int, default=4096, help="Randomly downsample points for HTML export.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for downsampling.")
    return parser.parse_args()


def error(message: str) -> None:
    raise ValueError(message)


def resolve_path(root: Path, path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else root / path


def load_points(path: Path) -> np.ndarray:
    if not path.exists():
        error(f"Point file does not exist: {path}")
    points = np.load(path, allow_pickle=False)
    if points.ndim != 2 or points.shape[1] not in (3, 6):
        error(f"Point cloud must have shape [N,3] or [N,6], got {points.shape} from {path}")
    if points.shape[0] == 0:
        error(f"Point cloud is empty: {path}")
    return points[:, :3].astype(np.float32, copy=False)


def load_masks(path: Path, n_points: int) -> np.ndarray:
    if not path.exists():
        error(f"Mask file does not exist: {path}")
    masks = np.load(path, allow_pickle=False)
    if masks.ndim != 2 or masks.shape != (n_points, len(EXECUTOR_ORDER)):
        error(f"Mask must have shape [{n_points},4], got {masks.shape} from {path}")
    return (masks > 0).astype(np.uint8)


def normalize_points(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    shifted = points - center
    scale = float(np.linalg.norm(shifted, axis=1).max())
    if scale <= 1e-12:
        scale = 1.0
    return shifted / scale


def sample_arrays(points: np.ndarray, masks: np.ndarray, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0:
        error("--max-points must be positive")
    if points.shape[0] <= max_points:
        return points, masks
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(points.shape[0], size=max_points, replace=False))
    return points[indices], masks[indices]


def compact_float_rows(points: np.ndarray) -> list[list[float]]:
    rounded = np.round(points.astype(np.float64), 5)
    return rounded.tolist()


def html_template(title: str, points: np.ndarray, masks: np.ndarray, metadata: dict[str, Any]) -> str:
    counts = {executor: int(masks[:, index].sum()) for index, executor in enumerate(EXECUTOR_ORDER)}
    payload = {
        "title": title,
        "points": compact_float_rows(points),
        "masks": masks.astype(np.uint8).tolist(),
        "executors": EXECUTOR_ORDER,
        "counts": counts,
        "metadata": metadata,
    }
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape_html(title)}</title>
  <style>
    html, body {{ margin: 0; height: 100%; font-family: Arial, sans-serif; background: #111318; color: #f2f4f8; }}
    .layout {{ display: grid; grid-template-columns: 280px 1fr; height: 100%; }}
    aside {{ padding: 18px; background: #191d25; border-right: 1px solid #2b313d; overflow: auto; }}
    h1 {{ font-size: 16px; line-height: 1.35; margin: 0 0 12px; font-weight: 700; }}
    .meta {{ color: #b8c0cc; font-size: 12px; line-height: 1.55; word-break: break-all; }}
    .buttons {{ display: grid; grid-template-columns: 1fr; gap: 8px; margin: 18px 0; }}
    button {{ background: #262d3a; color: #f2f4f8; border: 1px solid #3a4354; border-radius: 6px; padding: 9px 10px; cursor: pointer; text-align: left; }}
    button.active {{ background: #da3f3f; border-color: #ff7070; }}
    .legend {{ font-size: 12px; color: #b8c0cc; }}
    .swatch {{ display: inline-block; width: 10px; height: 10px; margin-right: 6px; border-radius: 2px; }}
    main {{ position: relative; overflow: hidden; }}
    canvas {{ width: 100%; height: 100%; display: block; cursor: grab; }}
    canvas:active {{ cursor: grabbing; }}
    .hud {{ position: absolute; left: 14px; bottom: 12px; font-size: 12px; color: #cbd2dd; background: rgba(17, 19, 24, 0.72); padding: 8px 10px; border-radius: 6px; }}
  </style>
</head>
<body>
<div class="layout">
  <aside>
    <h1 id="title"></h1>
    <div class="meta" id="meta"></div>
    <div class="buttons" id="buttons"></div>
    <div class="legend">
      <div><span class="swatch" style="background:#d83c3c"></span>positive points</div>
      <div><span class="swatch" style="background:#aeb6c2"></span>other points</div>
      <div style="margin-top:10px">Drag to rotate. Mouse wheel to zoom.</div>
    </div>
  </aside>
  <main>
    <canvas id="view"></canvas>
    <div class="hud" id="hud"></div>
  </main>
</div>
<script>
const DATA = {data_json};
const canvas = document.getElementById('view');
const ctx = canvas.getContext('2d');
let channel = 'raw';
let rotX = -0.55;
let rotY = 0.65;
let zoom = 1.0;
let dragging = false;
let lastX = 0;
let lastY = 0;

document.getElementById('title').textContent = DATA.title;
document.getElementById('meta').innerHTML = Object.entries(DATA.metadata)
  .map(([k,v]) => `<div><b>${{k}}</b>: ${{String(v)}}</div>`).join('');

function addButton(name, label) {{
  const b = document.createElement('button');
  b.textContent = label;
  b.onclick = () => {{ channel = name; updateButtons(); draw(); }};
  b.dataset.channel = name;
  document.getElementById('buttons').appendChild(b);
}}
addButton('raw', `raw (${{DATA.points.length}})`);
DATA.executors.forEach((name) => addButton(name, `${{name}} (${{DATA.counts[name]}})`));

function updateButtons() {{
  document.querySelectorAll('button').forEach(b => b.classList.toggle('active', b.dataset.channel === channel));
}}

function resize() {{
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}}

function project(p) {{
  const sx = Math.sin(rotX), cx = Math.cos(rotX);
  const sy = Math.sin(rotY), cy = Math.cos(rotY);
  let x = p[0], y = p[1], z = p[2];
  let x1 = cy * x + sy * z;
  let z1 = -sy * x + cy * z;
  let y1 = cx * y - sx * z1;
  let z2 = sx * y + cx * z1;
  const scale = Math.min(canvas.clientWidth, canvas.clientHeight) * 0.42 * zoom;
  return [canvas.clientWidth / 2 + x1 * scale, canvas.clientHeight / 2 - y1 * scale, z2];
}}

function pointColor(maskRow) {{
  if (channel === 'raw') return '#aeb6c2';
  const idx = DATA.executors.indexOf(channel);
  return maskRow[idx] ? '#d83c3c' : '#aeb6c2';
}}

function draw() {{
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  const rows = DATA.points.map((p, i) => {{
    const q = project(p);
    return {{x: q[0], y: q[1], z: q[2], mask: DATA.masks[i]}};
  }}).sort((a, b) => a.z - b.z);
  const radius = Math.max(1.3, Math.min(3.2, 8000 / Math.max(2200, rows.length)));
  for (const row of rows) {{
    ctx.fillStyle = pointColor(row.mask);
    ctx.globalAlpha = channel === 'raw' ? 0.92 : (pointColor(row.mask) === '#d83c3c' ? 0.98 : 0.36);
    ctx.beginPath();
    ctx.arc(row.x, row.y, radius, 0, Math.PI * 2);
    ctx.fill();
  }}
  ctx.globalAlpha = 1;
  const count = channel === 'raw' ? DATA.points.length : DATA.counts[channel];
  document.getElementById('hud').textContent = `${{channel}} | points: ${{DATA.points.length}} | positive: ${{count}}`;
}}

canvas.addEventListener('mousedown', e => {{ dragging = true; lastX = e.clientX; lastY = e.clientY; }});
window.addEventListener('mouseup', () => {{ dragging = false; }});
window.addEventListener('mousemove', e => {{
  if (!dragging) return;
  rotY += (e.clientX - lastX) * 0.008;
  rotX += (e.clientY - lastY) * 0.008;
  lastX = e.clientX;
  lastY = e.clientY;
  draw();
}});
canvas.addEventListener('wheel', e => {{
  e.preventDefault();
  zoom *= e.deltaY > 0 ? 0.9 : 1.1;
  zoom = Math.max(0.25, Math.min(5, zoom));
  draw();
}}, {{passive: false}});
window.addEventListener('resize', resize);
updateButtons();
resize();
</script>
</body>
</html>
"""


def escape_html(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def export_one(
    points_path: Path,
    mask_path: Path,
    output_path: Path,
    title: str,
    metadata: dict[str, Any],
    max_points: int,
    seed: int,
) -> None:
    points = load_points(points_path)
    masks = load_masks(mask_path, points.shape[0])
    points, masks = sample_arrays(points, masks, max_points, seed)
    points = normalize_points(points)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_template(title, points, masks, metadata), encoding="utf-8")
    print(f"wrote {output_path}")


def read_samples(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def batch_export(args: argparse.Namespace) -> None:
    root = Path(args.dataset_root)
    samples_path = resolve_path(root, args.samples)
    if not samples_path.exists():
        error(f"samples file does not exist: {samples_path}")
    rows = read_samples(samples_path)
    sample_filter = set(args.sample_id or [])
    task_filter = set(args.task or [])
    if task_filter and not task_filter.issubset({"pick_up", "lift_carry", "open_pull", "press_push"}):
        error(f"Unknown task filter values: {sorted(task_filter)}")
    selected: list[dict[str, Any]] = []
    for row in rows:
        if sample_filter and row.get("sample_id") not in sample_filter:
            continue
        if task_filter and row.get("task") not in task_filter:
            continue
        selected.append(row)
        if args.limit is not None and len(selected) >= args.limit:
            break
    if not selected:
        error("No samples selected for HTML export")

    output_dir = resolve_path(root, args.output_dir)
    exported: list[tuple[str, str, str, str]] = []
    for index, row in enumerate(selected):
        sample_id = row.get("sample_id") or f"{row.get('object_id')}_{row.get('task')}"
        output_path = output_dir / f"{sample_id}.html"
        title = f"{sample_id} | {row.get('object_category')} | {row.get('task')}"
        metadata = {
            "sample_id": sample_id,
            "object_id": row.get("object_id"),
            "category": row.get("object_category"),
            "task": row.get("task"),
            "quality": row.get("quality_flag"),
            "points": row.get("point_cloud_path"),
            "mask": row.get("multi_channel_mask_path"),
        }
        export_one(
            resolve_path(root, row["point_cloud_path"]),
            resolve_path(root, row["multi_channel_mask_path"]),
            output_path,
            title,
            metadata,
            args.max_points,
            args.seed + index,
        )
        exported.append((output_path.name, str(sample_id), str(row.get("object_category")), str(row.get("task"))))
    if args.write_index:
        write_index(output_dir / "index.html", exported)


def write_index(path: Path, exported: list[tuple[str, str, str, str]]) -> None:
    rows = "\n".join(
        f"<tr><td><a href=\"{escape_html(filename)}\">{escape_html(sample_id)}</a></td>"
        f"<td>{escape_html(category)}</td><td>{escape_html(task)}</td></tr>"
        for filename, sample_id, category, task in exported
    )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Multi-EE 可视化索引</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f7f8fb; color: #1f2430; }}
    h1 {{ font-size: 22px; }}
    table {{ border-collapse: collapse; width: 100%; background: white; }}
    th, td {{ border: 1px solid #d9dee8; padding: 8px 10px; text-align: left; font-size: 13px; }}
    th {{ background: #edf1f7; }}
    a {{ color: #b42323; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <h1>Multi-EE Affordance HTML 可视化索引</h1>
  <p>共导出 {len(exported)} 个样本。打开单个页面后，可拖拽旋转点云，并切换 raw / gripper / suction / hook / dexterous_hand 通道。</p>
  <table>
    <thead><tr><th>sample_id</th><th>category</th><th>task</th></tr></thead>
    <tbody>
{rows}
    </tbody>
  </table>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
    print(f"wrote {path}")


def single_export(args: argparse.Namespace) -> None:
    if not args.points or not args.masks or not args.output:
        error("Single export requires --points, --masks, and --output")
    root = Path(args.dataset_root)
    title = args.title or Path(args.masks).stem
    metadata = {"points": args.points, "mask": args.masks}
    export_one(
        resolve_path(root, args.points),
        resolve_path(root, args.masks),
        resolve_path(root, args.output),
        title,
        metadata,
        args.max_points,
        args.seed,
    )


def main() -> int:
    args = parse_args()
    try:
        if args.samples:
            batch_export(args)
        else:
            single_export(args)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
