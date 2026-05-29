#!/usr/bin/env python3
"""Serve a read-only viewer for reviewed Multi-EE dataset samples."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import Counter
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.task_taxonomy import EXECUTOR_ORDER, NEW_DEFAULT_ACTIVE_TASKS


DEFAULT_ACTIVE_TASKS = set(NEW_DEFAULT_ACTIVE_TASKS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a read-only reviewed dataset viewer.")
    parser.add_argument("--dataset-root", default=".", help="Dataset root directory.")
    parser.add_argument(
        "--samples",
        default="processed/metadata/reviewed_dataset_v0_1.jsonl",
        help="Reviewed dataset JSONL relative to dataset root.",
    )
    parser.add_argument(
        "--fallback-samples",
        default=(
            "processed/metadata/v2_manual_refined_samples_v0_1.jsonl,"
            "processed/metadata/v3_manual_refined_samples_v0_1.jsonl"
        ),
        help="Comma-separated fallback reviewed JSONLs if --samples does not exist.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--max-points", type=int, default=0, help="0 sends all points.")
    parser.add_argument("--include-tasks", default=",".join(NEW_DEFAULT_ACTIVE_TASKS), help="Comma-separated tasks or 'all'.")
    parser.add_argument("--exclude-tasks", default="", help="Comma-separated tasks to hide.")
    return parser.parse_args()


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def parse_task_filter(value: str) -> set[str] | None:
    raw = str(value or "").strip()
    if raw.lower() == "all":
        return None
    return set(parse_list(raw))


def sample_key(row: dict[str, Any]) -> str:
    explicit = str(row.get("row_key") or "").strip()
    if explicit:
        return explicit
    point_edit = row.get("v2_point_edit", {}) if isinstance(row.get("v2_point_edit"), dict) else {}
    update = row.get("v2_candidate_update", {}) if isinstance(row.get("v2_candidate_update"), dict) else {}
    executor = str(point_edit.get("executor") or update.get("executor") or row.get("executor") or "").strip()
    task = str(row.get("task") or row.get("target_task") or "").strip()
    return "|".join(part for part in (str(row.get("pilot_id") or ""), str(row.get("sample_id") or ""), task, executor) if part)


def target_executor(row: dict[str, Any]) -> str:
    point_edit = row.get("v2_point_edit", {}) if isinstance(row.get("v2_point_edit"), dict) else {}
    update = row.get("v2_candidate_update", {}) if isinstance(row.get("v2_candidate_update"), dict) else {}
    executor = str(point_edit.get("executor") or update.get("executor") or row.get("executor") or "")
    return executor if executor in EXECUTOR_ORDER else "hook"


def normalize_points(points: np.ndarray) -> np.ndarray:
    xyz = points[:, :3].astype(np.float32, copy=False)
    center = xyz.mean(axis=0)
    shifted = xyz - center
    scale = float(np.linalg.norm(shifted, axis=1).max())
    if scale <= 1e-12:
        scale = 1.0
    return shifted / scale


def compact_points(points: np.ndarray) -> list[list[float]]:
    return np.round(points.astype(np.float64), 5).tolist()


def choose_indices(n: int, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or n <= max_points:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_points, replace=False)).astype(np.int64)


class ReviewedDatasetStore:
    def __init__(self, dataset_root: Path, samples_path: Path, fallback_values: str, max_points: int, include_tasks: str, exclude_tasks: str):
        self.dataset_root = dataset_root
        self.samples_path = samples_path
        self.fallback_values = fallback_values
        self.max_points = int(max_points)
        self.include_tasks = parse_task_filter(include_tasks)
        self.exclude_tasks = set(parse_list(exclude_tasks))
        self.reload()

    def load_rows(self) -> tuple[list[dict[str, Any]], list[str]]:
        paths = [self.samples_path] if self.samples_path.exists() else [resolve_path(self.dataset_root, item) for item in parse_list(self.fallback_values)]
        rows_by_key: dict[str, dict[str, Any]] = {}
        sources: list[str] = []
        for path in paths:
            if not path.exists():
                continue
            sources.append(self.rel(path))
            for row in read_jsonl(path):
                key = sample_key(row)
                if key:
                    rows_by_key[key] = row
        if not rows_by_key:
            raise FileNotFoundError("No reviewed dataset rows found. Build a release or pass --fallback-samples.")
        return list(rows_by_key.values()), sources

    def rel(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.dataset_root.resolve()).as_posix()
        except ValueError:
            return str(path)

    def reload(self) -> None:
        rows, sources = self.load_rows()
        filtered: list[dict[str, Any]] = []
        for row in rows:
            task = str(row.get("task") or "")
            if self.include_tasks is not None and task not in self.include_tasks:
                continue
            if task in self.exclude_tasks:
                continue
            filtered.append(row)
        self.samples = filtered
        self.sources = sources
        self.samples_by_key = {sample_key(row): row for row in self.samples}

    def list_samples(self) -> dict[str, Any]:
        rows = []
        for row in self.samples:
            executor = target_executor(row)
            mask_count = self.metadata_positive_count(row)
            rows.append(
                {
                    "row_key": sample_key(row),
                    "pilot_id": row.get("pilot_id", ""),
                    "object_id": row.get("object_id", ""),
                    "sample_id": row.get("sample_id", ""),
                    "object_category": row.get("object_category", ""),
                    "task": row.get("task", ""),
                    "executor": executor,
                    "quality_flag": row.get("quality_flag", ""),
                    "review_status": row.get("point_review_status", ""),
                    "positive_points": mask_count,
                    "split": row.get("split", ""),
                }
            )
        summary = {
            "count": len(rows),
            "sources": self.sources,
            "tasks": dict(sorted(Counter(str(row["task"]) for row in rows).items())),
            "executors": dict(sorted(Counter(str(row["executor"]) for row in rows).items())),
            "categories": dict(sorted(Counter(str(row["object_category"]) for row in rows).items())),
        }
        return {"samples": rows, "summary": summary}

    def metadata_positive_count(self, row: dict[str, Any]) -> int:
        release = row.get("reviewed_dataset_release", {}) if isinstance(row.get("reviewed_dataset_release"), dict) else {}
        point_edit = row.get("v2_point_edit", {}) if isinstance(row.get("v2_point_edit"), dict) else {}
        update = row.get("v2_candidate_update", {}) if isinstance(row.get("v2_candidate_update"), dict) else {}
        for value in (
            release.get("target_positive_points"),
            point_edit.get("positive_points_after"),
            update.get("positive_points"),
        ):
            try:
                if value not in (None, ""):
                    return int(value)
            except (TypeError, ValueError):
                continue
        return 0

    def load_mask(self, row: dict[str, Any], executor: str) -> np.ndarray:
        path = resolve_path(self.dataset_root, str(row.get("multi_channel_mask_path") or ""))
        if not path.exists():
            raise FileNotFoundError(f"Mask not found: {path}")
        raw = np.load(path, allow_pickle=False)
        if raw.ndim == 2 and raw.shape[1] == len(EXECUTOR_ORDER):
            return raw.astype(np.uint8)
        if raw.ndim == 1:
            masks = np.zeros((raw.shape[0], len(EXECUTOR_ORDER)), dtype=np.uint8)
            masks[:, EXECUTOR_ORDER.index(executor)] = (raw > 0).astype(np.uint8)
            return masks
        raise ValueError(f"Bad mask shape: {raw.shape}")

    def sample_payload(self, key: str) -> dict[str, Any]:
        row = self.samples_by_key.get(key)
        if row is None:
            raise KeyError(f"Unknown sample key: {key}")
        executor = target_executor(row)
        points_path = resolve_path(self.dataset_root, str(row.get("point_cloud_path") or ""))
        points = np.load(points_path, allow_pickle=False)
        if points.ndim != 2 or points.shape[1] not in (3, 6):
            raise ValueError(f"Invalid point cloud shape: {points.shape}")
        masks = self.load_mask(row, executor)
        if masks.shape[0] != points.shape[0]:
            raise ValueError(f"Mask/point length mismatch: mask={masks.shape}, points={points.shape}")
        seed = abs(hash(key)) % (2**32)
        indices = choose_indices(points.shape[0], self.max_points, seed)
        visible_masks = (masks[indices] > 0).astype(np.uint8)
        point_edit = row.get("v2_point_edit", {}) if isinstance(row.get("v2_point_edit"), dict) else {}
        return {
            "row_key": key,
            "sample": row,
            "executor_order": EXECUTOR_ORDER,
            "target_executor": executor,
            "target_channel": EXECUTOR_ORDER.index(executor),
            "points": compact_points(normalize_points(points)[indices]),
            "point_indices": indices.astype(int).tolist(),
            "masks": visible_masks.tolist(),
            "counts": {name: int((masks[:, idx] > 0).sum()) for idx, name in enumerate(EXECUTOR_ORDER)},
            "review": {
                "status": row.get("point_review_status", ""),
                "decision": row.get("point_review_decision", ""),
                "quality": row.get("quality_flag", ""),
                "reviewer": row.get("point_review_reviewer", ""),
                "notes": row.get("point_review_notes", ""),
                "updated_at": row.get("point_review_updated_at", ""),
                "selected_candidate_ids": point_edit.get("selected_candidate_ids", []),
            },
            "paths": {
                "point_cloud_path": row.get("point_cloud_path", ""),
                "mask_path": row.get("multi_channel_mask_path", ""),
            },
        }


APP_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Multi-EE Reviewed Dataset Viewer</title>
  <style>
    * { box-sizing: border-box; }
    :root {
      --blue-900: #0f2f57;
      --blue-800: #16477f;
      --blue-700: #1d5fbf;
      --blue-600: #2563eb;
      --blue-100: #dbeafe;
      --blue-050: #f4f8ff;
      --border: #d7e5f6;
      --text: #172033;
      --muted: #64748b;
      --panel: #ffffff;
    }
    body { margin: 0; background: var(--blue-050); color: var(--text); font-family: Arial, "Microsoft YaHei", sans-serif; }
    header { height: 56px; padding: 0 18px; background: linear-gradient(90deg, var(--blue-900), var(--blue-700)); color: white; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 2px 12px rgba(15,47,87,.18); }
    header h1 { margin: 0; font-size: 18px; letter-spacing: 0; }
    .app { display: grid; grid-template-columns: 330px minmax(660px, 1fr) 390px; height: calc(100vh - 56px); }
    aside, .panel { background: var(--panel); overflow: auto; }
    aside { border-right: 1px solid var(--border); padding: 14px; }
    .panel { border-left: 1px solid var(--border); padding: 16px; }
    .viewer { position: relative; overflow: hidden; background: radial-gradient(circle at 50% 42%, #ffffff 0, #f7fbff 48%, #eef6ff 100%); }
    input, button, select { font-family: inherit; font-size: 13px; }
    input, select { width: 100%; padding: 9px 10px; border: 1px solid var(--border); border-radius: 8px; outline: none; color: var(--text); background: #fff; }
    input:focus { border-color: var(--blue-600); box-shadow: 0 0 0 3px rgba(37,99,235,.12); }
    button { border: 1px solid var(--blue-600); background: var(--blue-600); color: white; border-radius: 8px; padding: 8px 11px; cursor: pointer; }
    button.secondary { background: #fff; color: var(--blue-700); border-color: var(--border); }
    button.active { background: var(--blue-900); border-color: var(--blue-900); }
    .stats { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin: 12px 0; }
    .stat { border: 1px solid var(--border); border-radius: 10px; background: #f8fbff; padding: 9px; }
    .stat b { display: block; font-size: 18px; color: var(--blue-700); }
    .stat span { color: var(--muted); font-size: 12px; }
    .sample-list { display: flex; flex-direction: column; gap: 8px; margin-top: 10px; }
    .sample-section { margin: 12px 0 5px; padding: 6px 8px; border-radius: 999px; background: #eaf2ff; color: var(--blue-900); font-size: 12px; font-weight: 700; display: flex; justify-content: space-between; }
    .sample { border: 1px solid var(--border); border-radius: 10px; padding: 10px; cursor: pointer; background: #fff; transition: border .15s, box-shadow .15s, transform .15s; }
    .sample:hover { border-color: #9ec5fe; transform: translateY(-1px); }
    .sample.active { border-color: var(--blue-600); box-shadow: 0 0 0 3px rgba(37,99,235,.13); }
    .sample-id { font-size: 11px; color: #334155; word-break: break-all; line-height: 1.35; }
    .tags { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 7px; }
    .variant-row { margin-top: 7px; padding: 7px; border-radius: 8px; border: 1px solid #e4edf8; background: #f8fbff; }
    .variant-row.active { border-color: var(--blue-600); background: #eff6ff; }
    .tag { font-size: 11px; padding: 2px 7px; border-radius: 999px; background: var(--blue-100); color: var(--blue-900); }
    .tag.checked, .tag.verified { background: #dcfce7; color: #166534; }
    .tag.pending { background: #fef9c3; color: #854d0e; }
    canvas { width: 100%; height: 100%; display: block; cursor: grab; }
    canvas.dragging { cursor: grabbing; }
    .toolbar { position: absolute; left: 14px; top: 14px; right: 14px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; z-index: 2; }
    .task-banner { position: absolute; left: 14px; right: 14px; top: 58px; z-index: 2; display: flex; justify-content: center; pointer-events: none; }
    .task-banner-inner { background: rgba(255,255,255,.94); border: 1px solid var(--border); box-shadow: 0 8px 24px rgba(15,47,87,.10); color: var(--blue-900); border-radius: 12px; padding: 9px 14px; font-size: 15px; font-weight: 750; }
    .task-banner-inner span { display: inline-block; margin: 0 5px; padding: 2px 8px; border-radius: 999px; background: #eaf2ff; color: var(--blue-700); font-size: 12px; font-weight: 700; }
    .toolbar label { display: inline-flex; align-items: center; gap: 7px; background: rgba(255,255,255,.92); color: #334155; border: 1px solid var(--border); border-radius: 8px; padding: 7px 9px; font-size: 12px; }
    .toolbar input[type="checkbox"] { width: auto; }
    .hud { position: absolute; left: 14px; bottom: 14px; background: rgba(15,47,87,.86); color: white; padding: 10px 12px; border-radius: 10px; font-size: 12px; line-height: 1.55; max-width: 430px; }
    .card { border: 1px solid var(--border); background: #fff; border-radius: 10px; padding: 12px; margin-bottom: 12px; }
    .card h2 { font-size: 15px; margin: 0 0 9px; color: var(--blue-900); }
    .kv { display: grid; grid-template-columns: 120px 1fr; gap: 7px 10px; font-size: 12px; line-height: 1.4; }
    .kv .k { color: var(--muted); }
    .kv .v { color: var(--text); word-break: break-all; }
    .bar { height: 8px; border-radius: 99px; background: #eaf2ff; overflow: hidden; margin-top: 6px; }
    .bar span { display: block; height: 100%; background: linear-gradient(90deg, #60a5fa, #2563eb); }
    .legend { display: grid; gap: 8px; }
    .legend-row { display: grid; grid-template-columns: 14px 1fr auto; gap: 8px; align-items: center; font-size: 12px; }
    .swatch { width: 12px; height: 12px; border-radius: 3px; }
    code { color: var(--blue-700); }
  </style>
</head>
<body>
<header>
  <h1>Multi-EE 已审查数据集可视化</h1>
  <div id="topStatus">loading...</div>
</header>
<div class="app">
  <aside>
    <input id="search" placeholder="搜索 sample/category/task/executor" />
    <div class="stats">
      <div class="stat"><b id="statCount">0</b><span>reviewed samples</span></div>
      <div class="stat"><b id="statTask">0</b><span>task types</span></div>
    </div>
    <div class="sample-list" id="sampleList"></div>
  </aside>
  <main class="viewer">
    <div class="toolbar">
      <button id="resetView" class="secondary">重置视角</button>
      <button id="targetOnly" class="active">只看目标通道</button>
      <button id="allChannels" class="secondary">显示全部通道</button>
      <label><input id="largePoints" type="checkbox" checked /> 大点显示</label>
    </div>
    <div class="task-banner" id="taskBanner"></div>
    <canvas id="canvas"></canvas>
    <div class="hud" id="hud">请选择样本</div>
  </main>
  <section class="panel">
    <div class="card">
      <h2>样本信息</h2>
      <div class="kv" id="sampleMeta"></div>
    </div>
    <div class="card">
      <h2>执行器通道统计</h2>
      <div class="legend" id="channelLegend"></div>
    </div>
    <div class="card">
      <h2>人工审查记录</h2>
      <div class="kv" id="reviewMeta"></div>
    </div>
    <div class="card">
      <h2>文件路径</h2>
      <div class="kv" id="pathMeta"></div>
    </div>
  </section>
</div>
<script>
let samples = [];
let summary = {};
let current = null;
let currentKey = "";
let targetOnly = true;
let rotX = -0.55, rotY = 0.65, zoom = 1.0;
let dragging = false, lastX = 0, lastY = 0;
let projectedCache = null, projectedCacheKey = "";
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const channelColors = {
  gripper: "#2563eb",
  suction: "#14b8a6",
  hook: "#f59e0b",
  dexterous_hand: "#a855f7"
};

function renderSummary() {
  document.getElementById("statCount").textContent = summary.count || 0;
  document.getElementById("statTask").textContent = Object.keys(summary.tasks || {}).length;
  document.getElementById("topStatus").textContent = `${summary.count || 0} reviewed | ${Object.entries(summary.tasks || {}).map(([k,v]) => `${k}:${v}`).join("  ")}`;
}

function renderList() {
  const q = document.getElementById("search").value.toLowerCase();
  const list = document.getElementById("sampleList");
  list.innerHTML = "";
  const filtered = samples.filter(s => `${s.sample_id} ${s.object_category} ${s.task} ${s.executor}`.toLowerCase().includes(q));
  const objectKey = (s) => s.object_id || String(s.sample_id || "").replace(/_(pick_up|open_pull|press_push|lift_carry|lift|open|pull|press|push)$/,"");
  const groups = new Map();
  filtered.forEach(s => {
    const key = objectKey(s);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(s);
  });
  const sections = [
    ["checked", "已审查", rows => rows.some(x => (x.review_status || x.quality_flag || "") === "checked" || (x.review_status || x.quality_flag || "") === "verified")],
    ["other", "其他", rows => !rows.some(x => (x.review_status || x.quality_flag || "") === "checked" || (x.review_status || x.quality_flag || "") === "verified")],
  ];
  sections.forEach(([keyName, title, predicate]) => {
    const grouped = [...groups.entries()].filter(([, rows]) => predicate(rows));
    if (!grouped.length) return;
    const header = document.createElement("div");
    header.className = "sample-section";
    header.innerHTML = `<span>${title}</span><span>${grouped.length}</span>`;
    list.appendChild(header);
    grouped.forEach(([key, rows]) => {
      rows.sort((a,b) => `${a.task} ${a.executor}`.localeCompare(`${b.task} ${b.executor}`));
      const first = rows[0];
      const active = rows.some(x => x.row_key === currentKey);
      const div = document.createElement("div");
      div.className = "sample" + (active ? " active" : "");
      div.onclick = () => loadSample(rows[0].row_key);
      div.innerHTML = `<div class="sample-id">${key}</div><div class="tags"><span class="tag">${first.object_category || ""}</span><span class="tag">${rows.length} variants</span></div>`;
      rows.forEach(s => {
        const v = document.createElement("div");
        v.className = "variant-row" + (s.row_key === currentKey ? " active" : "");
        v.onclick = (event) => { event.stopPropagation(); loadSample(s.row_key); };
        v.innerHTML = `<div class="tags" style="margin-top:0">
          <span class="tag">${s.task || ""}</span>
          <span class="tag">${s.executor || ""}</span>
          <span class="tag ${s.review_status || ""}">${s.review_status || s.quality_flag || ""}</span>
          <span class="tag">pos=${s.positive_points || 0}</span>
        </div>`;
        div.appendChild(v);
      });
      list.appendChild(div);
    });
  });
  return;
  filtered.forEach(s => {
    const div = document.createElement("div");
    div.className = "sample" + (s.row_key === currentKey ? " active" : "");
    div.onclick = () => loadSample(s.row_key);
    div.innerHTML = `<div class="sample-id">${s.sample_id}</div>
      <div class="tags">
        <span class="tag">${s.object_category || ""}</span>
        <span class="tag">${s.task || ""}</span>
        <span class="tag">${s.executor || ""}</span>
        <span class="tag ${s.review_status || ""}">${s.review_status || s.quality_flag || ""}</span>
        <span class="tag">pos=${s.positive_points || 0}</span>
      </div>`;
    list.appendChild(div);
  });
}

async function loadSamples() {
  const res = await fetch("/api/samples");
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  samples = data.samples;
  summary = data.summary;
  renderSummary();
  renderList();
  if (samples.length) loadSample(samples[0].row_key);
}

async function loadSample(key) {
  const res = await fetch(`/api/sample?key=${encodeURIComponent(key)}`);
  if (!res.ok) throw new Error(await res.text());
  current = await res.json();
  currentKey = key;
  projectedCache = null;
  fillPanel();
  renderList();
  resize();
}

function kvHtml(items) {
  return items.map(([k, v]) => `<div class="k">${k}</div><div class="v">${v || ""}</div>`).join("");
}

function fillPanel() {
  if (!current) return;
  const s = current.sample;
  document.getElementById("taskBanner").innerHTML =
    `<div class="task-banner-inner">${s.object_category || ""}<span>${s.task || ""}</span><span>${current.target_executor}</span>positive=${current.counts[current.target_executor] || 0}</div>`;
  document.getElementById("sampleMeta").innerHTML = kvHtml([
    ["sample_id", s.sample_id],
    ["category", s.object_category],
    ["task", s.task],
    ["target_executor", current.target_executor],
    ["split", s.split],
    ["quality", s.quality_flag]
  ]);
  document.getElementById("reviewMeta").innerHTML = kvHtml([
    ["status", current.review.status],
    ["decision", current.review.decision],
    ["reviewer", current.review.reviewer],
    ["updated_at", current.review.updated_at],
    ["candidates", (current.review.selected_candidate_ids || []).join(",")],
    ["notes", current.review.notes]
  ]);
  document.getElementById("pathMeta").innerHTML = kvHtml([
    ["points", current.paths.point_cloud_path],
    ["mask", current.paths.mask_path]
  ]);
  const maxCount = Math.max(1, ...Object.values(current.counts || {}));
  document.getElementById("channelLegend").innerHTML = current.executor_order.map(name => {
    const count = current.counts[name] || 0;
    const pct = Math.round(count / maxCount * 100);
    return `<div class="legend-row"><span class="swatch" style="background:${channelColors[name]}"></span><span>${name}</span><b>${count}</b></div><div class="bar" style="grid-column:1 / -1"><span style="width:${pct}%"></span></div>`;
  }).join("");
}

function project(p) {
  const sx = Math.sin(rotX), cx = Math.cos(rotX);
  const sy = Math.sin(rotY), cy = Math.cos(rotY);
  let x = p[0], y = p[1], z = p[2];
  let x1 = cy * x + sy * z;
  let z1 = -sy * x + cy * z;
  let y1 = cx * y - sx * z1;
  let z2 = sx * y + cx * z1;
  const scale = Math.min(canvas.clientWidth, canvas.clientHeight) * 0.43 * zoom;
  return [canvas.clientWidth / 2 + x1 * scale, canvas.clientHeight / 2 - y1 * scale, z2];
}

function projectedPoints() {
  if (!current) return [];
  const key = [current.row_key, canvas.clientWidth, canvas.clientHeight, rotX.toFixed(5), rotY.toFixed(5), zoom.toFixed(5)].join("|");
  if (projectedCache && projectedCacheKey === key) return projectedCache;
  projectedCache = current.points.map((p, i) => {
    const pr = project(p);
    return {x: pr[0], y: pr[1], z: pr[2], i};
  }).sort((a,b) => a.z - b.z);
  projectedCacheKey = key;
  return projectedCache;
}

function draw() {
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  if (!current) return;
  const large = document.getElementById("largePoints").checked;
  const targetCh = current.target_channel;
  for (const p of projectedPoints()) {
    const mask = current.masks[p.i] || [];
    const isTarget = !!mask[targetCh];
    let color = "#7f8fa3";
    let alpha = 0.52;
    let radius = large ? 3.4 : 2.5;
    if (isTarget) {
      color = channelColors[current.target_executor] || "#2563eb";
      alpha = 0.96;
      radius = large ? 5.0 : 3.7;
    } else if (!targetOnly) {
      const ch = mask.findIndex(v => v);
      if (ch >= 0) {
        const name = current.executor_order[ch];
        color = channelColors[name] || "#60a5fa";
        alpha = 0.55;
        radius = large ? 3.8 : 2.8;
      }
    }
    ctx.beginPath();
    ctx.fillStyle = color;
    ctx.globalAlpha = alpha;
    ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
  document.getElementById("hud").innerHTML =
    `sample=${current.sample.sample_id}<br/>target=${current.target_executor} | positive=${current.counts[current.target_executor] || 0}<br/>right-drag/drag=rotate | wheel=zoom`;
}

function resize() {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  projectedCache = null;
  draw();
}

canvas.addEventListener("mousedown", e => {
  dragging = true;
  canvas.classList.add("dragging");
  lastX = e.clientX;
  lastY = e.clientY;
});
window.addEventListener("mouseup", () => {
  dragging = false;
  canvas.classList.remove("dragging");
});
window.addEventListener("mousemove", e => {
  if (!dragging) return;
  rotY += (e.clientX - lastX) * 0.008;
  rotX += (e.clientY - lastY) * 0.008;
  lastX = e.clientX;
  lastY = e.clientY;
  projectedCache = null;
  draw();
});
canvas.addEventListener("wheel", e => {
  e.preventDefault();
  zoom *= e.deltaY < 0 ? 1.08 : 0.92;
  zoom = Math.max(0.35, Math.min(4.0, zoom));
  projectedCache = null;
  draw();
}, {passive:false});
canvas.addEventListener("contextmenu", e => e.preventDefault());

document.getElementById("search").oninput = renderList;
document.getElementById("resetView").onclick = () => { rotX = -0.55; rotY = 0.65; zoom = 1.0; projectedCache = null; draw(); };
document.getElementById("targetOnly").onclick = () => {
  targetOnly = true;
  document.getElementById("targetOnly").className = "active";
  document.getElementById("allChannels").className = "secondary";
  draw();
};
document.getElementById("allChannels").onclick = () => {
  targetOnly = false;
  document.getElementById("targetOnly").className = "secondary";
  document.getElementById("allChannels").className = "active";
  draw();
};
document.getElementById("largePoints").onchange = draw;
window.addEventListener("resize", resize);
loadSamples().catch(err => {
  document.getElementById("topStatus").textContent = String(err);
  console.error(err);
});
</script>
</body>
</html>
"""


class ViewerHandler(BaseHTTPRequestHandler):
    store: ReviewedDatasetStore

    def send_json(self, data: Any, status: int = 200) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_text(self, text: str, status: int = 200, content_type: str = "text/plain; charset=utf-8") -> None:
        payload = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path in ("", "/"):
                self.send_text(APP_HTML, content_type="text/html; charset=utf-8")
                return
            if parsed.path == "/api/samples":
                self.send_json(self.store.list_samples())
                return
            if parsed.path == "/api/sample":
                key = parse_qs(parsed.query).get("key", [""])[0]
                self.send_json(self.store.sample_payload(key))
                return
            self.send_text("Not found", status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            traceback.print_exc()
            self.send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    store = ReviewedDatasetStore(
        dataset_root=dataset_root,
        samples_path=resolve_path(dataset_root, args.samples),
        fallback_values=args.fallback_samples,
        max_points=args.max_points,
        include_tasks=args.include_tasks,
        exclude_tasks=args.exclude_tasks,
    )
    ViewerHandler.store = store
    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    print(f"Serving reviewed dataset viewer at http://{args.host}:{args.port}")
    print(f"Dataset root: {dataset_root}")
    print(f"Rows: {len(store.samples)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
