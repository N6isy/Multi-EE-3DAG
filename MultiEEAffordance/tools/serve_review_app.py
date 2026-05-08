#!/usr/bin/env python3
"""Serve a local manual-review web app for Multi-EE affordance samples.

The app lets a reviewer inspect one sample at a time and save review decisions
directly back to processed/metadata/manual_review_v0_1.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np


EXECUTOR_ORDER = ["gripper", "suction", "hook", "dexterous_hand"]
REVIEW_FIELDS = [
    "sample_id",
    "object_id",
    "object_category",
    "task",
    "visualization_html_path",
    "point_cloud_path",
    "multi_channel_mask_path",
    "review_status",
    "keep_sample",
    "quality_after_review",
    "sample_issue_type",
    "sample_notes",
    "gripper_decision",
    "gripper_issue_type",
    "gripper_notes",
    "suction_decision",
    "suction_issue_type",
    "suction_notes",
    "hook_decision",
    "hook_issue_type",
    "hook_notes",
    "dexterous_hand_decision",
    "dexterous_hand_issue_type",
    "dexterous_hand_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a browser-based manual review app.")
    parser.add_argument("--dataset-root", default=".", help="Dataset root.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind.")
    parser.add_argument("--samples", default="processed/metadata/samples.jsonl", help="samples.jsonl path.")
    parser.add_argument("--review-csv", default="processed/metadata/manual_review_v0_1.csv", help="Review CSV path.")
    parser.add_argument("--max-points", type=int, default=4096, help="Max points returned to the browser per sample.")
    return parser.parse_args()


def resolve_path(root: Path, path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else root / path


def read_samples(samples_path: Path) -> list[dict[str, Any]]:
    if not samples_path.exists():
        raise FileNotFoundError(f"samples.jsonl not found: {samples_path}")
    rows: list[dict[str, Any]] = []
    with samples_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if "sample_id" not in row:
                row["sample_id"] = f"{row.get('object_id', '')}_{row.get('task', '')}".strip("_")
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def initial_review_row(sample: dict[str, Any]) -> dict[str, str]:
    sample_id = sample["sample_id"]
    return {
        "sample_id": sample_id,
        "object_id": sample.get("object_id", ""),
        "object_category": sample.get("object_category", ""),
        "task": sample.get("task", ""),
        "visualization_html_path": f"processed/visualizations/html_v3/{sample_id}.html",
        "point_cloud_path": sample.get("point_cloud_path", ""),
        "multi_channel_mask_path": sample.get("multi_channel_mask_path", ""),
        "review_status": "pending",
        "keep_sample": "",
        "quality_after_review": "",
        "sample_issue_type": "",
        "sample_notes": "",
        "gripper_decision": "",
        "gripper_issue_type": "",
        "gripper_notes": "",
        "suction_decision": "",
        "suction_issue_type": "",
        "suction_notes": "",
        "hook_decision": "",
        "hook_issue_type": "",
        "hook_notes": "",
        "dexterous_hand_decision": "",
        "dexterous_hand_issue_type": "",
        "dexterous_hand_notes": "",
    }


def read_review_csv(review_path: Path, samples: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    if not review_path.exists():
        rows = {sample["sample_id"]: initial_review_row(sample) for sample in samples}
        write_review_csv(review_path, rows)
        return rows

    rows: dict[str, dict[str, str]] = {}
    with review_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Review CSV has no header: {review_path}")
        for row in reader:
            sample_id = row.get("sample_id", "")
            if sample_id:
                normalized = {field: row.get(field, "") or "" for field in REVIEW_FIELDS}
                rows[sample_id] = normalized

    changed = False
    for sample in samples:
        sample_id = sample["sample_id"]
        if sample_id not in rows:
            rows[sample_id] = initial_review_row(sample)
            changed = True
        else:
            base = initial_review_row(sample)
            for field in REVIEW_FIELDS:
                rows[sample_id].setdefault(field, base.get(field, ""))
            # Keep identity/path columns synchronized with current metadata.
            for field in (
                "object_id",
                "object_category",
                "task",
                "visualization_html_path",
                "point_cloud_path",
                "multi_channel_mask_path",
            ):
                if rows[sample_id].get(field) != base[field]:
                    rows[sample_id][field] = base[field]
                    changed = True
    if changed:
        write_review_csv(review_path, rows)
    return rows


def write_review_csv(review_path: Path, rows_by_id: dict[str, dict[str, str]]) -> None:
    review_path.parent.mkdir(parents=True, exist_ok=True)
    with review_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for sample_id in rows_by_id:
            row = {field: rows_by_id[sample_id].get(field, "") for field in REVIEW_FIELDS}
            writer.writerow(row)


def normalize_points(points: np.ndarray) -> np.ndarray:
    xyz = points[:, :3].astype(np.float32, copy=False)
    center = xyz.mean(axis=0)
    shifted = xyz - center
    scale = float(np.linalg.norm(shifted, axis=1).max())
    if scale <= 1e-12:
        scale = 1.0
    return shifted / scale


def sample_arrays(points: np.ndarray, masks: np.ndarray, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[0] <= max_points:
        return points, masks
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(points.shape[0], size=max_points, replace=False))
    return points[indices], masks[indices]


def compact_points(points: np.ndarray) -> list[list[float]]:
    return np.round(points.astype(np.float64), 5).tolist()


class ReviewServer:
    def __init__(self, dataset_root: Path, samples_path: Path, review_path: Path, max_points: int):
        self.dataset_root = dataset_root
        self.samples_path = samples_path
        self.review_path = review_path
        self.max_points = max_points
        self.reload()

    def reload(self) -> None:
        self.samples = read_samples(self.samples_path)
        self.samples_by_id = {sample["sample_id"]: sample for sample in self.samples}
        self.review_rows = read_review_csv(self.review_path, self.samples)

    def samples_payload(self) -> dict[str, Any]:
        rows = []
        for sample in self.samples:
            sample_id = sample["sample_id"]
            review = self.review_rows.get(sample_id, {})
            rows.append(
                {
                    "sample_id": sample_id,
                    "object_category": sample.get("object_category", ""),
                    "task": sample.get("task", ""),
                    "review_status": review.get("review_status", "pending"),
                    "keep_sample": review.get("keep_sample", ""),
                    "quality_after_review": review.get("quality_after_review", ""),
                }
            )
        return {"samples": rows, "count": len(rows)}

    def sample_payload(self, sample_id: str) -> dict[str, Any]:
        if sample_id not in self.samples_by_id:
            raise KeyError(f"Unknown sample_id: {sample_id}")
        sample = self.samples_by_id[sample_id]
        points_path = resolve_path(self.dataset_root, sample["point_cloud_path"])
        mask_path = resolve_path(self.dataset_root, sample["multi_channel_mask_path"])
        points = np.load(points_path, allow_pickle=False)
        masks = np.load(mask_path, allow_pickle=False)
        if points.ndim != 2 or points.shape[1] not in (3, 6):
            raise ValueError(f"Invalid point shape for {sample_id}: {points.shape}")
        if masks.ndim != 2 or masks.shape != (points.shape[0], len(EXECUTOR_ORDER)):
            raise ValueError(f"Invalid mask shape for {sample_id}: {masks.shape}")
        masks = (masks > 0).astype(np.uint8)
        seed = abs(hash(sample_id)) % (2**32)
        points, masks = sample_arrays(points, masks, self.max_points, seed)
        points = normalize_points(points)
        counts = {executor: int(masks[:, index].sum()) for index, executor in enumerate(EXECUTOR_ORDER)}
        return {
            "sample": sample,
            "review": self.review_rows.get(sample_id, initial_review_row(sample)),
            "executors": EXECUTOR_ORDER,
            "points": compact_points(points),
            "masks": masks.astype(np.uint8).tolist(),
            "counts": counts,
        }

    def save_review(self, payload: dict[str, Any]) -> dict[str, Any]:
        sample_id = str(payload.get("sample_id", ""))
        if sample_id not in self.samples_by_id:
            raise KeyError(f"Unknown sample_id: {sample_id}")
        existing = self.review_rows.get(sample_id, initial_review_row(self.samples_by_id[sample_id]))
        updated = dict(existing)
        for field in REVIEW_FIELDS:
            if field in payload:
                value = payload[field]
                updated[field] = "" if value is None else str(value)
        # Identity/path columns stay authoritative from metadata.
        base = initial_review_row(self.samples_by_id[sample_id])
        for field in (
            "sample_id",
            "object_id",
            "object_category",
            "task",
            "visualization_html_path",
            "point_cloud_path",
            "multi_channel_mask_path",
        ):
            updated[field] = base[field]
        self.review_rows[sample_id] = updated
        write_review_csv(self.review_path, self.review_rows)
        return {"ok": True, "sample_id": sample_id, "review": updated}


APP_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Multi-EE 人工审查</title>
  <style>
    :root { color-scheme: light; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Arial, "Microsoft YaHei", sans-serif; background: #f4f6fa; color: #202635; }
    header { height: 50px; display: flex; align-items: center; justify-content: space-between; padding: 0 16px; background: #18202e; color: white; }
    header h1 { margin: 0; font-size: 17px; }
    header .status { font-size: 13px; color: #cbd5e1; }
    .app { display: grid; grid-template-columns: 330px minmax(500px, 1fr) 420px; height: calc(100vh - 50px); }
    aside, section.form { overflow: auto; background: white; border-right: 1px solid #dce2ec; }
    aside { padding: 12px; }
    section.viewer { position: relative; overflow: hidden; background: #fafbfe; }
    section.form { border-right: 0; border-left: 1px solid #dce2ec; padding: 14px; }
    .filters { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 10px; }
    input, select, textarea, button { font-family: inherit; font-size: 13px; }
    input, select, textarea { width: 100%; border: 1px solid #cbd3df; border-radius: 6px; padding: 7px 8px; background: white; }
    textarea { min-height: 56px; resize: vertical; }
    button { border: 1px solid #bf2f2f; background: #cf3b3b; color: white; border-radius: 6px; padding: 8px 10px; cursor: pointer; }
    button.secondary { background: #eef2f7; color: #263043; border-color: #cbd3df; }
    .sample-list { display: flex; flex-direction: column; gap: 6px; }
    .sample-card { border: 1px solid #dce2ec; border-radius: 7px; padding: 8px; cursor: pointer; background: #fff; }
    .sample-card.active { border-color: #cf3b3b; box-shadow: 0 0 0 2px rgba(207, 59, 59, 0.12); }
    .sample-id { font-size: 11px; color: #526070; word-break: break-all; }
    .sample-meta { margin-top: 5px; display: flex; gap: 6px; flex-wrap: wrap; }
    .tag { font-size: 11px; border-radius: 999px; background: #edf1f7; color: #334155; padding: 2px 7px; }
    .tag.pending { background: #fff7d6; }
    .tag.checked { background: #dcfce7; }
    .tag.needs_fix { background: #ffedd5; }
    .tag.reject { background: #fee2e2; }
    .toolbar { position: absolute; left: 12px; right: 12px; top: 12px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; z-index: 2; }
    .toolbar button { background: #263043; border-color: #3b4658; }
    .toolbar button.active { background: #cf3b3b; border-color: #cf3b3b; }
    canvas { width: 100%; height: 100%; display: block; cursor: grab; }
    canvas:active { cursor: grabbing; }
    .hud { position: absolute; left: 14px; bottom: 12px; background: rgba(24, 32, 46, 0.78); color: white; padding: 8px 10px; border-radius: 6px; font-size: 12px; }
    .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .field { margin-bottom: 10px; }
    .field label { display: block; font-size: 12px; color: #526070; margin-bottom: 4px; }
    .help { margin-top: 4px; color: #6b7688; font-size: 12px; line-height: 1.45; }
    .explain { border: 1px solid #dce2ec; background: #f8fafc; border-radius: 8px; padding: 10px 11px; margin: 10px 0 12px; color: #445064; font-size: 12px; line-height: 1.55; }
    .explain b { color: #202635; }
    .executor-block { border: 1px solid #dce2ec; border-radius: 8px; padding: 10px; margin: 12px 0; }
    .executor-block h3 { margin: 0 0 8px; font-size: 14px; }
    .executor-title-note { color: #6b7688; font-size: 12px; line-height: 1.45; margin: -3px 0 8px; }
    .save-row { position: sticky; bottom: -14px; background: white; border-top: 1px solid #dce2ec; padding: 12px 0 0; display: flex; gap: 8px; }
    .message { min-height: 18px; font-size: 12px; color: #0f766e; margin-top: 8px; }
  </style>
</head>
<body>
  <header>
    <h1>Multi-EE Affordance 人工审查</h1>
    <div class="status" id="topStatus">加载中...</div>
  </header>
  <div class="app">
    <aside>
      <div class="filters">
        <select id="taskFilter">
          <option value="">全部任务</option>
          <option value="pick_up">pick_up</option>
          <option value="lift_carry">lift_carry</option>
          <option value="open_pull">open_pull</option>
          <option value="press_push">press_push</option>
        </select>
        <select id="statusFilter">
          <option value="">全部状态</option>
          <option value="pending">pending</option>
          <option value="checked">checked</option>
          <option value="needs_fix">needs_fix</option>
          <option value="reject">reject</option>
        </select>
      </div>
      <input id="searchBox" placeholder="搜索 sample/category">
      <div style="height:10px"></div>
      <div class="sample-list" id="sampleList"></div>
    </aside>
    <section class="viewer">
      <div class="toolbar" id="channelButtons"></div>
      <canvas id="canvas"></canvas>
      <div class="hud" id="hud">请选择样本</div>
    </section>
    <section class="form">
      <h2 style="margin:0 0 10px;font-size:17px">审查表单</h2>
      <div class="explain">
        <b>使用说明：</b>先在中间视图切换 raw 和四个执行器通道，判断当前物体-任务样本是否应该保留，再分别判断每个执行器通道是否合理。点击保存后会写回 CSV。
      </div>
      <div class="field">
        <label>sample_id</label>
        <input id="sample_id" disabled>
        <div class="help">当前审查样本的唯一 ID，不需要修改。</div>
      </div>
      <div class="form-grid">
        <div class="field"><label>object_category</label><input id="object_category" disabled><div class="help">物体类别，用于判断任务是否匹配。</div></div>
        <div class="field"><label>task</label><input id="task" disabled><div class="help">当前任务：抓取、搬运、拉开或按压。</div></div>
      </div>
      <div class="explain">
        <b>样本级判断：</b>回答“这个物体和任务组合是否应该进入数据集”。如果整个样本不合理，填 `reject` 和 `keep_sample=no`；如果样本可用但某些通道要修，填 `needs_fix`。
      </div>
      <div class="form-grid">
        <div class="field"><label>review_status</label><select id="review_status"></select><div class="help">审查状态：未审、已检查、需要修正或拒绝。</div></div>
        <div class="field"><label>keep_sample</label><select id="keep_sample"></select><div class="help">是否保留该样本：yes/no/maybe。</div></div>
      </div>
      <div class="form-grid">
        <div class="field"><label>quality_after_review</label><select id="quality_after_review"></select><div class="help">审查后的质量：weak、checked 或 verified。</div></div>
        <div class="field"><label>sample_issue_type</label><select id="sample_issue_type"></select><div class="help">样本级问题类型，例如任务不匹配、几何异常或需要 PartNet 补充。</div></div>
      </div>
      <div class="field"><label>sample_notes</label><textarea id="sample_notes" placeholder="例如：suction 区域过大；hook 需要 PartNet-Mobility 的孔洞/把手结构验证。"></textarea><div class="help">写给后续规则修正或人工精修看的备注。</div></div>
      <div class="explain">
        <b>执行器级判断：</b>分别回答“这个执行器通道是否合理”。`keep` 表示保留；`disable` 表示该通道应清零并设为不可行；`refine` 表示有一定道理但区域需要修；`add_missing` 表示当前漏标。
      </div>
      <div id="executorForms"></div>
      <div class="save-row">
        <button id="saveBtn">保存当前样本</button>
        <button class="secondary" id="saveNextBtn">保存并下一个</button>
      </div>
      <div class="message" id="message"></div>
    </section>
  </div>
<script>
const EXECUTORS = ["gripper", "suction", "hook", "dexterous_hand"];
const SELECTS = {
  review_status: [
    ["pending", "pending - 尚未审查"],
    ["checked", "checked - 已检查，基本可用"],
    ["needs_fix", "needs_fix - 需要修正规则或 mask"],
    ["reject", "reject - 样本不适合，建议剔除"]
  ],
  keep_sample: [
    ["", "(空) - 暂未决定"],
    ["yes", "yes - 保留样本"],
    ["no", "no - 删除或排除样本"],
    ["maybe", "maybe - 需要二次确认"]
  ],
  quality_after_review: [
    ["", "(空) - 暂未决定"],
    ["weak", "weak - 仍是弱标签"],
    ["checked", "checked - 已人工检查"],
    ["verified", "verified - 可作为高质量样本"]
  ],
  sample_issue_type: [
    ["", "(空) - 暂无记录"],
    ["none", "none - 无明显样本级问题"],
    ["task_mismatch", "task_mismatch - 物体和任务不匹配"],
    ["bad_geometry", "bad_geometry - 点云形状异常或缺失严重"],
    ["ambiguous_object", "ambiguous_object - 类别或结构不清晰"],
    ["all_masks_bad", "all_masks_bad - 四个通道都明显不合理"],
    ["needs_partnet", "needs_partnet - 需要 PartNet-Mobility 部件补充"]
  ],
  decision: [
    ["", "(空) - 暂未决定"],
    ["keep", "keep - 当前通道基本合理"],
    ["disable", "disable - 该执行器应不可行，通道应清零"],
    ["refine", "refine - 区域需要修正或精修"],
    ["add_missing", "add_missing - 应该可行但当前漏标"],
    ["not_applicable", "not_applicable - 当前任务下不适用"],
    ["uncertain", "uncertain - 暂时不确定"]
  ],
  issue: [
    ["", "(空) - 暂无记录"],
    ["none", "none - 无明显问题"],
    ["over_label", "over_label - 标得过大"],
    ["under_label", "under_label - 标得过小"],
    ["wrong_region", "wrong_region - 标到了错误位置"],
    ["task_mismatch", "task_mismatch - 对当前任务不合理"],
    ["executor_mismatch", "executor_mismatch - 对该执行器不合理"],
    ["too_noisy", "too_noisy - mask 噪声太多"],
    ["missing_positive", "missing_positive - 应有正样本但为空"],
    ["needs_geometry_rule", "needs_geometry_rule - 需要几何规则补充"],
    ["needs_part_annotation", "needs_part_annotation - 需要部件标注补充"]
  ],
};
const EXECUTOR_HELP = {
  gripper: "两指夹爪：重点看把手外侧、细长柄、可夹持边缘；大平面中心通常不应作为 gripper 正样本。",
  suction: "吸盘：重点看平整、低曲率、大面积表面；孔洞、边缘、细杆、把手通常不应作为 suction 正样本。",
  hook: "钩爪：重点看内孔、拉环、可挂接边界；如果只是普通平面或无孔把手，通常应 disable 或 refine。",
  dexterous_hand: "灵巧手：只标当前任务下适合多指稳定抓握、按压或精细操作的区域，不要把所有可接触表面都标为正。"
};
let samples = [];
let current = null;
let currentIndex = 0;
let channel = "raw";
let rotX = -0.55, rotY = 0.65, zoom = 1.0;
let dragging = false, lastX = 0, lastY = 0;
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

function fillSelect(id, values) {
  const el = document.getElementById(id);
  el.innerHTML = "";
  values.forEach(item => {
    const v = Array.isArray(item) ? item[0] : item;
    const label = Array.isArray(item) ? item[1] : (item || "(空)");
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = label;
    el.appendChild(opt);
  });
}

function initForm() {
  fillSelect("review_status", SELECTS.review_status);
  fillSelect("keep_sample", SELECTS.keep_sample);
  fillSelect("quality_after_review", SELECTS.quality_after_review);
  fillSelect("sample_issue_type", SELECTS.sample_issue_type);
  const box = document.getElementById("executorForms");
  box.innerHTML = "";
  EXECUTORS.forEach(ex => {
    const div = document.createElement("div");
    div.className = "executor-block";
    div.innerHTML = `
      <h3>${ex}</h3>
      <div class="executor-title-note">${EXECUTOR_HELP[ex]}</div>
      <div class="form-grid">
        <div class="field"><label>${ex}_decision</label><select id="${ex}_decision"></select><div class="help">对该执行器通道的处理决定。</div></div>
        <div class="field"><label>${ex}_issue_type</label><select id="${ex}_issue_type"></select><div class="help">该通道的问题类型；没有问题可选 none。</div></div>
      </div>
      <div class="field"><label>${ex}_notes</label><textarea id="${ex}_notes" placeholder="例如：区域过大；没有可挂接孔洞；按钮区域漏标。"></textarea><div class="help">记录你的判断依据，后续会用于自动修正或人工精修。</div></div>
    `;
    box.appendChild(div);
    fillSelect(`${ex}_decision`, SELECTS.decision);
    fillSelect(`${ex}_issue_type`, SELECTS.issue);
  });
}

function renderList() {
  const task = document.getElementById("taskFilter").value;
  const status = document.getElementById("statusFilter").value;
  const search = document.getElementById("searchBox").value.toLowerCase();
  const list = document.getElementById("sampleList");
  list.innerHTML = "";
  const filtered = samples.filter(s => {
    if (task && s.task !== task) return false;
    if (status && s.review_status !== status) return false;
    const hay = `${s.sample_id} ${s.object_category} ${s.task}`.toLowerCase();
    return !search || hay.includes(search);
  });
  document.getElementById("topStatus").textContent = `${samples.length} samples | 当前显示 ${filtered.length}`;
  filtered.forEach(s => {
    const card = document.createElement("div");
    card.className = "sample-card" + (current && current.sample.sample_id === s.sample_id ? " active" : "");
    card.onclick = () => loadSample(s.sample_id);
    card.innerHTML = `
      <div class="sample-id">${s.sample_id}</div>
      <div class="sample-meta">
        <span class="tag">${s.object_category}</span>
        <span class="tag">${s.task}</span>
        <span class="tag ${s.review_status || "pending"}">${s.review_status || "pending"}</span>
        ${s.keep_sample ? `<span class="tag">keep=${s.keep_sample}</span>` : ""}
      </div>
    `;
    list.appendChild(card);
  });
}

function setupChannels() {
  const box = document.getElementById("channelButtons");
  box.innerHTML = "";
  ["raw", ...EXECUTORS].forEach(ch => {
    const b = document.createElement("button");
    b.textContent = ch;
    b.onclick = () => { channel = ch; setupChannels(); draw(); };
    if (channel === ch) b.classList.add("active");
    box.appendChild(b);
  });
}

async function loadSamples() {
  const res = await fetch("/api/samples");
  const data = await res.json();
  samples = data.samples;
  renderList();
  if (samples.length) await loadSample(samples[0].sample_id);
}

async function loadSample(sampleId) {
  const res = await fetch(`/api/sample?id=${encodeURIComponent(sampleId)}`);
  if (!res.ok) throw new Error(await res.text());
  current = await res.json();
  currentIndex = samples.findIndex(s => s.sample_id === sampleId);
  channel = "raw";
  setupChannels();
  fillForm(current);
  renderList();
  resize();
}

function setValue(id, value) {
  const el = document.getElementById(id);
  if (el) el.value = value ?? "";
}

function fillForm(data) {
  const s = data.sample;
  const r = data.review || {};
  setValue("sample_id", s.sample_id);
  setValue("object_category", s.object_category);
  setValue("task", s.task);
  ["review_status", "keep_sample", "quality_after_review", "sample_issue_type", "sample_notes"].forEach(f => setValue(f, r[f]));
  EXECUTORS.forEach(ex => {
    setValue(`${ex}_decision`, r[`${ex}_decision`]);
    setValue(`${ex}_issue_type`, r[`${ex}_issue_type`]);
    setValue(`${ex}_notes`, r[`${ex}_notes`]);
  });
}

function collectForm() {
  const payload = { sample_id: document.getElementById("sample_id").value };
  ["review_status", "keep_sample", "quality_after_review", "sample_issue_type", "sample_notes"].forEach(f => payload[f] = document.getElementById(f).value);
  EXECUTORS.forEach(ex => {
    payload[`${ex}_decision`] = document.getElementById(`${ex}_decision`).value;
    payload[`${ex}_issue_type`] = document.getElementById(`${ex}_issue_type`).value;
    payload[`${ex}_notes`] = document.getElementById(`${ex}_notes`).value;
  });
  return payload;
}

async function saveReview(next=false) {
  const res = await fetch("/api/review", { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(collectForm()) });
  const data = await res.json();
  if (!res.ok || !data.ok) throw new Error(data.error || "save failed");
  document.getElementById("message").textContent = `已保存：${data.sample_id}`;
  await loadSamples();
  if (next && currentIndex + 1 < samples.length) await loadSample(samples[currentIndex + 1].sample_id);
  else await loadSample(data.sample_id);
}

function project(p) {
  const sx = Math.sin(rotX), cx = Math.cos(rotX);
  const sy = Math.sin(rotY), cy = Math.cos(rotY);
  let x = p[0], y = p[1], z = p[2];
  let x1 = cy * x + sy * z;
  let z1 = -sy * x + cy * z;
  let y1 = cx * y - sx * z1;
  let z2 = sx * y + cx * z1;
  const scale = Math.min(canvas.clientWidth, canvas.clientHeight) * 0.42 * zoom;
  return [canvas.clientWidth / 2 + x1 * scale, canvas.clientHeight / 2 - y1 * scale, z2];
}

function pointColor(row) {
  if (channel === "raw") return "#7b8492";
  const idx = EXECUTORS.indexOf(channel);
  return row[idx] ? "#d83c3c" : "#bfc7d2";
}

function draw() {
  if (!current) return;
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  const rows = current.points.map((p, i) => {
    const q = project(p);
    return {x: q[0], y: q[1], z: q[2], mask: current.masks[i]};
  }).sort((a,b) => a.z - b.z);
  const radius = Math.max(1.2, Math.min(3.0, 8000 / Math.max(2200, rows.length)));
  rows.forEach(row => {
    const c = pointColor(row.mask);
    ctx.fillStyle = c;
    ctx.globalAlpha = channel === "raw" ? 0.92 : (c === "#d83c3c" ? 0.98 : 0.34);
    ctx.beginPath();
    ctx.arc(row.x, row.y, radius, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.globalAlpha = 1;
  const count = channel === "raw" ? current.points.length : current.counts[channel];
  document.getElementById("hud").textContent = `${current.sample.sample_id} | ${channel} | positive=${count}`;
}

function resize() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}

canvas.addEventListener("mousedown", e => { dragging = true; lastX = e.clientX; lastY = e.clientY; });
window.addEventListener("mouseup", () => { dragging = false; });
window.addEventListener("mousemove", e => {
  if (!dragging) return;
  rotY += (e.clientX - lastX) * 0.008;
  rotX += (e.clientY - lastY) * 0.008;
  lastX = e.clientX;
  lastY = e.clientY;
  draw();
});
canvas.addEventListener("wheel", e => {
  e.preventDefault();
  zoom *= e.deltaY > 0 ? 0.9 : 1.1;
  zoom = Math.max(0.25, Math.min(5, zoom));
  draw();
}, {passive:false});
window.addEventListener("resize", resize);
["taskFilter", "statusFilter", "searchBox"].forEach(id => document.getElementById(id).addEventListener("input", renderList));
document.getElementById("saveBtn").onclick = () => saveReview(false).catch(err => alert(err.message));
document.getElementById("saveNextBtn").onclick = () => saveReview(true).catch(err => alert(err.message));
initForm();
setupChannels();
loadSamples().catch(err => alert(err.message));
</script>
</body>
</html>
"""


def make_handler(server_state: ReviewServer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            print(f"{self.address_string()} - {format % args}")

        def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def send_text(self, text: str, content_type: str = "text/html; charset=utf-8") -> None:
            data = text.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def send_error_json(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
            self.send_json({"ok": False, "error": message}, status)

        def do_GET(self) -> None:
            try:
                parsed = urlparse(self.path)
                if parsed.path == "/":
                    self.send_text(APP_HTML)
                elif parsed.path == "/api/samples":
                    server_state.reload()
                    self.send_json(server_state.samples_payload())
                elif parsed.path == "/api/sample":
                    query = parse_qs(parsed.query)
                    sample_id = query.get("id", [""])[0]
                    self.send_json(server_state.sample_payload(sample_id))
                else:
                    self.send_error_json(f"Not found: {parsed.path}", HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self.send_error_json(str(exc), HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_POST(self) -> None:
            try:
                parsed = urlparse(self.path)
                if parsed.path != "/api/review":
                    self.send_error_json(f"Not found: {parsed.path}", HTTPStatus.NOT_FOUND)
                    return
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8")
                payload = json.loads(body)
                result = server_state.save_review(payload)
                self.send_json(result)
            except Exception as exc:
                self.send_error_json(str(exc), HTTPStatus.INTERNAL_SERVER_ERROR)

    return Handler


def main() -> int:
    args = parse_args()
    try:
        dataset_root = Path(args.dataset_root).resolve()
        samples_path = resolve_path(dataset_root, args.samples)
        review_path = resolve_path(dataset_root, args.review_csv)
        state = ReviewServer(dataset_root, samples_path, review_path, args.max_points)
        httpd = ThreadingHTTPServer((args.host, args.port), make_handler(state))
        print(f"Manual review app: http://{args.host}:{args.port}/")
        print(f"Dataset root: {dataset_root}")
        print(f"Review CSV: {review_path}")
        httpd.serve_forever()
        return 0
    except KeyboardInterrupt:
        print("Stopped")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
