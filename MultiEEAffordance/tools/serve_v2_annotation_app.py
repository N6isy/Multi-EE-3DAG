#!/usr/bin/env python3
"""Serve a point-level v2 candidate annotation app.

This app is a local/deployable MVP for reviewing v2 candidate masks. It loads
candidate samples, displays the target executor channel, lets a reviewer add or
delete individual points by clicking the rendered point cloud, and saves a new
manual-refined mask plus an audit record.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np


EXECUTOR_ORDER = ["gripper", "suction", "hook", "dexterous_hand"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve v2 point-level annotation app.")
    parser.add_argument("--dataset-root", default=".", help="Dataset root.")
    parser.add_argument(
        "--samples",
        default="processed/metadata/v2_candidate_samples_v0_1.jsonl",
        help="Candidate sample JSONL relative to dataset root.",
    )
    parser.add_argument(
        "--review-jsonl",
        default="processed/metadata/v2_point_level_review_records.jsonl",
        help="Audit log JSONL relative to dataset root.",
    )
    parser.add_argument(
        "--output-mask-root",
        default="processed/vlm_candidate_v2/manual_refined_masks",
        help="Directory for manual-refined masks relative to dataset root.",
    )
    parser.add_argument(
        "--output-samples",
        default="processed/metadata/v2_manual_refined_samples_v0_1.jsonl",
        help="Output JSONL with latest refined sample metadata.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8770, help="Port to bind.")
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="0 means send all points. Point-level saving is safest with all points.",
    )
    parser.add_argument(
        "--allow-partial-save",
        action="store_true",
        help="Allow saving when max-points downsampling hides some original points.",
    )
    parser.add_argument(
        "--top-k-candidates",
        type=int,
        default=8,
        help="Number of ranked candidates shown in the review UI. Use 0 to show all.",
    )
    return parser.parse_args()


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def resolve_portable_path(root: Path, value: str | Path | None, base_dir: Path | None = None) -> Path:
    if value in (None, ""):
        return Path("")
    raw = str(value).replace("\\", "/")
    path = Path(raw)
    if path.is_absolute():
        return path
    if base_dir is not None:
        candidate = base_dir / path
        if candidate.exists():
            return candidate
    return root / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL not found: {path}")
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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        f.write("\n")


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


def safe_name(value: str) -> str:
    out = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "sample"


class AnnotationStore:
    def __init__(
        self,
        dataset_root: Path,
        samples_path: Path,
        review_path: Path,
        output_mask_root: Path,
        output_samples_path: Path,
        max_points: int,
        allow_partial_save: bool,
        top_k_candidates: int,
    ):
        self.dataset_root = dataset_root
        self.samples_path = samples_path
        self.review_path = review_path
        self.output_mask_root = output_mask_root
        self.output_samples_path = output_samples_path
        self.max_points = int(max_points)
        self.allow_partial_save = allow_partial_save
        self.top_k_candidates = int(top_k_candidates)
        self.reload()

    def reload(self) -> None:
        self.samples = read_jsonl(self.samples_path)
        self.samples_by_id = {str(row["sample_id"]): row for row in self.samples}
        self.refined_by_id: dict[str, dict[str, Any]] = {}
        if self.output_samples_path.exists():
            for row in read_jsonl(self.output_samples_path):
                self.refined_by_id[str(row.get("sample_id", ""))] = row

    def list_samples(self) -> dict[str, Any]:
        rows = []
        for sample in self.samples:
            sample_id = str(sample["sample_id"])
            update = sample.get("v2_candidate_update", {})
            refined = self.refined_by_id.get(sample_id, {})
            rows.append(
                {
                    "sample_id": sample_id,
                    "object_category": sample.get("object_category", ""),
                    "task": sample.get("task", ""),
                    "executor": update.get("executor", sample.get("executor", "")),
                    "selected_candidates": update.get("selected_candidates", []),
                    "positive_points": update.get("positive_points", ""),
                    "review_status": refined.get("point_review_status", "pending"),
                    "reviewer": refined.get("point_review_reviewer", ""),
                    "quality_flag": refined.get("quality_flag", sample.get("quality_flag", "")),
                }
            )
        return {"samples": rows, "count": len(rows)}

    def load_candidate_context(self, sample: dict[str, Any], visible_indices: np.ndarray) -> dict[str, Any]:
        update = sample.get("v2_candidate_update", {})
        manifest_value = update.get("candidate_manifest", "")
        if not manifest_value:
            return {"available": False, "candidates": [], "default_selected_candidates": update.get("selected_candidates", [])}
        manifest_path = resolve_portable_path(self.dataset_root, manifest_value)
        if not manifest_path.exists():
            return {
                "available": False,
                "error": f"candidate_manifest not found: {manifest_value}",
                "candidates": [],
                "default_selected_candidates": update.get("selected_candidates", []),
            }
        manifest = read_json(manifest_path)
        npz_path = resolve_portable_path(self.dataset_root, manifest.get("candidate_npz", ""), manifest_path.parent)
        if not npz_path.exists():
            return {
                "available": False,
                "error": f"candidate_npz not found: {manifest.get('candidate_npz', '')}",
                "candidates": [],
                "default_selected_candidates": update.get("selected_candidates", []),
            }
        data = np.load(npz_path, allow_pickle=True)
        candidate_ids = [str(item).upper() for item in data["candidate_ids"].tolist()]
        candidate_masks = data["candidate_masks"].astype(np.uint8)
        rule_value = update.get("rule_filter_path", "")
        rule: dict[str, Any] = {}
        rule_path = resolve_portable_path(self.dataset_root, rule_value) if rule_value else Path("")
        if rule_path and rule_path.exists():
            rule = read_json(rule_path)
        score_by_id: dict[str, dict[str, Any]] = {}
        for item in rule.get("candidate_scores", []):
            if isinstance(item, dict):
                cid = str(item.get("candidate_id", "")).upper()
                if cid:
                    score_by_id[cid] = item
        accepted = {str(item).upper() for item in rule.get("accepted_candidates", [])}
        uncertain = {str(item).upper() for item in rule.get("uncertain_candidates", [])}
        default_selected = [str(item).upper() for item in update.get("selected_candidates", [])]
        default_selected_set = set(default_selected)

        candidates: list[dict[str, Any]] = []
        records = {str(item.get("candidate_id", "")).upper(): item for item in manifest.get("candidates", []) if item.get("candidate_id")}
        for idx, cid in enumerate(candidate_ids):
            record = records.get(cid, {})
            score = score_by_id.get(cid, {})
            visible_mask = candidate_masks[idx][visible_indices].astype(bool)
            visible_point_indices = visible_indices[visible_mask].astype(int).tolist()
            selected_votes = int(score.get("selected_votes", 0) or 0)
            uncertain_votes = int(score.get("uncertain_votes", 0) or 0)
            rule_score = float(score.get("rule_score", 0.0) or 0.0)
            decision = str(score.get("decision", ""))
            if cid in default_selected_set:
                rank_bucket = 0
            elif cid in accepted:
                rank_bucket = 1
            elif selected_votes > 0:
                rank_bucket = 2
            elif cid in uncertain or uncertain_votes > 0:
                rank_bucket = 3
            else:
                rank_bucket = 4
            candidates.append(
                {
                    "candidate_id": cid,
                    "candidate_name": record.get("candidate_name", score.get("candidate_name", "")),
                    "candidate_family": record.get("candidate_family", score.get("candidate_family", "")),
                    "description": record.get("description", ""),
                    "point_count": int(record.get("point_count", score.get("point_count", len(visible_point_indices))) or 0),
                    "visible_point_count": len(visible_point_indices),
                    "point_indices": visible_point_indices,
                    "selected_votes": selected_votes,
                    "uncertain_votes": uncertain_votes,
                    "rule_score": rule_score,
                    "decision": decision,
                    "auto_status": "selected" if cid in default_selected_set else ("accepted" if cid in accepted else ("uncertain" if cid in uncertain else "candidate")),
                    "default_checked": (cid in default_selected_set) if default_selected_set else (cid in accepted),
                    "_rank": (rank_bucket, -selected_votes, -rule_score, cid),
                }
            )
        candidates.sort(key=lambda item: item["_rank"])
        for item in candidates:
            item.pop("_rank", None)
        if self.top_k_candidates > 0:
            pinned = [item for item in candidates if item["candidate_id"] in default_selected_set]
            rest = [item for item in candidates if item["candidate_id"] not in default_selected_set]
            candidates = (pinned + rest)[: self.top_k_candidates]
        return {
            "available": True,
            "candidate_manifest": str(manifest_path.relative_to(self.dataset_root).as_posix()),
            "rule_filter_path": str(rule_path.relative_to(self.dataset_root).as_posix()) if rule_path and rule_path.exists() else "",
            "default_selected_candidates": default_selected,
            "accepted_candidates": sorted(accepted),
            "uncertain_candidates": sorted(uncertain),
            "candidates": candidates,
            "notes": "Candidates are ranked proposals. Reviewers choose a subset, then refine points manually.",
        }

    def sample_payload(self, sample_id: str) -> dict[str, Any]:
        sample = self.refined_by_id.get(sample_id) or self.samples_by_id.get(sample_id)
        if sample is None:
            raise KeyError(f"Unknown sample_id: {sample_id}")
        points_path = resolve_path(self.dataset_root, sample["point_cloud_path"])
        mask_path = resolve_path(self.dataset_root, sample["multi_channel_mask_path"])
        points = np.load(points_path, allow_pickle=False)
        masks = np.load(mask_path, allow_pickle=False)
        if points.ndim != 2 or points.shape[1] not in (3, 6):
            raise ValueError(f"Invalid points shape for {sample_id}: {points.shape}")
        if masks.ndim != 2 or masks.shape != (points.shape[0], len(EXECUTOR_ORDER)):
            raise ValueError(f"Invalid mask shape for {sample_id}: {masks.shape}")
        update = sample.get("v2_candidate_update", {})
        executor = str(update.get("executor") or sample.get("executor") or "hook")
        if executor not in EXECUTOR_ORDER:
            executor = "hook"
        channel = EXECUTOR_ORDER.index(executor)
        seed = abs(hash(sample_id)) % (2**32)
        indices = choose_indices(points.shape[0], self.max_points, seed)
        visible_all_points = int(indices.size) == int(points.shape[0])
        normalized = normalize_points(points)
        visible_points = normalized[indices]
        visible_masks = (masks[indices] > 0).astype(np.uint8)
        candidate_context = self.load_candidate_context(sample, indices)
        return {
            "sample": sample,
            "executor_order": EXECUTOR_ORDER,
            "target_executor": executor,
            "target_channel": channel,
            "points": compact_points(visible_points),
            "point_indices": indices.astype(int).tolist(),
            "masks": visible_masks.tolist(),
            "visible_all_points": visible_all_points,
            "counts": {name: int((masks[:, idx] > 0).sum()) for idx, name in enumerate(EXECUTOR_ORDER)},
            "review_hint": {
                "selected_candidates": update.get("selected_candidates", []),
                "positive_points": update.get("positive_points", ""),
                "requires_human_review": update.get("requires_human_review", True),
                "candidate_manifest": update.get("candidate_manifest", ""),
                "rule_filter_path": update.get("rule_filter_path", ""),
            },
            "candidate_context": candidate_context,
        }

    def save_edit(self, payload: dict[str, Any]) -> dict[str, Any]:
        sample_id = str(payload.get("sample_id", ""))
        base_sample = self.refined_by_id.get(sample_id) or self.samples_by_id.get(sample_id)
        if base_sample is None:
            raise KeyError(f"Unknown sample_id: {sample_id}")
        executor = str(payload.get("executor") or base_sample.get("v2_candidate_update", {}).get("executor") or "hook")
        if executor not in EXECUTOR_ORDER:
            raise ValueError(f"Unknown executor: {executor}")
        visible_all_points = bool(payload.get("visible_all_points", False))
        if not visible_all_points and not self.allow_partial_save:
            raise ValueError("Refusing partial save: restart app with --max-points 0 or pass --allow-partial-save.")
        positive_indices_raw = payload.get("positive_indices", [])
        if not isinstance(positive_indices_raw, list):
            raise ValueError("positive_indices must be a list.")
        positive_indices = sorted({int(x) for x in positive_indices_raw if int(x) >= 0})
        mask_path = resolve_path(self.dataset_root, base_sample["multi_channel_mask_path"])
        masks = np.load(mask_path, allow_pickle=False).astype(np.uint8)
        n = masks.shape[0]
        positive_indices = [idx for idx in positive_indices if idx < n]
        channel = EXECUTOR_ORDER.index(executor)
        old_positive = set(np.where(masks[:, channel] > 0)[0].astype(int).tolist())
        new_positive = set(positive_indices)
        refined = masks.copy()
        refined[:, channel] = 0
        if positive_indices:
            refined[np.asarray(positive_indices, dtype=np.int64), channel] = 1
        output_mask_path = self.output_mask_root / f"{safe_name(sample_id)}_{executor}_manual_refined.npy"
        output_mask_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_mask_path, refined)

        now = datetime.now(timezone.utc).isoformat()
        refined_sample = dict(base_sample)
        refined_sample["multi_channel_mask_path"] = str(output_mask_path.relative_to(self.dataset_root).as_posix())
        refined_sample["quality_flag"] = str(payload.get("quality_after_review") or "checked")
        refined_sample["point_review_status"] = str(payload.get("review_status") or "checked")
        refined_sample["point_review_reviewer"] = str(payload.get("reviewer") or "")
        refined_sample["point_review_notes"] = str(payload.get("notes") or "")
        refined_sample["point_review_updated_at"] = now
        refined_sample["v2_point_edit"] = {
            "executor": executor,
            "source_mask_path": base_sample["multi_channel_mask_path"],
            "output_mask_path": refined_sample["multi_channel_mask_path"],
            "selected_candidate_ids": [str(item).upper() for item in payload.get("selected_candidate_ids", [])],
            "positive_points_before": len(old_positive),
            "positive_points_after": len(new_positive),
            "added_points": sorted(new_positive - old_positive),
            "removed_points": sorted(old_positive - new_positive),
            "review_decision": str(payload.get("review_decision") or ""),
            "reviewer": str(payload.get("reviewer") or ""),
            "updated_at": now,
        }
        self.refined_by_id[sample_id] = refined_sample
        ordered = []
        for sample in self.samples:
            sid = str(sample["sample_id"])
            if sid in self.refined_by_id:
                ordered.append(self.refined_by_id[sid])
        write_jsonl(self.output_samples_path, ordered)
        record = {
            "created_at": now,
            "sample_id": sample_id,
            "executor": executor,
            "reviewer": str(payload.get("reviewer") or ""),
            "review_status": str(payload.get("review_status") or ""),
            "review_decision": str(payload.get("review_decision") or ""),
            "quality_after_review": str(payload.get("quality_after_review") or ""),
            "notes": str(payload.get("notes") or ""),
            "positive_points_before": len(old_positive),
            "positive_points_after": len(new_positive),
            "selected_candidate_ids": [str(item).upper() for item in payload.get("selected_candidate_ids", [])],
            "added_points": sorted(new_positive - old_positive),
            "removed_points": sorted(old_positive - new_positive),
            "output_mask_path": refined_sample["multi_channel_mask_path"],
        }
        append_jsonl(self.review_path, record)
        return {"ok": True, "sample_id": sample_id, "record": record, "sample": refined_sample}


APP_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Multi-EE v2 点级审查系统</title>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; background: #f5f7fb; color: #202635; font-family: Arial, "Microsoft YaHei", sans-serif; }
    header { height: 52px; padding: 0 16px; background: #17202e; color: white; display: flex; align-items: center; justify-content: space-between; }
    header h1 { margin: 0; font-size: 17px; }
    .app { display: grid; grid-template-columns: 330px minmax(620px, 1fr) 380px; height: calc(100vh - 52px); }
    aside, .panel { overflow: auto; background: #fff; border-right: 1px solid #d8e0eb; }
    aside { padding: 12px; }
    .viewer { position: relative; background: #f9fbff; overflow: hidden; }
    .panel { border-right: 0; border-left: 1px solid #d8e0eb; padding: 14px; }
    input, select, textarea, button { font-family: inherit; font-size: 13px; }
    input, select, textarea { width: 100%; padding: 8px; border: 1px solid #c9d2df; border-radius: 6px; background: white; }
    textarea { min-height: 88px; resize: vertical; }
    button { border: 1px solid #c13d3d; background: #d54444; color: white; border-radius: 6px; padding: 8px 10px; cursor: pointer; }
    button.secondary { background: #eef2f7; color: #263043; border-color: #cbd3df; }
    button.active { background: #1f2a3d; border-color: #1f2a3d; }
    .sample-list { display: flex; flex-direction: column; gap: 7px; margin-top: 10px; }
    .sample { border: 1px solid #d8e0eb; border-radius: 8px; padding: 9px; cursor: pointer; }
    .sample.active { border-color: #d54444; box-shadow: 0 0 0 2px rgba(213,68,68,.12); }
    .sample-id { font-size: 11px; color: #526070; word-break: break-all; }
    .tags { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 6px; }
    .tag { font-size: 11px; padding: 2px 7px; border-radius: 999px; background: #edf1f7; color: #334155; }
    .tag.pending { background: #fff7d6; }
    .tag.checked { background: #dcfce7; }
    .tag.refine_needed { background: #ffedd5; }
    canvas { width: 100%; height: 100%; display: block; cursor: crosshair; }
    .toolbar { position: absolute; left: 12px; top: 12px; right: 12px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; z-index: 2; }
    .hud { position: absolute; left: 12px; bottom: 12px; background: rgba(23,32,46,.82); color: white; padding: 9px 11px; border-radius: 7px; font-size: 12px; line-height: 1.55; }
    .box { border: 1px solid #d8e0eb; background: #f8fafc; border-radius: 8px; padding: 10px; margin-bottom: 12px; font-size: 12px; line-height: 1.55; color: #475569; }
    .candidate-list { display: flex; flex-direction: column; gap: 7px; margin: 8px 0 10px; }
    .candidate-item { border: 1px solid #d8e0eb; border-radius: 8px; padding: 8px; background: #fff; }
    .candidate-item.selected { border-color: #d54444; background: #fff7f7; }
    .candidate-main { display: flex; gap: 8px; align-items: flex-start; }
    .candidate-swatch { width: 12px; height: 12px; border-radius: 3px; margin-top: 3px; flex: 0 0 auto; }
    .candidate-title { font-size: 12px; font-weight: 650; color: #202635; }
    .candidate-meta { font-size: 11px; color: #64748b; line-height: 1.45; margin-top: 2px; }
    .candidate-actions { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px; }
    .field { margin-bottom: 11px; }
    .field label { display: block; color: #526070; font-size: 12px; margin-bottom: 5px; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 9px; }
    .savebar { position: sticky; bottom: -14px; background: #fff; border-top: 1px solid #d8e0eb; padding-top: 12px; display: flex; gap: 8px; }
    .msg { margin-top: 8px; min-height: 18px; color: #0f766e; font-size: 12px; }
    code { color: #b42323; }
  </style>
</head>
<body>
<header>
  <h1>Multi-EE v2 点级审查系统</h1>
  <div id="topStatus">加载中...</div>
</header>
<div class="app">
  <aside>
    <input id="search" placeholder="搜索 sample/category/executor" />
    <div class="sample-list" id="sampleList"></div>
  </aside>
  <main class="viewer">
    <div class="toolbar">
      <button id="modeView">查看/旋转</button>
      <button id="modeToggle">点击切换点</button>
      <button id="modeAdd">只添加</button>
      <button id="modeDelete">只删除</button>
      <button class="secondary" id="resetView">重置视角</button>
      <button class="secondary" id="undoBtn">撤销</button>
    </div>
    <canvas id="canvas"></canvas>
    <div class="hud" id="hud">请选择样本</div>
  </main>
  <section class="panel">
    <h2 style="margin:0 0 10px;font-size:17px">点级审查</h2>
    <div class="box">
      <b>审查目标：</b>只编辑当前目标执行器通道。点击点云点可以删除误标点或补充漏标点；保存后会写出新的 refined mask 和审查记录。
    </div>
    <div class="field"><label>sample_id</label><input id="sampleId" disabled /></div>
    <div class="row">
      <div class="field"><label>object_category</label><input id="category" disabled /></div>
      <div class="field"><label>task</label><input id="task" disabled /></div>
    </div>
    <div class="row">
      <div class="field"><label>target_executor</label><input id="executor" disabled /></div>
      <div class="field"><label>positive count</label><input id="count" disabled /></div>
    </div>
    <div class="box" id="candidateHint"></div>
    <div class="box">
      <b>候选选择：</b>系统按 VLM 投票和规则过滤给出 top-k 候选。先勾选候选组合，再点击“应用勾选候选”，随后进行点级删除/补点。
      <label style="display:flex;align-items:center;gap:7px;margin-top:8px;color:#334155;">
        <input id="showCandidatePreview" type="checkbox" checked style="width:auto;"> 显示候选预览颜色
      </label>
      <div class="candidate-list" id="candidateList"></div>
      <div class="candidate-actions">
        <button class="secondary" id="applyCandidatesBtn">应用勾选候选</button>
        <button class="secondary" id="clearMaskBtn">清空当前 mask</button>
      </div>
    </div>
    <div class="field"><label>reviewer</label><input id="reviewer" placeholder="填写姓名或学号" /></div>
    <div class="row">
      <div class="field">
        <label>review_status</label>
        <select id="reviewStatus">
          <option value="checked">checked - 已点级检查</option>
          <option value="refine_needed">refine_needed - 仍需进一步精修</option>
          <option value="reject">reject - 候选不可用</option>
        </select>
      </div>
      <div class="field">
        <label>review_decision</label>
        <select id="reviewDecision">
          <option value="accept_refined">accept_refined - 保存当前精修结果</option>
          <option value="needs_more_candidates">needs_more_candidates - 需要加入更多候选</option>
          <option value="candidate_too_noisy">candidate_too_noisy - 当前候选太噪</option>
          <option value="uncertain">uncertain - 不确定</option>
        </select>
      </div>
    </div>
    <div class="field">
      <label>quality_after_review</label>
      <select id="quality">
        <option value="checked">checked - 已人工点级检查</option>
        <option value="weak">weak - 仍是弱标签</option>
        <option value="verified">verified - 高质量确认</option>
      </select>
    </div>
    <div class="field"><label>notes</label><textarea id="notes" placeholder="例如：A 有 4 个 false positive 已删除；E 覆盖 handle 但混入包体边缘。"></textarea></div>
    <div class="savebar">
      <button id="saveBtn">保存 refined mask</button>
      <button class="secondary" id="reloadBtn">重新加载</button>
    </div>
    <div class="msg" id="message"></div>
  </section>
</div>
<script>
let samples = [];
let current = null;
let currentIndex = -1;
let positives = new Set();
let initialPositives = new Set();
let candidateSets = new Map();
let candidateInfo = [];
let selectedCandidateIds = new Set();
let mode = "toggle";
let rotX = -0.55, rotY = 0.65, zoom = 1.0;
let dragging = false, lastX = 0, lastY = 0;
let history = [];
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const CANDIDATE_COLORS = ["#ff4848","#32dc78","#ffd240","#78a0ff","#ff70d2","#46e6eb","#ff9150","#be7dff","#aae65a","#ffffff","#ff7878","#78ffb4"];

function setMode(next) {
  mode = next;
  ["modeView","modeToggle","modeAdd","modeDelete"].forEach(id => document.getElementById(id).classList.remove("active"));
  const map = {view:"modeView", toggle:"modeToggle", add:"modeAdd", delete:"modeDelete"};
  document.getElementById(map[next]).classList.add("active");
  draw();
}

function renderList() {
  const q = document.getElementById("search").value.toLowerCase();
  const list = document.getElementById("sampleList");
  list.innerHTML = "";
  const filtered = samples.filter(s => `${s.sample_id} ${s.object_category} ${s.task} ${s.executor}`.toLowerCase().includes(q));
  document.getElementById("topStatus").textContent = `${samples.length} samples | 显示 ${filtered.length}`;
  filtered.forEach(s => {
    const div = document.createElement("div");
    div.className = "sample" + (current && current.sample.sample_id === s.sample_id ? " active" : "");
    div.onclick = () => loadSample(s.sample_id);
    div.innerHTML = `<div class="sample-id">${s.sample_id}</div>
      <div class="tags">
        <span class="tag">${s.object_category}</span>
        <span class="tag">${s.task}</span>
        <span class="tag">${s.executor}</span>
        <span class="tag ${s.review_status || "pending"}">${s.review_status || "pending"}</span>
        <span class="tag">pos=${s.positive_points}</span>
      </div>`;
    list.appendChild(div);
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
  positives = new Set();
  candidateSets = new Map();
  candidateInfo = (current.candidate_context && current.candidate_context.candidates) || [];
  selectedCandidateIds = new Set();
  candidateInfo.forEach(c => {
    candidateSets.set(c.candidate_id, new Set(c.point_indices || []));
    if (c.default_checked) selectedCandidateIds.add(c.candidate_id);
  });
  const ch = current.target_channel;
  current.point_indices.forEach((idx, i) => {
    if (current.masks[i][ch]) positives.add(idx);
  });
  initialPositives = new Set(positives);
  history = [];
  fillPanel();
  renderList();
  resize();
}

function fillPanel() {
  const s = current.sample;
  document.getElementById("sampleId").value = s.sample_id;
  document.getElementById("category").value = s.object_category || "";
  document.getElementById("task").value = s.task || "";
  document.getElementById("executor").value = current.target_executor;
  document.getElementById("count").value = positives.size;
  const hint = current.review_hint || {};
  document.getElementById("candidateHint").innerHTML =
    `<b>自动候选来源：</b><br/>
     selected_candidates: <code>${(hint.selected_candidates || []).join(",") || "(none)"}</code><br/>
     positive_points_before: <code>${hint.positive_points ?? ""}</code><br/>
     这个页面保存的是人工点级 refinement，不会把自动候选直接当 GT。`;
  renderCandidateList();
}

function candidateColor(cid) {
  const idx = candidateInfo.findIndex(c => c.candidate_id === cid);
  return CANDIDATE_COLORS[(idx < 0 ? 0 : idx) % CANDIDATE_COLORS.length];
}

function renderCandidateList() {
  const box = document.getElementById("candidateList");
  box.innerHTML = "";
  if (!candidateInfo.length) {
    box.innerHTML = `<div class="candidate-meta">没有可用候选上下文。可继续编辑当前 mask，但无法从 top-k 候选中勾选组合。</div>`;
    return;
  }
  candidateInfo.forEach(c => {
    const item = document.createElement("div");
    item.className = "candidate-item" + (selectedCandidateIds.has(c.candidate_id) ? " selected" : "");
    const color = candidateColor(c.candidate_id);
    item.innerHTML = `
      <div class="candidate-main">
        <input type="checkbox" style="width:auto;margin-top:1px" ${selectedCandidateIds.has(c.candidate_id) ? "checked" : ""}>
        <span class="candidate-swatch" style="background:${color}"></span>
        <div>
          <div class="candidate-title">${c.candidate_id} ${c.candidate_name || ""}</div>
          <div class="candidate-meta">
            ${c.candidate_family || ""} | status=${c.auto_status || "candidate"} | decision=${c.decision || ""}<br>
            votes=${c.selected_votes || 0}, uncertain=${c.uncertain_votes || 0}, rule=${Number(c.rule_score || 0).toFixed(2)}, points=${c.point_count || 0}
          </div>
        </div>
      </div>`;
    const checkbox = item.querySelector("input");
    checkbox.onchange = () => {
      if (checkbox.checked) selectedCandidateIds.add(c.candidate_id);
      else selectedCandidateIds.delete(c.candidate_id);
      renderCandidateList();
      draw();
    };
    box.appendChild(item);
  });
}

function candidateUnion(ids) {
  const out = new Set();
  ids.forEach(cid => {
    const s = candidateSets.get(cid);
    if (!s) return;
    s.forEach(idx => out.add(idx));
  });
  return out;
}

function applySelectedCandidates() {
  saveHistory();
  positives = candidateUnion(selectedCandidateIds);
  initialPositives = new Set(positives);
  draw();
}

function clearMask() {
  saveHistory();
  positives = new Set();
  draw();
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

function projectedPoints() {
  if (!current) return [];
  return current.points.map((p, i) => {
    const pr = project(p);
    return {x: pr[0], y: pr[1], z: pr[2], original: current.point_indices[i], i};
  });
}

function nearestPoint(x, y, positiveOnly=false) {
  let best = null;
  let bestD = Infinity;
  const pts = projectedPoints();
  for (const p of pts) {
    if (positiveOnly && !positives.has(p.original)) continue;
    const d = (p.x - x) ** 2 + (p.y - y) ** 2;
    if (d < bestD) { bestD = d; best = p; }
  }
  return bestD <= 18 * 18 ? best : null;
}

function previewColorForPoint(originalIndex) {
  for (const c of candidateInfo) {
    const s = candidateSets.get(c.candidate_id);
    if (s && s.has(originalIndex)) return candidateColor(c.candidate_id);
  }
  return null;
}

function saveHistory() {
  history.push(new Set(positives));
  if (history.length > 50) history.shift();
}

function draw() {
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  if (!current) return;
  const pts = projectedPoints().sort((a,b) => a.z - b.z);
  const showPreview = document.getElementById("showCandidatePreview").checked;
  for (const p of pts) {
    const on = positives.has(p.original);
    const preview = showPreview ? previewColorForPoint(p.original) : null;
    ctx.beginPath();
    ctx.fillStyle = on ? "#d83c3c" : (preview || "#aeb8c6");
    ctx.globalAlpha = on ? 0.96 : (preview ? 0.72 : 0.34);
    ctx.arc(p.x, p.y, on ? 4.4 : (preview ? 3.4 : 2.4), 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
  const added = [...positives].filter(x => !initialPositives.has(x)).length;
  const removed = [...initialPositives].filter(x => !positives.has(x)).length;
  document.getElementById("hud").innerHTML =
    `mode=${mode} | executor=${current.target_executor}<br/>checked_candidates=${[...selectedCandidateIds].join(",") || "(none)"}<br/>positive=${positives.size} | added=${added} | removed=${removed}<br/>拖拽旋转，滚轮缩放；点击按当前模式编辑`;
  document.getElementById("count").value = positives.size;
}

function resize() {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  draw();
}

canvas.addEventListener("mousedown", e => {
  dragging = true; lastX = e.clientX; lastY = e.clientY;
});
window.addEventListener("mouseup", e => {
  if (!dragging) return;
  const moved = Math.abs(e.clientX - lastX) + Math.abs(e.clientY - lastY);
  dragging = false;
  if (moved < 4 && mode !== "view" && current) {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left, y = e.clientY - rect.top;
    const p = nearestPoint(x, y, mode === "delete");
    if (!p) return;
    saveHistory();
    if (mode === "add") positives.add(p.original);
    else if (mode === "delete") positives.delete(p.original);
    else if (positives.has(p.original)) positives.delete(p.original);
    else positives.add(p.original);
    draw();
  }
});
window.addEventListener("mousemove", e => {
  if (!dragging || mode !== "view") return;
  rotY += (e.clientX - lastX) * 0.008;
  rotX += (e.clientY - lastY) * 0.008;
  lastX = e.clientX; lastY = e.clientY;
  draw();
});
canvas.addEventListener("wheel", e => {
  e.preventDefault();
  zoom *= e.deltaY < 0 ? 1.08 : 0.92;
  zoom = Math.max(0.25, Math.min(5, zoom));
  draw();
}, {passive:false});

async function saveEdit() {
  if (!current) return;
  const payload = {
    sample_id: current.sample.sample_id,
    executor: current.target_executor,
    selected_candidate_ids: [...selectedCandidateIds].sort(),
    positive_indices: [...positives].sort((a,b) => a-b),
    visible_all_points: current.visible_all_points,
    reviewer: document.getElementById("reviewer").value,
    review_status: document.getElementById("reviewStatus").value,
    review_decision: document.getElementById("reviewDecision").value,
    quality_after_review: document.getElementById("quality").value,
    notes: document.getElementById("notes").value,
  };
  const res = await fetch("/api/save_edit", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)});
  const data = await res.json();
  if (!res.ok || !data.ok) throw new Error(data.error || "save failed");
  document.getElementById("message").textContent =
    `已保存 refined mask：positive ${data.record.positive_points_before} -> ${data.record.positive_points_after}`;
  await loadSamples();
  await loadSample(data.sample_id);
}

document.getElementById("search").oninput = renderList;
document.getElementById("modeView").onclick = () => setMode("view");
document.getElementById("modeToggle").onclick = () => setMode("toggle");
document.getElementById("modeAdd").onclick = () => setMode("add");
document.getElementById("modeDelete").onclick = () => setMode("delete");
document.getElementById("resetView").onclick = () => { rotX = -0.55; rotY = 0.65; zoom = 1; draw(); };
document.getElementById("undoBtn").onclick = () => { if (history.length) { positives = history.pop(); draw(); } };
document.getElementById("applyCandidatesBtn").onclick = applySelectedCandidates;
document.getElementById("clearMaskBtn").onclick = clearMask;
document.getElementById("showCandidatePreview").onchange = draw;
document.getElementById("saveBtn").onclick = () => saveEdit().catch(err => alert(err.message));
document.getElementById("reloadBtn").onclick = () => current && loadSample(current.sample.sample_id).catch(err => alert(err.message));
window.addEventListener("resize", resize);
setMode("toggle");
loadSamples().catch(err => alert(err.message));
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    store: AnnotationStore

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.client_address[0], self.log_date_time_string(), fmt % args))

    def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_html(self) -> None:
        data = APP_HTML.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                self.send_html()
                return
            if parsed.path == "/api/samples":
                self.send_json(self.store.list_samples())
                return
            if parsed.path == "/api/sample":
                query = parse_qs(parsed.query)
                sample_id = query.get("id", [""])[0]
                self.send_json(self.store.sample_payload(sample_id))
                return
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            if self.path == "/api/save_edit":
                self.send_json(self.store.save_edit(payload))
                return
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    store = AnnotationStore(
        dataset_root=root,
        samples_path=resolve_path(root, args.samples),
        review_path=resolve_path(root, args.review_jsonl),
        output_mask_root=resolve_path(root, args.output_mask_root),
        output_samples_path=resolve_path(root, args.output_samples),
        max_points=args.max_points,
        allow_partial_save=args.allow_partial_save,
        top_k_candidates=args.top_k_candidates,
    )
    Handler.store = store
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving v2 annotation app at http://{args.host}:{args.port}")
    print(f"Samples: {resolve_path(root, args.samples)}")
    print(f"Review log: {resolve_path(root, args.review_jsonl)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
