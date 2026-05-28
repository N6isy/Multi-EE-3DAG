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
import traceback
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np


EXECUTOR_ORDER = ["gripper", "suction", "hook", "dexterous_hand"]
VALID_REVIEWERS = {"reviewer_a", "reviewer_b"}


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
        default=5,
        help="Number of ranked candidates shown in the review UI. Use 0 to show all.",
    )
    parser.add_argument(
        "--candidate-min-selected-votes",
        type=int,
        default=2,
        help="Only show non-default candidates with at least this many VLM selected votes. Use 0 to show low-confidence rule-only candidates.",
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


def sample_key(row: dict[str, Any]) -> str:
    """Build a stable row key.

    The annotation app accepts both legacy tasks
    (pick_up/open_pull/press_push/lift_carry) and the new five-task taxonomy
    (lift/open/pull/press/push).

    For expanded task samples, one legacy row may become multiple annotation rows.
    Therefore task is included in the fallback key to avoid collisions between
    open vs pull and press vs push.
    """
    explicit = str(row.get("row_key") or "").strip()
    if explicit:
        return explicit

    pilot_id = str(row.get("pilot_id") or "").strip()
    sample_id = str(row.get("sample_id") or "").strip()
    task = str(row.get("task") or row.get("target_task") or "").strip()

    update = row.get("v2_candidate_update", {}) if isinstance(row.get("v2_candidate_update", {}), dict) else {}
    executor = str(update.get("executor") or row.get("executor") or row.get("target_executor") or "").strip()

    parts = [part for part in (pilot_id, sample_id, task, executor) if part]
    return "|".join(parts) or sample_id


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
        candidate_min_selected_votes: int,
    ):
        self.dataset_root = dataset_root
        self.samples_path = samples_path
        self.review_path = review_path
        self.output_mask_root = output_mask_root
        self.output_samples_path = output_samples_path
        self.max_points = int(max_points)
        self.allow_partial_save = allow_partial_save
        self.top_k_candidates = int(top_k_candidates)
        self.candidate_min_selected_votes = int(candidate_min_selected_votes)
        self.payload_cache: dict[str, dict[str, Any]] = {}
        self.reload()

    def reload(self) -> None:
        self.payload_cache.clear()
        self.samples = read_jsonl(self.samples_path)
        self.samples_by_key = {sample_key(row): row for row in self.samples}
        self.sample_keys_by_id: dict[str, list[str]] = {}
        for row in self.samples:
            self.sample_keys_by_id.setdefault(str(row.get("sample_id", "")), []).append(sample_key(row))
        self.refined_by_key: dict[str, dict[str, Any]] = {}
        if self.output_samples_path.exists():
            for row in read_jsonl(self.output_samples_path):
                self.refined_by_key[sample_key(row)] = row

    def list_samples(self) -> dict[str, Any]:
        rows = []
        for sample in self.samples:
            key = sample_key(sample)
            sample_id = str(sample["sample_id"])
            update = sample.get("v2_candidate_update", {})
            refined = self.refined_by_key.get(key, {})

            task = sample.get("task", "")
            task_display = sample.get("task_display") or sample.get("target_task") or task

            rows.append(
                {
                    "row_key": key,
                    "pilot_id": sample.get("pilot_id", ""),
                    "object_id": sample.get("object_id", ""),
                    "sample_id": sample_id,
                    "object_category": sample.get("object_category", ""),
                    "task": task,
                    "task_display": task_display,
                    "target_task": sample.get("target_task", ""),
                    "source_task": sample.get("source_task", ""),
                    "source_sample_id": sample.get("source_sample_id", ""),
                    "task_taxonomy_version": sample.get("task_taxonomy_version", ""),
                    "task_split_source": sample.get("task_split_source", ""),
                    "executor": update.get("executor", sample.get("executor", sample.get("target_executor", ""))),
                    "selected_candidates": update.get("selected_candidates", []),
                    "positive_points": update.get("positive_points", ""),
                    "review_status": refined.get("point_review_status", "pending"),
                    "reviewer": refined.get("point_review_reviewer", ""),
                    "quality_flag": refined.get("quality_flag", sample.get("quality_flag", "")),
                    "review_mode": update.get("review_mode", sample.get("review_mode", "")),
                    "negative_reason": update.get("negative_reason", sample.get("negative_reason", "")),
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
        try:
            data = np.load(npz_path, allow_pickle=True)
            candidate_ids = [str(item).upper() for item in data["candidate_ids"].tolist()]
            candidate_masks = data["candidate_masks"].astype(np.uint8)
        except Exception as exc:
            return {
                "available": False,
                "error": f"failed to load candidate_npz: {npz_path.relative_to(self.dataset_root).as_posix()} ({exc})",
                "candidates": [],
                "default_selected_candidates": update.get("selected_candidates", []),
            }
        max_visible_index = int(visible_indices.max()) if visible_indices.size else -1
        if candidate_masks.ndim != 2 or candidate_masks.shape[1] <= max_visible_index:
            return {
                "available": False,
                "error": (
                    "candidate mask shape does not match point cloud: "
                    f"candidate_masks={candidate_masks.shape}, max_visible_index={max_visible_index}"
                ),
                "candidates": [],
                "default_selected_candidates": update.get("selected_candidates", []),
            }
        rule_value = update.get("rule_filter_path", "")
        rule: dict[str, Any] = {}
        rule_path = resolve_portable_path(self.dataset_root, rule_value) if rule_value else Path("")
        if rule_path.is_file():
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
        total_candidates = len(candidates)
        if self.top_k_candidates > 0:
            pinned = [item for item in candidates if item["candidate_id"] in default_selected_set]
            rest = [
                item
                for item in candidates
                if item["candidate_id"] not in default_selected_set
                and int(item.get("selected_votes", 0) or 0) >= self.candidate_min_selected_votes
            ]
            candidates = (pinned + rest)[: self.top_k_candidates]
        return {
            "available": True,
            "candidate_manifest": str(manifest_path.relative_to(self.dataset_root).as_posix()),
            "rule_filter_path": str(rule_path.relative_to(self.dataset_root).as_posix()) if rule_path.is_file() else "",
            "default_selected_candidates": default_selected,
            "accepted_candidates": sorted(accepted),
            "uncertain_candidates": sorted(uncertain),
            "candidates": candidates,
            "shown_candidate_count": len(candidates),
            "total_candidate_count": total_candidates,
            "candidate_min_selected_votes": self.candidate_min_selected_votes,
            "notes": "Candidates are ranked proposals. Reviewers choose a subset, then refine points manually.",
        }

    def load_masks(self, sample: dict[str, Any], executor: str, n_points: int) -> tuple[np.ndarray, str]:
        channel = EXECUTOR_ORDER.index(executor)
        candidates = [
            sample.get("multi_channel_mask_path", ""),
            sample.get("checked_mask_path", ""),
            sample.get("source_mask_path", ""),
        ]
        errors: list[str] = []
        for value in candidates:
            if not value:
                continue
            path = resolve_path(self.dataset_root, value)
            if not path.exists():
                errors.append(f"missing {value}")
                continue
            try:
                raw = np.load(path, allow_pickle=False)
            except Exception as exc:
                errors.append(f"failed to load {value}: {exc}")
                continue
            if raw.ndim == 2 and raw.shape == (n_points, len(EXECUTOR_ORDER)):
                return raw.astype(np.uint8), str(value)
            if raw.ndim == 1 and raw.shape[0] == n_points:
                masks = np.zeros((n_points, len(EXECUTOR_ORDER)), dtype=np.uint8)
                masks[:, channel] = (raw > 0).astype(np.uint8)
                return masks, str(value)
            errors.append(f"bad shape {value}: {raw.shape}")
        raise ValueError("No usable mask found. " + "; ".join(errors))

    def resolve_sample_key(self, key: str) -> str:
        if key in self.samples_by_key or key in self.refined_by_key:
            return key
        matches = self.sample_keys_by_id.get(key, [])
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise KeyError(f"Ambiguous sample_id '{key}'. Use row_key instead.")
        return key

    def sample_payload(self, key: str) -> dict[str, Any]:
        key = self.resolve_sample_key(key)
        cached = self.payload_cache.get(key)
        if cached is not None:
            return cached
        sample = self.refined_by_key.get(key) or self.samples_by_key.get(key)
        if sample is None:
            raise KeyError(f"Unknown sample key: {key}")
        sample_id = str(sample["sample_id"])
        points_path = resolve_path(self.dataset_root, sample["point_cloud_path"])
        points = np.load(points_path, allow_pickle=False)
        if points.ndim != 2 or points.shape[1] not in (3, 6):
            raise ValueError(f"Invalid points shape for {sample_id}: {points.shape}")
        update = sample.get("v2_candidate_update", {})
        executor = str(update.get("executor") or sample.get("executor") or "hook")
        if executor not in EXECUTOR_ORDER:
            executor = "hook"
        channel = EXECUTOR_ORDER.index(executor)
        masks, mask_source_path = self.load_masks(sample, executor, points.shape[0])
        seed = abs(hash(key)) % (2**32)
        indices = choose_indices(points.shape[0], self.max_points, seed)
        visible_all_points = int(indices.size) == int(points.shape[0])
        normalized = normalize_points(points)
        visible_points = normalized[indices]
        visible_masks = (masks[indices] > 0).astype(np.uint8)
        candidate_context = self.load_candidate_context(sample, indices)
        payload = {
            "sample": sample,
            "row_key": key,
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
                "mask_source_path": mask_source_path,
                "review_mode": update.get("review_mode", sample.get("review_mode", "")),
                "negative_reason": update.get("negative_reason", sample.get("negative_reason", "")),
            },
            "candidate_context": candidate_context,
        }
        self.payload_cache[key] = payload
        return payload

    def save_edit(self, payload: dict[str, Any]) -> dict[str, Any]:
        key = str(payload.get("row_key", "") or "")
        sample_id = str(payload.get("sample_id", ""))
        if not key:
            executor_hint = str(payload.get("executor", "") or "")
            pilot_hint = str(payload.get("pilot_id", "") or "")
            if pilot_hint:
                key = "|".join(part for part in (pilot_hint, sample_id, executor_hint) if part)
            else:
                matches = [
                    match
                    for match in self.sample_keys_by_id.get(sample_id, [])
                    if not executor_hint
                    or str(self.samples_by_key.get(match, {}).get("v2_candidate_update", {}).get("executor") or self.samples_by_key.get(match, {}).get("executor") or "")
                    == executor_hint
                ]
                key = matches[0] if len(matches) == 1 else self.resolve_sample_key(sample_id)
        key = self.resolve_sample_key(key)
        base_sample = self.refined_by_key.get(key) or self.samples_by_key.get(key)
        if base_sample is None:
            raise KeyError(f"Unknown sample key: {key or sample_id}")
        sample_id = str(base_sample["sample_id"])
        executor = str(payload.get("executor") or base_sample.get("v2_candidate_update", {}).get("executor") or "hook")
        if executor not in EXECUTOR_ORDER:
            raise ValueError(f"Unknown executor: {executor}")
        reviewer = str(payload.get("reviewer") or "").strip()
        if reviewer not in VALID_REVIEWERS:
            raise ValueError("Reviewer identity is required. Choose reviewer_a or reviewer_b before saving.")
        visible_all_points = bool(payload.get("visible_all_points", False))
        if not visible_all_points and not self.allow_partial_save:
            raise ValueError("Refusing partial save: restart app with --max-points 0 or pass --allow-partial-save.")
        positive_indices_raw = payload.get("positive_indices", [])
        if not isinstance(positive_indices_raw, list):
            raise ValueError("positive_indices must be a list.")
        positive_indices = sorted({int(x) for x in positive_indices_raw if int(x) >= 0})
        points_path = resolve_path(self.dataset_root, base_sample["point_cloud_path"])
        points = np.load(points_path, allow_pickle=False)
        masks, source_mask_path = self.load_masks(base_sample, executor, int(points.shape[0]))
        n = masks.shape[0]
        positive_indices = [idx for idx in positive_indices if idx < n]
        channel = EXECUTOR_ORDER.index(executor)
        old_positive = set(np.where(masks[:, channel] > 0)[0].astype(int).tolist())
        new_positive = set(positive_indices)
        refined = masks.copy()
        refined[:, channel] = 0
        if positive_indices:
            refined[np.asarray(positive_indices, dtype=np.int64), channel] = 1
        output_mask_path = self.output_mask_root / f"{safe_name(key or sample_id)}_manual_refined.npy"
        output_mask_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_mask_path, refined)

        now = datetime.now(timezone.utc).isoformat()
        refined_sample = dict(base_sample)
        refined_sample["multi_channel_mask_path"] = str(output_mask_path.relative_to(self.dataset_root).as_posix())
        refined_sample["quality_flag"] = str(payload.get("quality_after_review") or "checked")
        refined_sample["point_review_status"] = str(payload.get("review_status") or "checked")
        refined_sample["point_review_reviewer"] = reviewer
        refined_sample["point_review_notes"] = str(payload.get("notes") or "")
        refined_sample["point_review_updated_at"] = now
        refined_sample["row_key"] = key
        refined_sample["task"] = base_sample.get("task", "")
        refined_sample["task_display"] = base_sample.get(
            "task_display",
            base_sample.get("target_task", base_sample.get("task", "")),
        )
        refined_sample["target_task"] = base_sample.get("target_task", "")
        refined_sample["source_task"] = base_sample.get("source_task", "")
        refined_sample["source_sample_id"] = base_sample.get("source_sample_id", "")
        refined_sample["task_taxonomy_version"] = base_sample.get("task_taxonomy_version", "")
        refined_sample["task_split_source"] = base_sample.get("task_split_source", "")
        refined_sample["v2_point_edit"] = {
            "executor": executor,
            "source_mask_path": source_mask_path,
            "output_mask_path": refined_sample["multi_channel_mask_path"],
            "selected_candidate_ids": [str(item).upper() for item in payload.get("selected_candidate_ids", [])],
            "positive_points_before": len(old_positive),
            "positive_points_after": len(new_positive),
            "added_points": sorted(new_positive - old_positive),
            "removed_points": sorted(old_positive - new_positive),
            "review_decision": str(payload.get("review_decision") or ""),
            "reviewer": reviewer,
            "updated_at": now,
        }
        self.refined_by_key[key] = refined_sample
        self.payload_cache.pop(key, None)
        ordered = []
        for sample in self.samples:
            row_key = sample_key(sample)
            if row_key in self.refined_by_key:
                ordered.append(self.refined_by_key[row_key])
        write_jsonl(self.output_samples_path, ordered)
        record = {
            "created_at": now,
            "row_key": key,
            "sample_id": sample_id,
            "object_id": base_sample.get("object_id", ""),
            "object_category": base_sample.get("object_category", ""),
            "task": base_sample.get("task", ""),
            "task_display": base_sample.get(
                "task_display",
                base_sample.get("target_task", base_sample.get("task", "")),
            ),
            "target_task": base_sample.get("target_task", ""),
            "source_task": base_sample.get("source_task", ""),
            "source_sample_id": base_sample.get("source_sample_id", ""),
            "task_taxonomy_version": base_sample.get("task_taxonomy_version", ""),
            "task_split_source": base_sample.get("task_split_source", ""),
            "executor": executor,
            "reviewer": reviewer,
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
        return {"ok": True, "row_key": key, "sample_id": sample_id, "record": record, "sample": refined_sample}


APP_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Multi-EE v2 点级审查系统</title>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; background: #f4f8ff; color: #172033; font-family: Arial, "Microsoft YaHei", sans-serif; }
    header { height: 52px; padding: 0 16px; background: linear-gradient(90deg, #0f2f57, #1d5fbf); color: white; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 2px 12px rgba(15,47,87,.18); }
    header h1 { margin: 0; font-size: 17px; }
    .header-right { display: flex; align-items: center; gap: 10px; font-size: 12px; }
    .reviewer-chip { border: 1px solid rgba(255,255,255,.55); background: rgba(255,255,255,.14); color: white; border-radius: 999px; padding: 6px 10px; font-size: 12px; }
    .reviewer-chip.missing { background: #b42318; border-color: #fecaca; }
    .reviewer-modal { position: fixed; inset: 0; z-index: 20; display: none; align-items: center; justify-content: center; background: rgba(15,23,42,.46); }
    .reviewer-modal.show { display: flex; }
    .reviewer-dialog { width: min(460px, calc(100vw - 32px)); background: #fff; border-radius: 14px; padding: 20px; box-shadow: 0 24px 80px rgba(15,23,42,.32); border: 1px solid #d7e5f6; }
    .reviewer-dialog h2 { margin: 0 0 8px; font-size: 18px; color: #0f2f57; }
    .reviewer-dialog p { margin: 0 0 14px; color: #475569; font-size: 13px; line-height: 1.6; }
    .reviewer-options { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 12px; }
    .reviewer-options button { padding: 12px 10px; font-size: 14px; }
    .reviewer-note { margin-top: 12px; color: #b42318; font-size: 12px; min-height: 16px; }
    .app { display: grid; grid-template-columns: 330px minmax(620px, 1fr) 380px; height: calc(100vh - 52px); }
    aside, .panel { overflow: auto; background: #fff; border-right: 1px solid #d7e5f6; }
    aside { padding: 12px; }
    .viewer { position: relative; background: radial-gradient(circle at 50% 42%, #ffffff 0, #f7fbff 48%, #eef6ff 100%); overflow: hidden; }
    .panel { border-right: 0; border-left: 1px solid #d7e5f6; padding: 14px; }
    input, select, textarea, button { font-family: inherit; font-size: 13px; }
    input, select, textarea { width: 100%; padding: 8px; border: 1px solid #d7e5f6; border-radius: 6px; background: white; }
    textarea { min-height: 88px; resize: vertical; }
    button { border: 1px solid #2563eb; background: #2563eb; color: white; border-radius: 6px; padding: 8px 10px; cursor: pointer; }
    button.secondary { background: #fff; color: #1d5fbf; border-color: #d7e5f6; }
    button.active { background: #0f2f57; border-color: #0f2f57; }
    .brush-control { display: flex; align-items: center; gap: 7px; padding: 6px 9px; background: rgba(255,255,255,.92); border: 1px solid #d7e5f6; border-radius: 7px; font-size: 12px; color: #334155; }
    .brush-control input { width: 92px; padding: 0; }
    .sample-filter-tabs { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px; margin-top: 9px; }
    .sample-filter-tabs button { padding: 7px 5px; font-size: 12px; border-color: #d7e5f6; background: #fff; color: #1d5fbf; }
    .sample-filter-tabs button.active { background: #0f2f57; border-color: #0f2f57; color: #fff; }
    .sample-list { display: flex; flex-direction: column; gap: 7px; margin-top: 10px; }
    .sample-section { margin: 12px 0 5px; padding: 6px 8px; border-radius: 999px; background: #eaf2ff; color: #0f2f57; font-size: 12px; font-weight: 700; display: flex; justify-content: space-between; }
    .sample { border: 1px solid #d7e5f6; border-radius: 8px; padding: 9px; cursor: pointer; }
    .sample.active { border-color: #2563eb; box-shadow: 0 0 0 2px rgba(37,99,235,.13); }
    .sample.loading { border-color: #2563eb; background: #eff6ff; }
    .sample-id { font-size: 11px; color: #526070; word-break: break-all; }
    .variant-row { margin-top: 7px; padding: 7px; border-radius: 7px; border: 1px solid #e4edf8; background: #f8fbff; }
    .variant-row.active { border-color: #2563eb; background: #eff6ff; }
    .tags { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 6px; }
    .tag { font-size: 11px; padding: 2px 7px; border-radius: 999px; background: #edf1f7; color: #334155; }
    .tag.pending { background: #fff7d6; }
    .tag.checked { background: #dcfce7; }
    .tag.refine_needed { background: #ffedd5; }
    canvas { width: 100%; height: 100%; display: block; cursor: crosshair; }
    .toolbar { position: absolute; left: 12px; top: 12px; right: 12px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; z-index: 2; }
    .task-banner { position: absolute; left: 12px; right: 12px; top: 58px; z-index: 2; display: flex; align-items: center; justify-content: center; pointer-events: none; }
    .task-banner-inner { background: rgba(255,255,255,.94); border: 1px solid #c7dcf5; box-shadow: 0 8px 24px rgba(15,47,87,.10); color: #0f2f57; border-radius: 12px; padding: 9px 14px; font-size: 15px; font-weight: 750; }
    .task-banner-inner span { display: inline-block; margin: 0 5px; padding: 2px 8px; border-radius: 999px; background: #eaf2ff; color: #1d5fbf; font-size: 12px; font-weight: 700; }
    .hud { position: absolute; left: 12px; bottom: 12px; background: rgba(15,47,87,.86); color: white; padding: 9px 11px; border-radius: 7px; font-size: 12px; line-height: 1.55; }
    .box { border: 1px solid #d7e5f6; background: #f8fbff; border-radius: 8px; padding: 10px; margin-bottom: 12px; font-size: 12px; line-height: 1.55; color: #475569; }
    .candidate-list { display: flex; flex-direction: column; gap: 7px; margin: 8px 0 10px; }
    .candidate-item { border: 1px solid #d7e5f6; border-radius: 8px; padding: 8px; background: #fff; cursor: pointer; }
    .candidate-item.selected { border-color: #2563eb; background: #eff6ff; }
    .candidate-item.focused { border-color: #1f6feb; background: #eef6ff; box-shadow: 0 0 0 2px rgba(31,111,235,.12); }
    .candidate-item.locked { border-color: #0f766e; background: #ecfdf5; box-shadow: 0 0 0 2px rgba(15,118,110,.14); }
    .candidate-main { display: flex; gap: 8px; align-items: flex-start; }
    .candidate-swatch { width: 12px; height: 12px; border-radius: 3px; margin-top: 3px; flex: 0 0 auto; }
    .candidate-title { font-size: 12px; font-weight: 650; color: #202635; }
    .candidate-meta { font-size: 11px; color: #64748b; line-height: 1.45; margin-top: 2px; }
    .candidate-actions { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px; }
    .field { margin-bottom: 11px; }
    .field label { display: block; color: #526070; font-size: 12px; margin-bottom: 5px; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 9px; }
    .savebar { position: sticky; bottom: -14px; background: #fff; border-top: 1px solid #d7e5f6; padding-top: 12px; display: flex; gap: 8px; }
    .msg { margin-top: 8px; min-height: 18px; color: #0f766e; font-size: 12px; }
    code { color: #1d5fbf; }
  </style>
</head>
<body>
<header>
  <h1>Multi-EE v2 点级审查系统</h1>
  <div class="header-right">
    <button id="reviewerSwitch" class="reviewer-chip missing" type="button">reviewer: 未选择</button>
    <div id="topStatus">加载中...</div>
  </div>
</header>
<div id="reviewerModal" class="reviewer-modal">
  <div class="reviewer-dialog">
    <h2>选择当前审查身份</h2>
    <p>每次打开网页需要确认本次标注身份。保存时该身份会写入 <code>review_records.jsonl</code> 和 refined samples，避免 reviewer 字段为空。</p>
    <div class="reviewer-options">
      <button id="chooseReviewerA" type="button">reviewer_a</button>
      <button id="chooseReviewerB" type="button">reviewer_b</button>
    </div>
    <div class="reviewer-note" id="reviewerNote"></div>
  </div>
</div>
<div class="app">
  <aside>
    <input id="search" placeholder="搜索 sample/category/task/executor" />
    <div class="sample-filter-tabs">
      <button id="filterPending" class="active">pending</button>
      <button id="filterChecked">checked</button>
      <button id="filterAll">all</button>
    </div>
    <div class="sample-list" id="sampleList"></div>
  </aside>
  <main class="viewer">
    <div class="toolbar">
      <label class="brush-control">brush <input id="brushRadius" type="range" min="6" max="70" value="24"> <span id="brushValue">24px</span></label>
      <button id="modeView">查看/旋转</button>
      <button id="modeToggle">点击切换点</button>
      <button id="modeAdd">只添加</button>
      <button id="modeDelete">只删除</button>
      <button class="secondary" id="resetView">重置视角</button>
      <button class="secondary" id="undoBtn">撤销</button>
    </div>
    <div class="task-banner" id="taskBanner"></div>
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
      <b>候选选择：</b>勾选框表示当前准备采用的候选组合，点云会实时显示已勾选候选的集合。点击候选卡片只用于单独查看某个候选；确认组合后再点击“应用勾选候选”，随后进行点级删除/补点。
      <label style="display:flex;align-items:center;gap:7px;margin-top:8px;color:#334155;">
        <input id="showCandidatePreview" type="checkbox" style="width:auto;"> 显示候选预览颜色
      </label>
      <div class="candidate-list" id="candidateList"></div>
      <div class="candidate-actions">
        <button class="secondary" id="applyCandidatesBtn">应用勾选候选</button>
        <button class="secondary" id="clearMaskBtn">清空当前 mask</button>
      </div>
      <div class="candidate-actions">
        <button class="secondary" id="focusCheckedBtn">预览勾选组合</button>
        <button class="secondary" id="clearFocusBtn">取消候选预览</button>
      </div>
    </div>
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
let listFilter = "pending";
let currentReviewer = "";
let current = null;
let currentIndex = -1;
let sampleCache = new Map();
let loadSeq = 0;
let pendingSampleController = null;
let activeLoadingId = "";
let positives = new Set();
let initialPositives = new Set();
let candidateSets = new Map();
let candidateInfo = [];
let candidateColorById = new Map();
let selectedCandidateIds = new Set();
let focusedCandidateIds = new Set();
let previewLocked = false;
let mode = "toggle";
let rotX = -0.55, rotY = 0.65, zoom = 1.0;
let dragging = false, lastX = 0, lastY = 0;
let painting = false;
let rotating = false;
let paintAction = null;
let cursorX = 0, cursorY = 0, cursorInside = false;
let brushRadius = 24;
let history = [];
let projectedCache = null;
let projectedCacheKey = "";
let previewColorCache = null;
let previewColorCacheKey = "";
let drawQueued = false;
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const CANDIDATE_COLORS = ["#ff4848","#32dc78","#ffd240","#78a0ff","#ff70d2","#46e6eb","#ff9150","#be7dff","#aae65a","#ffffff","#ff7878","#78ffb4"];

function validReviewer(value) {
  return value === "reviewer_a" || value === "reviewer_b";
}

function updateReviewerUI() {
  const btn = document.getElementById("reviewerSwitch");
  if (!btn) return;
  if (validReviewer(currentReviewer)) {
    btn.textContent = `reviewer: ${currentReviewer}`;
    btn.classList.remove("missing");
  } else {
    btn.textContent = "reviewer: 未选择";
    btn.classList.add("missing");
  }
}

function showReviewerModal(force=false) {
  const modal = document.getElementById("reviewerModal");
  if (!modal) return;
  if (force || !validReviewer(currentReviewer)) {
    modal.classList.add("show");
    document.getElementById("reviewerNote").textContent = validReviewer(currentReviewer)
      ? "正在切换当前审查身份。"
      : "保存前必须选择 reviewer_a 或 reviewer_b。";
  }
}

function hideReviewerModal() {
  const modal = document.getElementById("reviewerModal");
  if (modal) modal.classList.remove("show");
}

function setReviewerIdentity(value) {
  if (!validReviewer(value)) return;
  currentReviewer = value;
  window.localStorage.setItem("multi_ee_current_reviewer", value);
  updateReviewerUI();
  hideReviewerModal();
  document.getElementById("message").textContent = `当前审查身份：${value}`;
}

function initReviewerIdentity() {
  const saved = window.localStorage.getItem("multi_ee_current_reviewer") || "";
  currentReviewer = validReviewer(saved) ? saved : "";
  updateReviewerUI();
  showReviewerModal(true);
}

function displayTask(sample) {
  if (!sample) return "";
  return sample.task_display || sample.target_task || sample.task || "";
}

function objectGroupKey(sample) {
  const sampleId = String(sample.sample_id || "");
  const stripped = sampleId.replace(/_(pick_up|open_pull|press_push|lift_carry|lift|open|pull|press|push)$/,"");
  return sample.object_id || stripped;
}

function reviewStatus(sample) {
  return sample.review_status || "pending";
}

function setListFilter(next) {
  listFilter = next;
  ["filterPending", "filterChecked", "filterAll"].forEach(id => document.getElementById(id).classList.remove("active"));
  const activeId = next === "checked" ? "filterChecked" : (next === "all" ? "filterAll" : "filterPending");
  document.getElementById(activeId).classList.add("active");
  renderList();
}


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

  const searched = samples.filter(s =>
    `${s.pilot_id || ""} ${s.sample_id || ""} ${s.object_category || ""} ${s.task || ""} ${s.task_display || ""} ${s.target_task || ""} ${s.source_task || ""} ${s.executor || ""}`
      .toLowerCase()
      .includes(q)
  );

  const pendingCount = samples.filter(s => reviewStatus(s) !== "checked").length;
  const checkedCount = samples.filter(s => reviewStatus(s) === "checked").length;

  const filtered = searched.filter(s => {
    if (listFilter === "pending") return reviewStatus(s) !== "checked";
    if (listFilter === "checked") return reviewStatus(s) === "checked";
    return true;
  });

  document.getElementById("filterPending").textContent = `pending (${pendingCount})`;
  document.getElementById("filterChecked").textContent = `checked (${checkedCount})`;
  document.getElementById("filterAll").textContent = `all (${samples.length})`;
  document.getElementById("topStatus").textContent = `${samples.length} samples | ${listFilter} 显示 ${filtered.length}`;

  const groups = new Map();
  filtered.forEach(s => {
    const key = objectGroupKey(s);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(s);
  });

  const grouped = [...groups.entries()];
  if (!grouped.length) {
    const empty = document.createElement("div");
    empty.className = "box";
    empty.textContent = "当前筛选条件下没有样本。";
    list.appendChild(empty);
    return;
  }

  const header = document.createElement("div");
  header.className = "sample-section";
  header.innerHTML = `<span>${listFilter}</span><span>${grouped.length} objects</span>`;
  list.appendChild(header);

  grouped.forEach(([key, rows]) => {
    rows.sort((a,b) =>
      `${displayTask(a)} ${a.executor}`.localeCompare(`${displayTask(b)} ${b.executor}`)
    );

    const first = rows[0];
    const active = current && rows.some(x => x.row_key === current.row_key);

    const div = document.createElement("div");
    div.className = "sample" + (active ? " active" : "");
    div.onclick = () => loadSample(rows[0].row_key);

    div.innerHTML = `<div class="sample-id">${key}</div>
      <div class="tags">
        <span class="tag">${first.object_category || ""}</span>
        <span class="tag">${rows.length} variants</span>
      </div>`;

    rows.forEach(s => {
      const v = document.createElement("div");
      v.className = "variant-row"
        + (current && current.row_key === s.row_key ? " active" : "")
        + (activeLoadingId === s.row_key ? " loading" : "");

      v.onclick = (event) => {
        event.stopPropagation();
        loadSample(s.row_key);
      };

      const sourceTaskTag = s.source_task ? `<span class="tag">src=${s.source_task}</span>` : "";
      const taxonomyTag = s.task_taxonomy_version ? `<span class="tag">${s.task_taxonomy_version}</span>` : "";

      v.innerHTML = `<div class="tags" style="margin-top:0">
        <span class="tag">${s.pilot_id || ""}</span>
        <span class="tag">${displayTask(s)}</span>
        ${sourceTaskTag}
        ${taxonomyTag}
        <span class="tag">${s.executor || ""}</span>
        <span class="tag ${reviewStatus(s)}">${reviewStatus(s)}</span>
        <span class="tag">pos=${s.positive_points || ""}</span>
      </div>`;

      div.appendChild(v);
    });

    list.appendChild(div);
  });
}


async function loadSamples(loadFirst=true) {
  const res = await fetch("/api/samples");
  const data = await res.json();
  samples = data.samples;
  renderList();
  if (loadFirst && samples.length && !current) await loadSample(samples[0].row_key);
}

function invalidateProjectionCache() {
  projectedCache = null;
  projectedCacheKey = "";
}

function invalidatePreviewColorCache() {
  previewColorCache = null;
  previewColorCacheKey = "";
}

function applySamplePayload(rowKey, payload) {
  current = payload;
  currentIndex = samples.findIndex(s => s.row_key === rowKey);
  positives = new Set();
  candidateSets = new Map();
  candidateColorById = new Map();
  candidateInfo = (current.candidate_context && current.candidate_context.candidates) || [];
  selectedCandidateIds = new Set();
  focusedCandidateIds = new Set();
  previewLocked = false;
  candidateInfo.forEach((c, idx) => {
    candidateSets.set(c.candidate_id, new Set(c.point_indices || []));
    candidateColorById.set(c.candidate_id, CANDIDATE_COLORS[idx % CANDIDATE_COLORS.length]);
    if (c.default_checked) selectedCandidateIds.add(c.candidate_id);
  });
  const ch = current.target_channel;
  current.point_indices.forEach((idx, i) => {
    if (current.masks[i][ch]) positives.add(idx);
  });
  initialPositives = new Set(positives);
  history = [];
  invalidateProjectionCache();
  invalidatePreviewColorCache();
  fillPanel();
  renderList();
  resize();
}

async function loadSample(rowKey, force=false) {
  const seq = ++loadSeq;
  activeLoadingId = rowKey;
  renderList();
  document.getElementById("message").textContent = `加载 ${rowKey} ...`;
  if (pendingSampleController) {
    pendingSampleController.abort();
    pendingSampleController = null;
  }
  if (!force && sampleCache.has(rowKey)) {
    applySamplePayload(rowKey, sampleCache.get(rowKey));
    if (seq === loadSeq) {
      activeLoadingId = "";
      document.getElementById("message").textContent = "";
      renderList();
    }
    return;
  }
  const controller = new AbortController();
  pendingSampleController = controller;
  let res;
  try {
    res = await fetch(`/api/sample?key=${encodeURIComponent(rowKey)}`, {signal: controller.signal});
  } catch (err) {
    if (err.name === "AbortError") return;
    if (seq === loadSeq) {
      activeLoadingId = "";
      pendingSampleController = null;
      renderList();
    }
    throw err;
  }
  if (!res.ok) {
    activeLoadingId = "";
    pendingSampleController = null;
    renderList();
    throw new Error(await res.text());
  }
  const payload = await res.json();
  if (seq !== loadSeq) return;
  sampleCache.set(rowKey, payload);
  applySamplePayload(rowKey, payload);
  activeLoadingId = "";
  pendingSampleController = null;
  document.getElementById("message").textContent = "";
  renderList();
}

function fillPanel() {
  const s = current.sample;
  const reviewMode = (current.review_hint && current.review_hint.review_mode) || (current.sample && current.sample.review_mode) || "";
  const taskName = displayTask(s);

  if (reviewMode === "confirm_empty") {
    ensureSelectOption("reviewDecision", "confirm_empty", "confirm_empty - 确认空标签");
    document.getElementById("reviewDecision").value = "confirm_empty";
  }

  const sourceTaskHtml = s.source_task ? `<span>src=${s.source_task}</span>` : "";
  const taxonomyHtml = s.task_taxonomy_version ? `<span>${s.task_taxonomy_version}</span>` : "";

  document.getElementById("taskBanner").innerHTML =
    `<div class="task-banner-inner">${s.object_category || ""}<span>${taskName}</span>${sourceTaskHtml}${taxonomyHtml}<span>${current.target_executor}</span><span>${reviewMode || "point_refine"}</span>positive=${positives.size}</div>`;

  document.getElementById("sampleId").value = s.sample_id;
  document.getElementById("category").value = s.object_category || "";
  document.getElementById("task").value = taskName;
  document.getElementById("executor").value = current.target_executor;
  document.getElementById("count").value = positives.size;

  const hint = current.review_hint || {};
  const ctxInfo = current.candidate_context || {};
  const candidateSummary = ctxInfo.available
    ? `<br/>shown_candidates: <code>${ctxInfo.shown_candidate_count || 0}/${ctxInfo.total_candidate_count || 0}</code>, min_votes=<code>${ctxInfo.candidate_min_selected_votes ?? ""}</code>`
    : "";
  const candidateError = ctxInfo.error ? `<br/><span style="color:#b42318">candidate warning: ${ctxInfo.error}</span>` : "";

  const taskMeta = `
     task_key: <code>${s.task || ""}</code><br/>
     task_display: <code>${taskName || ""}</code><br/>
     source_task: <code>${s.source_task || ""}</code><br/>
     taxonomy: <code>${s.task_taxonomy_version || ""}</code><br/>
     reviewer: <code>${currentReviewer || "未选择"}</code><br/>`;

  document.getElementById("candidateHint").innerHTML =
    `<b>自动候选来源：</b><br/>
     ${taskMeta}
     selected_candidates: <code>${(hint.selected_candidates || []).join(",") || "(none)"}</code><br/>
     positive_points_before: <code>${hint.positive_points ?? ""}</code>${candidateSummary}${candidateError}<br/>
     这个页面保存的是人工点级 refinement，不会把自动候选直接当 GT。`;

  renderCandidateList();
}


function ensureSelectOption(selectId, value, label) {
  const select = document.getElementById(selectId);
  if (!select) return;
  const exists = Array.from(select.options).some(opt => opt.value === value);
  if (exists) return;
  const option = document.createElement("option");
  option.value = value;
  option.textContent = label;
  select.appendChild(option);
}

function candidateColor(cid) {
  return candidateColorById.get(cid) || CANDIDATE_COLORS[0];
}

function updateCandidateFocusStyles() {
  document.querySelectorAll(".candidate-item").forEach(item => {
    const cid = item.dataset.candidateId;
    const focused = focusedCandidateIds.has(cid);
    item.classList.toggle("focused", focused && !previewLocked);
    item.classList.toggle("locked", focused && previewLocked);
    item.classList.toggle("selected", selectedCandidateIds.has(cid));
  });
}

function setCandidateFocus(ids, locked) {
  focusedCandidateIds = new Set(ids);
  previewLocked = Boolean(locked && focusedCandidateIds.size);
  invalidatePreviewColorCache();
  updateCandidateFocusStyles();
  draw();
}

function renderCandidateList() {
  const box = document.getElementById("candidateList");
  box.innerHTML = "";
  if (!candidateInfo.length) {
    const ctxInfo = current.candidate_context || {};
    const extra = ctxInfo.available
      ? `当前没有达到高置信阈值的候选。可用 --candidate-min-selected-votes 0 或增大 --top-k-candidates 查看低置信候选。`
      : `没有可用候选上下文。${ctxInfo.error || ""}`;
    box.innerHTML = `<div class="candidate-meta">${extra}<br/>仍可继续直接点级编辑当前 mask。</div>`;
    return;
    box.innerHTML = `<div class="candidate-meta">没有可用候选上下文。可继续编辑当前 mask，但无法从 top-k 候选中勾选组合。</div>`;
    return;
  }
  candidateInfo.forEach(c => {
    const item = document.createElement("div");
    item.className =
      "candidate-item"
      + (selectedCandidateIds.has(c.candidate_id) ? " selected" : "")
      + (focusedCandidateIds.has(c.candidate_id) ? (previewLocked ? " locked" : " focused") : "");
    item.dataset.candidateId = c.candidate_id;
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
    checkbox.addEventListener("mousedown", event => event.stopPropagation());
    checkbox.addEventListener("click", event => event.stopPropagation());
    checkbox.onchange = (event) => {
      event.stopPropagation();
      if (checkbox.checked) selectedCandidateIds.add(c.candidate_id);
      else selectedCandidateIds.delete(c.candidate_id);
      syncMaskToSelectedCandidates(true);
    };
    item.onmouseenter = () => {
      if (!previewLocked) setCandidateFocus([c.candidate_id], false);
    };
    item.onmouseleave = () => {
      if (!previewLocked) setCandidateFocus([], false);
    };
    item.onclick = (event) => {
      if (event.target && event.target.tagName === "INPUT") return;
      setCandidateFocus([c.candidate_id], true);
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

function syncMaskToSelectedCandidates(recordHistory=true) {
  if (recordHistory) saveHistory();
  positives = candidateUnion(selectedCandidateIds);
  initialPositives = new Set(positives);
  focusedCandidateIds = new Set();
  previewLocked = false;
  invalidatePreviewColorCache();
  updateCandidateFocusStyles();
  draw();
}

function applySelectedCandidates() {
  syncMaskToSelectedCandidates(true);
}

function clearMask() {
  saveHistory();
  positives = new Set();
  invalidatePreviewColorCache();
  draw();
}

function focusCheckedCandidates() {
  setCandidateFocus([...selectedCandidateIds], true);
}

function clearCandidateFocus() {
  setCandidateFocus([], false);
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
  const key = [
    current.row_key,
    canvas.clientWidth,
    canvas.clientHeight,
    rotX.toFixed(5),
    rotY.toFixed(5),
    zoom.toFixed(5),
  ].join("|");
  if (projectedCache && projectedCacheKey === key) return projectedCache;
  projectedCache = current.points.map((p, i) => {
    const pr = project(p);
    return {x: pr[0], y: pr[1], z: pr[2], original: current.point_indices[i], i};
  }).sort((a,b) => a.z - b.z);
  projectedCacheKey = key;
  return projectedCache;
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

function brushPoints(x, y, positiveOnly=false) {
  const r2 = brushRadius * brushRadius;
  const out = [];
  for (const p of projectedPoints()) {
    if (positiveOnly && !positives.has(p.original)) continue;
    const d = (p.x - x) ** 2 + (p.y - y) ** 2;
    if (d <= r2) out.push(p);
  }
  return out;
}

function applyBrush(x, y) {
  if (!current || mode === "view") return false;
  const action = paintAction || mode;
  let pts = brushPoints(x, y, action === "delete");
  if (!pts.length) {
    const nearest = nearestPoint(x, y, mode === "delete");
    if (nearest) pts = [nearest];
  }
  if (!pts.length) return false;
  if (action === "add") {
    pts.forEach(p => positives.add(p.original));
  } else if (action === "delete") {
    pts.forEach(p => positives.delete(p.original));
  } else {
    const positiveCount = pts.filter(p => positives.has(p.original)).length;
    const shouldDelete = positiveCount >= Math.max(1, pts.length / 2);
    pts.forEach(p => {
      if (shouldDelete) positives.delete(p.original);
      else positives.add(p.original);
    });
  }
  draw();
  return true;
}

function buildPreviewColorMap(ids) {
  const map = new Map();
  ids.forEach(cid => {
    const s = candidateSets.get(cid);
    if (!s) return;
    const color = candidateColor(cid);
    s.forEach(idx => {
      if (!map.has(idx)) map.set(idx, color);
    });
  });
  return map;
}

function previewColorMapForDraw(showPreview) {
  if (!showPreview) return null;
  let ids = [];
  if (focusedCandidateIds.size) {
    ids = [...focusedCandidateIds];
  } else if (!selectedCandidateIds.size) {
    ids = candidateInfo.map(c => c.candidate_id);
  }
  if (!ids.length) return null;
  const key = ids.join("|");
  if (!previewColorCache || previewColorCacheKey !== key) {
    previewColorCache = buildPreviewColorMap(ids);
    previewColorCacheKey = key;
  }
  return previewColorCache;
}

function saveHistory() {
  history.push(new Set(positives));
  if (history.length > 50) history.shift();
}

function draw() {
  if (drawQueued) return;
  drawQueued = true;
  window.requestAnimationFrame(() => {
    drawQueued = false;
    drawNow();
  });
}

function drawNow() {
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  if (!current) return;
  const pts = projectedPoints();
  const showPreview = document.getElementById("showCandidatePreview").checked;
  const previewMap = previewColorMapForDraw(showPreview);
  for (const p of pts) {
    const on = positives.has(p.original);
    let preview = previewMap ? previewMap.get(p.original) : null;
    if (!focusedCandidateIds.size && selectedCandidateIds.size && !on) preview = null;
    if (mode === "delete" && !on) preview = null;
    ctx.beginPath();
    if (focusedCandidateIds.size) {
      ctx.fillStyle = preview || (on ? "#2563eb" : "#7f8fa3");
      ctx.globalAlpha = preview ? 0.98 : (on ? 0.42 : 0.30);
      ctx.arc(p.x, p.y, preview ? 5.2 : (on ? 3.2 : 2.3), 0, Math.PI * 2);
    } else {
      ctx.fillStyle = on ? "#2563eb" : (preview || "#7f8fa3");
      ctx.globalAlpha = on ? 0.96 : (preview ? 0.78 : 0.48);
      ctx.arc(p.x, p.y, on ? 4.4 : (preview ? 3.4 : 2.8), 0, Math.PI * 2);
    }
    ctx.fill();
  }
  ctx.globalAlpha = 1;
  if (cursorInside && mode !== "view") {
    ctx.beginPath();
    ctx.strokeStyle = mode === "delete" ? "#d54444" : (mode === "add" ? "#0f766e" : "#1f6feb");
    ctx.lineWidth = 1.5;
    ctx.globalAlpha = 0.9;
    ctx.arc(cursorX, cursorY, brushRadius, 0, Math.PI * 2);
    ctx.stroke();
    ctx.globalAlpha = 1;
  }
  const added = [...positives].filter(x => !initialPositives.has(x)).length;
  const removed = [...initialPositives].filter(x => !positives.has(x)).length;
  document.getElementById("hud").innerHTML =
    `mode=${mode} | executor=${current.target_executor}<br/>checked_candidates=${[...selectedCandidateIds].join(",") || "(none)"}<br/>preview=${[...focusedCandidateIds].join(",") || "(all/off)"}<br/>positive=${positives.size} | added=${added} | removed=${removed}<br/>拖拽旋转，滚轮缩放；点击按当前模式编辑`;
  document.getElementById("hud").innerHTML =
    `mode=${mode} | executor=${current.target_executor}<br/>checked_candidates=${[...selectedCandidateIds].join(",") || "(none)"}<br/>preview=${[...focusedCandidateIds].join(",") || "(all/off)"}<br/>positive=${positives.size} | added=${added} | removed=${removed}<br/>left=brush edit | right=rotate | wheel=zoom`;
  document.getElementById("count").value = positives.size;
}

function resize() {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  draw();
}

canvas.addEventListener("contextmenu", e => e.preventDefault());
canvas.addEventListener("mousedown", e => {
  if (!current) return;
  e.preventDefault();
  dragging = true; lastX = e.clientX; lastY = e.clientY;
  const rect = canvas.getBoundingClientRect();
  cursorX = e.clientX - rect.left;
  cursorY = e.clientY - rect.top;
  cursorInside = true;
  if (e.button === 2) {
    rotating = true;
    painting = false;
    paintAction = null;
    draw();
    return;
  }
  if (e.button !== 0) return;
  if (mode !== "view" && current) {
    painting = true;
    if (mode === "toggle") {
      const pts = brushPoints(cursorX, cursorY, false);
      const positiveCount = pts.filter(p => positives.has(p.original)).length;
      paintAction = positiveCount >= Math.max(1, pts.length / 2) ? "delete" : "add";
    } else {
      paintAction = mode;
    }
    saveHistory();
    applyBrush(cursorX, cursorY);
  }
});
window.addEventListener("mouseup", e => {
  if (!dragging) return;
  if (painting) {
    painting = false;
    paintAction = null;
    dragging = false;
    draw();
    return;
  }
  if (rotating) {
    rotating = false;
    dragging = false;
    draw();
    return;
  }
  dragging = false;
});
window.addEventListener("mousemove", e => {
  const rect = canvas.getBoundingClientRect();
  cursorX = e.clientX - rect.left;
  cursorY = e.clientY - rect.top;
  cursorInside = cursorX >= 0 && cursorX <= rect.width && cursorY >= 0 && cursorY <= rect.height;
  if (painting) {
    applyBrush(cursorX, cursorY);
    return;
  }
  if (rotating) {
    rotY += (e.clientX - lastX) * 0.008;
    rotX += (e.clientY - lastY) * 0.008;
    lastX = e.clientX; lastY = e.clientY;
    draw();
    return;
  }
  if (!dragging) {
    draw();
    return;
  }
  draw();
});
canvas.addEventListener("mouseleave", () => {
  cursorInside = false;
  if (painting) {
    painting = false;
    paintAction = null;
    dragging = false;
  }
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
  if (!validReviewer(currentReviewer)) {
    showReviewerModal(true);
    throw new Error("请先选择当前审查身份 reviewer_a 或 reviewer_b。");
  }
  const payload = {
    row_key: current.row_key,
    pilot_id: current.sample.pilot_id || "",
    sample_id: current.sample.sample_id,
    executor: current.target_executor,
    selected_candidate_ids: [...selectedCandidateIds].sort(),
    positive_indices: [...positives].sort((a,b) => a-b),
    visible_all_points: current.visible_all_points,
    reviewer: currentReviewer,
    review_status: document.getElementById("reviewStatus").value,
    review_decision: document.getElementById("reviewDecision").value,
    quality_after_review: document.getElementById("quality").value,
    notes: document.getElementById("notes").value,
  };
  const res = await fetch("/api/save_edit", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)});
  const data = await res.json();
  if (!res.ok || !data.ok) throw new Error(data.error || "save failed");
  document.getElementById("message").textContent =
    `已保存 refined mask：${currentReviewer} | positive ${data.record.positive_points_before} -> ${data.record.positive_points_after}`;
  sampleCache.delete(data.row_key);
  await loadSamples(false);
  await loadSample(data.row_key, true);
}

document.getElementById("reviewerSwitch").onclick = () => showReviewerModal(true);
document.getElementById("chooseReviewerA").onclick = () => setReviewerIdentity("reviewer_a");
document.getElementById("chooseReviewerB").onclick = () => setReviewerIdentity("reviewer_b");
document.getElementById("search").oninput = renderList;
document.getElementById("filterPending").onclick = () => setListFilter("pending");
document.getElementById("filterChecked").onclick = () => setListFilter("checked");
document.getElementById("filterAll").onclick = () => setListFilter("all");
document.getElementById("modeView").onclick = () => setMode("view");
document.getElementById("modeToggle").onclick = () => setMode("toggle");
document.getElementById("modeAdd").onclick = () => setMode("add");
document.getElementById("modeDelete").onclick = () => setMode("delete");
document.getElementById("resetView").onclick = () => { rotX = -0.55; rotY = 0.65; zoom = 1; draw(); };
document.getElementById("undoBtn").onclick = () => { if (history.length) { positives = history.pop(); draw(); } };
document.getElementById("applyCandidatesBtn").onclick = applySelectedCandidates;
document.getElementById("clearMaskBtn").onclick = clearMask;
document.getElementById("focusCheckedBtn").onclick = focusCheckedCandidates;
document.getElementById("clearFocusBtn").onclick = clearCandidateFocus;
document.getElementById("showCandidatePreview").onchange = draw;
document.getElementById("brushRadius").oninput = (event) => {
  brushRadius = Number(event.target.value || 24);
  document.getElementById("brushValue").textContent = `${brushRadius}px`;
  draw();
};
document.getElementById("saveBtn").onclick = () => saveEdit().catch(err => alert(err.message));
document.getElementById("reloadBtn").onclick = () => {
  if (!current) return;
  sampleCache.delete(current.row_key);
  loadSample(current.row_key, true).catch(err => alert(err.message));
};
window.addEventListener("resize", resize);
setMode("toggle");
initReviewerIdentity();
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
                key = query.get("key", query.get("id", [""]))[0]
                self.send_json(self.store.sample_payload(key))
                return
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
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
            traceback.print_exc(file=sys.stderr)
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
        candidate_min_selected_votes=args.candidate_min_selected_votes,
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
