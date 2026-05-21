#!/usr/bin/env python3
"""Filter v2 candidate selections with end-effector mechanism rules."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from path_utils import relative_to_dataset, resolve_portable_path


EXECUTOR_ORDER = ["gripper", "suction", "hook", "dexterous_hand"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply executor mechanism rules to v2 candidates.")
    parser.add_argument("--dataset-root", default=".", help="Dataset root directory.")
    parser.add_argument(
        "--pilot-csv",
        default="processed/metadata/vlm_pilot_samples_v0_1.csv",
        help="Pilot CSV relative to dataset root.",
    )
    parser.add_argument(
        "--candidate-root",
        default="processed/vlm_candidate_v2/3d_candidates",
        help="Candidate root relative to dataset root.",
    )
    parser.add_argument(
        "--selection-root",
        default="processed/vlm_candidate_v2/vlm_selection",
        help="VLM selection root relative to dataset root.",
    )
    parser.add_argument(
        "--output-root",
        default="processed/vlm_candidate_v2/rule_filter",
        help="Rule filter output root relative to dataset root.",
    )
    parser.add_argument("--pilot-id", default=None, help="Filter only one pilot row.")
    parser.add_argument("--limit", type=int, default=None, help="Limit selected pilot rows.")
    parser.add_argument("--min-selected-votes", type=int, default=1, help="Minimum VLM selected votes.")
    parser.add_argument(
        "--use-rule-only-if-no-vlm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If no VLM selection exists, allow high-score rule candidates as weak proposals.",
    )
    parser.add_argument(
        "--require-vlm-for-accept",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When VLM selection exists, require enough VLM votes before a candidate can be accepted.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite outputs.")
    return parser.parse_args()


def resolve_path(root: Path, value: str | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Pilot CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def family_score(executor: str, family: str, task: str) -> tuple[float, list[str]]:
    family = family.lower()
    reasons: list[str] = []
    score = 0.0
    hook_like = {
        "thin_structure",
        "edge_or_boundary",
        "extreme_edge_or_lip",
        "protruding_or_thin_part",
        "small_protrusion",
        "visual_component_expanded",
        "expanded_loop_or_handle",
        "loop_or_hole_boundary",
        "paired_loop_or_handle",
        "expanded_functional_seed",
        "expanded_axis_part_component",
        "expanded_extreme_part_component",
        "expanded_existing_weak_mask",
        "visual_detached_structure",
        "visual_detached_component",
        "visual_small_component",
    }
    smooth_like = {"smooth_surface", "smooth_extreme_patch", "central_body"}
    if family == "vlm_coverage_missing_region":
        score += 0.42
        reasons.append("VLM coverage check 指出该区域可能是当前候选池漏掉的任务相关目标部件")
    if executor == "suction":
        if family in {"smooth_surface", "smooth_extreme_patch", "existing_weak_mask"}:
            score += 0.55
            reasons.append("平滑/低曲率或已有 suction 先验，符合吸附候选方向")
        if family in hook_like:
            score -= 0.45
            reasons.append("细杆、边缘或高曲率结构通常不适合吸盘密封")
    elif executor == "hook":
        if family in hook_like:
            score += 0.50
            reasons.append("边界、细长或凸出结构可能提供插入/挂接/扣住条件")
        if family == "paired_loop_or_handle":
            score += 0.18
            reasons.append("成对环/把手候选对 hook 的插入和机械约束尤其重要")
        if family in smooth_like:
            score -= 0.35
            reasons.append("普通平滑面或主体区域通常不能形成 hook 机械互锁")
    elif executor == "gripper":
        if family in hook_like:
            score += 0.45
            reasons.append("细长、边缘或凸起结构更可能形成夹爪接触")
        if family in {"smooth_surface", "central_body"}:
            score -= 0.20
            reasons.append("大平面或主体中心需要相对接触面支持，否则不能直接当夹爪正例")
    elif executor == "dexterous_hand":
        if family in hook_like or family in {"central_body", "existing_weak_mask"}:
            score += 0.38
            reasons.append("可能对应多指包覆、捏取、按压或精细操作区域")
        if family in {"smooth_surface", "smooth_extreme_patch"} and task not in {"press_push"}:
            score -= 0.20
            reasons.append("普通平滑大面不能泛化为灵巧手正例")
    if task == "press_push" and executor == "dexterous_hand" and family in {"small_protrusion", "smooth_surface", "smooth_extreme_patch"}:
        score += 0.18
        reasons.append("press_push 任务允许按钮、开关或可推压面作为灵巧手候选")
    if task in {"open_pull", "lift_carry"} and executor == "hook" and family in hook_like:
        score += 0.12
        reasons.append("open_pull/lift_carry 更强调拉力或提拉约束，边界/环/柄类候选优先")
    return score, reasons


def size_score(executor: str, point_fraction: float, extent_ratio: list[float]) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 0.0
    max_extent = max(extent_ratio) if extent_ratio else 0.0
    min_extent = min(extent_ratio) if extent_ratio else 0.0
    if point_fraction > 0.45:
        if executor in {"hook", "gripper"}:
            score -= 0.35
            reasons.append("候选过大，容易是整物体或主体区域")
        elif executor == "suction":
            score -= 0.10
            reasons.append("候选很大，需要后续人工确认是否为连续可吸附面")
    if point_fraction < 0.005:
        score -= 0.12
        reasons.append("候选点数极少，保留为 uncertain 更稳妥")
    if executor == "hook" and max_extent > 0.65 and point_fraction > 0.18:
        score -= 0.25
        reasons.append("hook 候选跨度过大且点数较多，可能只是普通轮廓边")
    if executor == "suction" and (min_extent < 0.02 or point_fraction < 0.02):
        score -= 0.30
        reasons.append("吸盘候选过窄或面积过小，不利于形成密封 footprint")
    return score, reasons


def build_vlm_vote_maps(selection: dict[str, Any] | None) -> tuple[dict[str, int], dict[str, int]]:
    selected: dict[str, int] = {}
    uncertain: dict[str, int] = {}
    if not selection:
        return selected, uncertain
    for item in selection.get("ranked_candidates", []):
        if not isinstance(item, dict):
            continue
        cid = str(item.get("candidate_id", "")).strip().upper()
        if not cid:
            continue
        selected[cid] = int(item.get("selected_votes", item.get("votes", 0)) or 0)
        uncertain[cid] = int(item.get("uncertain_votes", 0) or 0)
    return selected, uncertain


def visible_candidate_ids(selection: dict[str, Any] | None) -> set[str]:
    if not selection:
        return set()
    ids: set[str] = set()
    for item in selection.get("candidate_ids", []):
        cid = str(item).strip().upper()
        if cid:
            ids.add(cid)
    for item in selection.get("ranked_candidates", []):
        if not isinstance(item, dict):
            continue
        cid = str(item.get("candidate_id", "")).strip().upper()
        if cid:
            ids.add(cid)
    return ids


def enforce_vlm_acceptance_gate(
    scored: list[dict[str, Any]],
    selection: dict[str, Any] | None,
    min_selected_votes: int,
) -> None:
    """Keep rule filtering from accepting candidates that VLM did not support.

    The rule score is a mechanism sanity check, not an independent positive-label
    generator once a VLM selection pass has actually run. Candidates not shown to
    VLM, or shown but not selected/marked uncertain, should not silently enter
    the positive mask.
    """
    if selection is None:
        return
    shown_ids = visible_candidate_ids(selection)
    for item in scored:
        cid = str(item["candidate_id"]).upper()
        selected = int(item.get("selected_votes", 0))
        uncertain = int(item.get("uncertain_votes", 0))
        if shown_ids and cid not in shown_ids:
            if item["decision"] != "reject_candidate":
                item["decision"] = "reject_candidate"
                item["reasons"].append("该候选没有出现在 VLM overlay/selection 候选列表中，不能仅凭规则自动 accept")
            continue
        if selected >= min_selected_votes:
            continue
        if selected > 0 or uncertain > 0:
            if item["decision"] == "accept_candidate":
                item["decision"] = "uncertain"
                item["reasons"].append(
                    f"VLM 支持不足：selected_votes={selected}，低于 min_selected_votes={min_selected_votes}，降为 uncertain"
                )
            continue
        if item["decision"] == "accept_candidate":
            item["decision"] = "reject_candidate"
            item["reasons"].append("VLM selection 已运行，但该候选未被选择或标为 uncertain，不能自动写入正例")


def score_candidate(
    candidate: dict[str, Any],
    executor: str,
    task: str,
    selected_votes: int,
    uncertain_votes: int,
    min_votes: int,
) -> dict[str, Any]:
    score = 0.0
    reasons: list[str] = []
    if executor in candidate.get("recommended_executors", []):
        score += 0.25
        reasons.append("候选生成器推荐给当前执行器")
    if selected_votes >= min_votes:
        score += 0.40
        reasons.append(f"VLM selected votes={selected_votes}")
    elif uncertain_votes > 0:
        score += 0.10
        reasons.append(f"VLM uncertain votes={uncertain_votes}")
    family_delta, family_reasons = family_score(executor, candidate.get("candidate_family", ""), task)
    score += family_delta
    reasons.extend(family_reasons)
    size_delta, size_reasons = size_score(
        executor,
        float(candidate.get("point_fraction", 0.0)),
        [float(x) for x in candidate.get("bbox_extent_ratio", [0, 0, 0])],
    )
    score += size_delta
    reasons.extend(size_reasons)
    if score >= 0.55:
        decision = "accept_candidate"
    elif score >= 0.25:
        decision = "uncertain"
    else:
        decision = "reject_candidate"
    return {
        "candidate_id": candidate["candidate_id"],
        "candidate_name": candidate.get("candidate_name", ""),
        "candidate_family": candidate.get("candidate_family", ""),
        "point_count": int(candidate.get("point_count", 0)),
        "point_fraction": float(candidate.get("point_fraction", 0.0)),
        "selected_votes": int(selected_votes),
        "uncertain_votes": int(uncertain_votes),
        "rule_score": float(round(score, 4)),
        "decision": decision,
        "reasons": reasons,
    }


def filter_for_row(root: Path, args: argparse.Namespace, row: dict[str, str]) -> dict[str, Any]:
    pilot_id = row["pilot_id"]
    candidate_manifest_path = resolve_path(root, args.candidate_root) / pilot_id / "candidate_manifest.json"
    candidate_manifest = read_json(candidate_manifest_path)
    selection_path = resolve_path(root, args.selection_root) / pilot_id / "combined_selection.json"
    selection = read_json(selection_path) if selection_path.exists() else None
    selected_votes, uncertain_votes = build_vlm_vote_maps(selection)
    executor = row.get("executor", candidate_manifest.get("executor", ""))
    task = row.get("task", candidate_manifest.get("task", ""))

    scored = []
    for candidate in candidate_manifest.get("candidates", []):
        cid = str(candidate["candidate_id"]).upper()
        scored.append(
            score_candidate(
                candidate=candidate,
                executor=executor,
                task=task,
                selected_votes=selected_votes.get(cid, 0),
                uncertain_votes=uncertain_votes.get(cid, 0),
                min_votes=args.min_selected_votes,
            )
        )

    has_vlm_signal = any(selected_votes.values()) or any(uncertain_votes.values())
    if selection is not None and not has_vlm_signal:
        for item in scored:
            if item["decision"] == "accept_candidate":
                item["decision"] = "uncertain"
                item["reasons"].append("存在 VLM selection 文件，但 VLM 没有选择/不确定投票；不能仅凭规则自动写入正例")

    if selection is not None and has_vlm_signal and args.require_vlm_for_accept:
        enforce_vlm_acceptance_gate(scored, selection, args.min_selected_votes)

    if selection is None and not args.use_rule_only_if_no_vlm:
        for item in scored:
            item["decision"] = "uncertain"
            item["reasons"].append("未找到 VLM selection，按配置不启用 rule-only accept")

    accepted = [item["candidate_id"] for item in scored if item["decision"] == "accept_candidate"]
    uncertain = [item["candidate_id"] for item in scored if item["decision"] == "uncertain"]
    rejected = [item["candidate_id"] for item in scored if item["decision"] == "reject_candidate"]
    output = {
        "version": "v2",
        "pipeline": "vlm_guided_candidate_selection",
        "pilot_id": pilot_id,
        "sample_id": row["sample_id"],
        "object_category": row.get("object_category", candidate_manifest.get("object_category", "")),
        "task": task,
        "executor": executor,
        "candidate_manifest": relative_to_dataset(root, candidate_manifest_path),
        "selection_path": relative_to_dataset(root, selection_path) if selection_path.exists() else None,
        "accepted_candidates": accepted,
        "uncertain_candidates": uncertain,
        "rejected_candidates": rejected,
        "candidate_scores": scored,
        "notes": "Rule filter is still a candidate-level check. Human review is required before GT use.",
    }
    output_dir = resolve_path(root, args.output_root) / pilot_id
    write_json(output_dir / "rule_filter.json", output, args.overwrite)
    return output


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    rows = read_csv(resolve_path(root, args.pilot_csv))
    if args.pilot_id:
        rows = [row for row in rows if row.get("pilot_id") == args.pilot_id]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No pilot rows selected.")
    outputs = [filter_for_row(root, args, row) for row in rows]
    summary = [
        {
            "pilot_id": item["pilot_id"],
            "sample_id": item["sample_id"],
            "executor": item["executor"],
            "accepted": item["accepted_candidates"],
            "uncertain": item["uncertain_candidates"],
        }
        for item in outputs
    ]
    print(json.dumps({"rows": len(outputs), "summary": summary}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
