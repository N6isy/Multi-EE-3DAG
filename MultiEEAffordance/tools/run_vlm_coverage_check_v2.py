#!/usr/bin/env python3
"""Check whether v2 candidates cover VLM-identified target parts.

This stage sits between candidate overlay rendering and candidate selection:

  rendered point-cloud views + candidate overlays
      -> VLM coverage check
      -> optional missing-candidate supplements
      -> rerender overlays and rerun candidate selection

The important distinction is that this script does not ask the VLM to create
ground truth. It asks the VLM to report coverage failure: "the target functional
part is visible, but none of the current candidate colors covers it." Missing
regions are converted into high-recall 3D candidate proposals through the
existing point-index maps, then still go through VLM selection, rule filtering,
and human review.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from path_utils import relative_to_dataset, resolve_portable_path
from run_qwen3vl_sam2_pilot import (
    EXECUTOR_DEFINITIONS,
    TASK_DEFINITIONS,
    extract_json,
    load_qwen_model,
    load_yaml,
    qwen_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VLM coverage checks for v2 candidates.")
    parser.add_argument("--dataset-root", default=".", help="Dataset root directory.")
    parser.add_argument("--config", default="configs/qwen3vl_sam2_pilot.yaml", help="Qwen config relative to root.")
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
        "--overlay-root",
        default="processed/vlm_candidate_v2/candidate_overlays",
        help="Candidate overlay root relative to dataset root.",
    )
    parser.add_argument(
        "--part-plan-root",
        default="processed/vlm_semantic_part/part_plans",
        help="Optional Qwen3-VL part-plan root relative to dataset root.",
    )
    parser.add_argument(
        "--output-root",
        default="processed/vlm_candidate_v2/coverage_check",
        help="Coverage-check output root relative to dataset root.",
    )
    parser.add_argument("--pilot-id", default=None, help="Run only one pilot row.")
    parser.add_argument("--limit", type=int, default=None, help="Limit selected pilot rows.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite outputs and supplemented candidates.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and write empty coverage checks without loading Qwen.")
    parser.add_argument("--validate-only", action="store_true", help="Validate inputs only.")
    parser.add_argument(
        "--supplement-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append VLM-reported missing regions as new 3D candidates.",
    )
    parser.add_argument("--max-missing-proposals-per-view", type=int, default=3)
    parser.add_argument("--max-missing-candidates", type=int, default=6)
    parser.add_argument("--missing-min-points", type=int, default=4)
    parser.add_argument("--missing-max-fraction", type=float, default=0.45)
    parser.add_argument("--box-point-radius", type=int, default=2, help="2D radius around VLM positive points.")
    parser.add_argument("--expand-hops", type=int, default=1, help="kNN expansion hops for supplemented candidates.")
    parser.add_argument("--k-neighbors", type=int, default=24, help="kNN size for supplemented-candidate expansion.")
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


def load_points(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Point cloud not found: {path}")
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] not in (3, 6):
        raise ValueError(f"Expected point cloud shape [N,3] or [N,6], got {arr.shape}: {path}")
    return arr[:, :3].astype(np.float32)


def pairwise_knn(xyz: np.ndarray, k: int) -> np.ndarray:
    n = xyz.shape[0]
    if n <= 1:
        return np.zeros((n, 0), dtype=np.int64)
    k = max(1, min(int(k), n - 1))
    diff = xyz[:, None, :] - xyz[None, :, :]
    dist2 = np.einsum("ijk,ijk->ij", diff, diff)
    np.fill_diagonal(dist2, np.inf)
    return np.argpartition(dist2, kth=k - 1, axis=1)[:, :k].astype(np.int64)


def knn_expand_mask(mask: np.ndarray, knn: np.ndarray, hops: int) -> np.ndarray:
    out = mask.astype(bool).copy()
    for _ in range(max(0, int(hops))):
        seeds = np.where(out)[0]
        if seeds.size == 0:
            break
        out[knn[seeds].reshape(-1)] = True
    return out.astype(np.uint8)


def bbox_extent_ratio(xyz: np.ndarray, mask: np.ndarray) -> list[float]:
    ids = np.where(mask.astype(bool))[0]
    if ids.size == 0:
        return [0.0, 0.0, 0.0]
    overall = np.ptp(xyz, axis=0) + 1e-8
    local = np.ptp(xyz[ids], axis=0)
    return (local / overall).astype(float).tolist()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return slug or "target_part"


def select_rows(root: Path, args: argparse.Namespace) -> list[dict[str, str]]:
    rows = read_csv(resolve_path(root, args.pilot_csv))
    if args.pilot_id:
        rows = [row for row in rows if row.get("pilot_id") == args.pilot_id]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No pilot rows selected.")
    return rows


def load_part_plan(root: Path, args: argparse.Namespace, pilot_id: str) -> dict[str, Any] | None:
    plan_dir = resolve_path(root, args.part_plan_root) / pilot_id
    for name in ("combined_part_plan.json", "combined_semantic_plan.json"):
        path = plan_dir / name
        if path.exists():
            return read_json(path)
    return None


def image_path_from_overlay_entry(root: Path, entry: dict[str, Any], overlay_dir: Path, key: str) -> Path:
    value = entry.get(key)
    if not value:
        raise FileNotFoundError(f"Overlay entry has no {key}: {entry}")
    path = resolve_portable_path(root, value, overlay_dir)
    if not path.exists():
        raise FileNotFoundError(f"Image path not found for {key}: {path}")
    return path


def expected_target_parts(row: dict[str, str], part_plan: dict[str, Any] | None) -> list[str]:
    parts: list[str] = []
    if part_plan:
        for key in ("target_part_names", "grounding_queries", "target_positive_parts", "target_grounding_queries"):
            for item in part_plan.get(key, []):
                text = str(item).strip()
                if text and text not in parts:
                    parts.append(text)
    if not parts:
        issue = row.get("pilot_reason") or row.get("issue_type") or ""
        if issue:
            parts.append(issue)
    return parts


def build_prompt(
    row: dict[str, str],
    view: str,
    candidates: list[dict[str, Any]],
    part_plan: dict[str, Any] | None,
    image_size: tuple[int, int],
) -> str:
    executor = row.get("executor", "")
    task = row.get("task", "")
    parts = expected_target_parts(row, part_plan)
    candidate_lines = "\n".join(
        "- {id}: {name} | family={family} | points={points} | {desc}".format(
            id=item["candidate_id"],
            name=item.get("candidate_name", ""),
            family=item.get("candidate_family", ""),
            points=item.get("point_count", ""),
            desc=item.get("description", ""),
        )
        for item in candidates
    )
    part_context = "No semantic part plan is available; infer the functional target part from object/task/end-effector."
    if part_plan:
        part_context = json.dumps(
            {
                "target_part_names": part_plan.get("target_part_names", []),
                "grounding_queries": part_plan.get("grounding_queries", []),
                "ranked_queries": part_plan.get("ranked_queries", []),
            },
            ensure_ascii=False,
        )
    return f"""
You are checking candidate-region coverage for a 3D affordance annotation pipeline.

Image 1 is the clean VLM-friendly point-cloud render.
Image 2 is the same view with colored candidate IDs and a legend.
All boxes and points you output must use Image 1 coordinates.

Object category: {row.get('object_category', '')}
Task: {task}
Task definition: {TASK_DEFINITIONS.get(task, task)}
End-effector: {executor}
End-effector definition: {EXECUTOR_DEFINITIONS.get(executor, executor)}
View: {view}
Image 1 size: width={image_size[0]}, height={image_size[1]}
Expected target parts from planner or pilot issue: {parts}
Semantic part-planner context: {part_context}

Candidate list shown in Image 2:
{candidate_lines}

Coverage-check rules:
1. First identify the task-related functional target part for this object/task/end-effector.
2. Decide whether the current colored candidates cover that target part.
3. If the target part is visible in Image 1 but not covered by any candidate color in Image 2, report it as uncovered.
4. If a candidate covers only a tiny seed of a larger functional part, mark partial coverage and propose the missing part.
5. Do not create ground truth. Only report missing candidate proposals for later human review.
6. A missing proposal should be a rough 2D box around the visible target part in Image 1, not around the whole object.
7. If the target part is not visible or the image is too ambiguous, set coverage_status to "uncertain" and do not force a box.

Return strict JSON only:
{{
  "view": "{view}",
  "expected_target_parts": ["part name"],
  "visible_target_parts": ["part name"],
  "candidate_coverage": [
    {{
      "candidate_id": "A",
      "covered_part": "part name",
      "coverage_level": "full|partial|none|uncertain",
      "reason": "short reason"
    }}
  ],
  "coverage_status": "covered|partially_covered|not_covered|uncertain",
  "uncovered_target_parts": [
    {{
      "part_name": "part name",
      "description": "what is visible but not covered",
      "reason": "why existing candidates do not cover it"
    }}
  ],
  "missing_region_proposals": [
    {{
      "part_name": "part name",
      "description": "rough missing functional part",
      "box": [x1, y1, x2, y2],
      "positive_points": [[x, y]],
      "confidence": 0.0,
      "reason": "why this region should become a new candidate"
    }}
  ],
  "should_trigger_missing_candidate": false,
  "reason": "one-sentence coverage judgment"
}}
"""


def run_qwen_json_two_images(
    model: Any,
    processor: Any,
    clean_image_path: Path,
    overlay_image_path: Path,
    prompt: str,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    import torch

    qwen_cfg = cfg.get("qwen3vl", {})
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(clean_image_path)},
                {"type": "image", "image": str(overlay_image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(qwen_device(model))
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": int(qwen_cfg.get("max_new_tokens", 512)),
        "do_sample": False,
    }
    if float(qwen_cfg.get("temperature", 0.0)) > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = float(qwen_cfg.get("temperature", 0.0))
        gen_kwargs["top_p"] = float(qwen_cfg.get("top_p", 1.0))
    with torch.inference_mode():
        generated = model.generate(**inputs, **gen_kwargs)
    trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated)]
    text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    raw = extract_json(text)
    raw["_raw_text"] = text
    return raw


def normalize_box(value: Any, width: int, height: int) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        vals = [float(x) for x in value]
    except (TypeError, ValueError):
        return None
    if max(vals) <= 1.5:
        vals = [vals[0] * width, vals[1] * height, vals[2] * width, vals[3] * height]
    x1, y1, x2, y2 = vals
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = int(max(0, min(width - 1, round(x1))))
    x2 = int(max(0, min(width - 1, round(x2))))
    y1 = int(max(0, min(height - 1, round(y1))))
    y2 = int(max(0, min(height - 1, round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def normalize_points(value: Any, width: int, height: int) -> list[list[int]]:
    if not isinstance(value, list):
        return []
    out: list[list[int]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        try:
            x, y = float(item[0]), float(item[1])
        except (TypeError, ValueError):
            continue
        if max(x, y) <= 1.5:
            x *= width
            y *= height
        xx = int(max(0, min(width - 1, round(x))))
        yy = int(max(0, min(height - 1, round(y))))
        point = [xx, yy]
        if point not in out:
            out.append(point)
    return out


def normalize_coverage(raw: dict[str, Any], view: str, width: int, height: int, max_proposals: int) -> dict[str, Any]:
    status = str(raw.get("coverage_status", "uncertain")).strip().lower()
    if status not in {"covered", "partially_covered", "not_covered", "uncertain"}:
        status = "uncertain"
    proposals: list[dict[str, Any]] = []
    for item in raw.get("missing_region_proposals", []) or []:
        if not isinstance(item, dict):
            continue
        box = normalize_box(item.get("box"), width, height)
        points = normalize_points(item.get("positive_points", []), width, height)
        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if box is None and not points:
            continue
        proposals.append(
            {
                "part_name": str(item.get("part_name", "target_part")).strip() or "target_part",
                "description": str(item.get("description", "")),
                "box": box,
                "positive_points": points,
                "confidence": max(0.0, min(1.0, confidence)),
                "reason": str(item.get("reason", "")),
            }
        )
        if len(proposals) >= max(0, int(max_proposals)):
            break
    trigger = bool(raw.get("should_trigger_missing_candidate", False)) and bool(proposals)
    if status in {"not_covered", "partially_covered"} and proposals:
        trigger = True
    return {
        "view": view,
        "expected_target_parts": raw.get("expected_target_parts", []),
        "visible_target_parts": raw.get("visible_target_parts", []),
        "candidate_coverage": raw.get("candidate_coverage", []),
        "coverage_status": status,
        "uncovered_target_parts": raw.get("uncovered_target_parts", []),
        "missing_region_proposals": proposals,
        "should_trigger_missing_candidate": trigger,
        "reason": str(raw.get("reason", raw.get("notes", ""))),
        "raw_qwen3vl_response": raw,
    }


def mask_from_box_and_points(
    index_map: np.ndarray,
    box: list[int] | None,
    points: list[list[int]],
    point_radius: int,
) -> np.ndarray:
    h, w = index_map.shape
    ids: set[int] = set()
    if box is not None:
        x1, y1, x2, y2 = box
        crop = index_map[max(0, y1) : min(h, y2 + 1), max(0, x1) : min(w, x2 + 1)]
        ids.update(int(x) for x in crop[crop >= 0].reshape(-1).tolist())
    radius = max(0, int(point_radius))
    for x, y in points:
        x0, x1 = max(0, x - radius), min(w - 1, x + radius)
        y0, y1 = max(0, y - radius), min(h - 1, y + radius)
        patch = index_map[y0 : y1 + 1, x0 : x1 + 1]
        ids.update(int(v) for v in patch[patch >= 0].reshape(-1).tolist())
    if not ids:
        return np.zeros((int(index_map.max()) + 1 if index_map.size else 0,), dtype=np.uint8)
    n = int(index_map[index_map >= 0].max()) + 1
    mask = np.zeros((n,), dtype=np.uint8)
    valid_ids = [idx for idx in ids if 0 <= idx < n]
    mask[valid_ids] = 1
    return mask


def candidate_record(
    candidate_id: str,
    name: str,
    mask: np.ndarray,
    family: str,
    description: str,
    executor: str,
    task: str,
    xyz: np.ndarray,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    point_count = int(mask.sum())
    return {
        "candidate_id": candidate_id,
        "candidate_name": name,
        "candidate_family": family,
        "description": description,
        "recommended_executors": [executor] if executor else [],
        "recommended_tasks": [task] if task else [],
        "point_count": point_count,
        "point_fraction": float(point_count / max(1, mask.shape[0])),
        "bbox_extent_ratio": bbox_extent_ratio(xyz, mask.astype(bool)),
        "quality_hint": "vlm_missing_candidate",
        "priority": 1.0,
        "provenance": "vlm_coverage_missing_candidate",
        "coverage_sources": sources,
    }


def next_missing_id(existing: set[str], index: int) -> str:
    idx = max(1, index)
    while True:
        cid = f"M{idx}"
        if cid not in existing:
            return cid
        idx += 1


def is_duplicate(mask: np.ndarray, existing_masks: np.ndarray, threshold: float = 0.88) -> bool:
    new_bool = mask.astype(bool)
    for old in existing_masks:
        old_bool = old.astype(bool)
        union = np.logical_or(new_bool, old_bool).sum()
        if union == 0:
            continue
        iou = np.logical_and(new_bool, old_bool).sum() / union
        if iou >= threshold:
            return True
    return False


def supplement_missing_candidates(
    root: Path,
    args: argparse.Namespace,
    row: dict[str, str],
    candidate_manifest_path: Path,
    overlay_manifest_path: Path,
    coverage_by_view: list[dict[str, Any]],
    coverage_output_path: Path,
) -> dict[str, Any]:
    if not args.overwrite:
        raise FileExistsError("Missing-candidate supplement modifies candidate files; rerun with --overwrite.")

    candidate_manifest = read_json(candidate_manifest_path)
    candidate_npz_path = resolve_portable_path(root, candidate_manifest["candidate_npz"], candidate_manifest_path.parent)
    data = np.load(candidate_npz_path, allow_pickle=True)
    old_masks = data["candidate_masks"].astype(np.uint8)
    old_candidates = list(candidate_manifest.get("candidates", []))
    if not old_candidates:
        raise ValueError(f"No existing candidates to supplement: {candidate_manifest_path}")

    point_path = resolve_portable_path(root, candidate_manifest["point_cloud_path"], candidate_manifest_path.parent)
    xyz = load_points(point_path)
    if old_masks.shape[1] != xyz.shape[0]:
        raise ValueError(f"Candidate mask length {old_masks.shape[1]} does not match point count {xyz.shape[0]}")
    knn = pairwise_knn(xyz, args.k_neighbors) if int(args.expand_hops) > 0 else np.zeros((xyz.shape[0], 0), dtype=np.int64)
    overlay_manifest = read_json(overlay_manifest_path)
    view_entries = {str(item["view"]): item for item in overlay_manifest.get("views", [])}

    grouped_masks: dict[str, list[np.ndarray]] = defaultdict(list)
    grouped_sources: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for coverage in coverage_by_view:
        if not coverage.get("should_trigger_missing_candidate"):
            continue
        view = str(coverage["view"])
        view_entry = view_entries.get(view)
        if not view_entry:
            continue
        index_path = resolve_portable_path(root, view_entry["point_index_path"], overlay_manifest_path.parent)
        if not index_path.exists():
            raise FileNotFoundError(f"Point index map not found for coverage supplement: {index_path}")
        index_map = np.load(index_path)
        for proposal in coverage.get("missing_region_proposals", []):
            mask = mask_from_box_and_points(
                index_map=index_map,
                box=proposal.get("box"),
                points=proposal.get("positive_points", []),
                point_radius=args.box_point_radius,
            )
            if mask.shape[0] < xyz.shape[0]:
                padded = np.zeros((xyz.shape[0],), dtype=np.uint8)
                padded[: mask.shape[0]] = mask
                mask = padded
            elif mask.shape[0] > xyz.shape[0]:
                mask = mask[: xyz.shape[0]]
            if int(args.expand_hops) > 0:
                mask = knn_expand_mask(mask, knn, args.expand_hops)
            count = int(mask.sum())
            if count < int(args.missing_min_points):
                continue
            if count > int(xyz.shape[0] * float(args.missing_max_fraction)):
                continue
            key = slugify(proposal.get("part_name", "target_part"))
            grouped_masks[key].append(mask.astype(np.uint8))
            grouped_sources[key].append(
                {
                    "view": view,
                    "part_name": proposal.get("part_name", "target_part"),
                    "box": proposal.get("box"),
                    "positive_points": proposal.get("positive_points", []),
                    "confidence": proposal.get("confidence", 0.0),
                    "reason": proposal.get("reason", ""),
                    "coverage_check_path": relative_to_dataset(root, coverage_output_path),
                }
            )

    new_masks: list[np.ndarray] = []
    new_candidates: list[dict[str, Any]] = []
    existing_ids = {str(item.get("candidate_id", "")).upper() for item in old_candidates}
    for index, (part_key, masks) in enumerate(grouped_masks.items(), start=1):
        if len(new_candidates) >= int(args.max_missing_candidates):
            break
        fused = (np.stack(masks, axis=0).sum(axis=0) > 0).astype(np.uint8)
        if int(fused.sum()) < int(args.missing_min_points):
            continue
        if is_duplicate(fused, old_masks):
            continue
        cid = next_missing_id(existing_ids, index)
        existing_ids.add(cid)
        sources = grouped_sources[part_key]
        part_label = sources[0].get("part_name", part_key) if sources else part_key
        new_masks.append(fused)
        new_candidates.append(
            candidate_record(
                candidate_id=cid,
                name=f"vlm_missing_{slugify(part_label)}",
                mask=fused,
                family="vlm_coverage_missing_region",
                description=(
                    "VLM coverage check reported that the task-related target part is visible but not covered "
                    "by existing candidate regions. This is a high-recall supplement for human review."
                ),
                executor=row.get("executor", ""),
                task=row.get("task", ""),
                xyz=xyz,
                sources=sources,
            )
        )

    if not new_candidates:
        candidate_manifest.setdefault("coverage_supplement_history", []).append(
            {
                "coverage_check_path": relative_to_dataset(root, coverage_output_path),
                "added_candidates": [],
                "reason": "No valid missing candidate survived projection, size filtering, or duplicate filtering.",
            }
        )
        write_json(candidate_manifest_path, candidate_manifest, args.overwrite)
        return {"added_candidates": [], "candidate_manifest": relative_to_dataset(root, candidate_manifest_path)}

    combined_masks = np.concatenate([np.stack(new_masks, axis=0).astype(np.uint8), old_masks], axis=0)
    combined_candidates = new_candidates + old_candidates
    np.savez_compressed(
        candidate_npz_path,
        pilot_id=candidate_manifest.get("pilot_id", row.get("pilot_id", "")),
        sample_id=candidate_manifest.get("sample_id", row.get("sample_id", "")),
        executor=row.get("executor", candidate_manifest.get("executor", "")),
        candidate_ids=np.asarray([item["candidate_id"] for item in combined_candidates], dtype=object),
        candidate_names=np.asarray([item["candidate_name"] for item in combined_candidates], dtype=object),
        candidate_families=np.asarray([item["candidate_family"] for item in combined_candidates], dtype=object),
        candidate_masks=combined_masks,
    )
    candidate_manifest["candidate_count"] = int(combined_masks.shape[0])
    candidate_manifest["candidates"] = combined_candidates
    candidate_manifest.setdefault("coverage_supplement_history", []).append(
        {
            "coverage_check_path": relative_to_dataset(root, coverage_output_path),
            "added_candidates": [item["candidate_id"] for item in new_candidates],
            "added_candidate_count": len(new_candidates),
            "notes": "Inserted before existing candidates so the next overlay render exposes them to VLM selection.",
        }
    )
    candidate_manifest["notes"] = (
        str(candidate_manifest.get("notes", ""))
        + " Coverage check may prepend VLM-reported missing candidates; all remain proposals requiring review."
    )
    write_json(candidate_manifest_path, candidate_manifest, args.overwrite)
    return {
        "added_candidates": [item["candidate_id"] for item in new_candidates],
        "candidate_manifest": relative_to_dataset(root, candidate_manifest_path),
        "candidate_npz": relative_to_dataset(root, candidate_npz_path),
    }


def run_for_row(
    root: Path,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    row: dict[str, str],
    model: Any | None,
    processor: Any | None,
) -> dict[str, Any]:
    pilot_id = row["pilot_id"]
    candidate_manifest_path = resolve_path(root, args.candidate_root) / pilot_id / "candidate_manifest.json"
    overlay_manifest_path = resolve_path(root, args.overlay_root) / pilot_id / "overlay_manifest.json"
    candidate_manifest = read_json(candidate_manifest_path)
    overlay_manifest = read_json(overlay_manifest_path)
    part_plan = load_part_plan(root, args, pilot_id)
    output_dir = resolve_path(root, args.output_root) / pilot_id
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage_output_path = output_dir / "combined_coverage_check.json"

    coverage_by_view: list[dict[str, Any]] = []
    for view_entry in overlay_manifest.get("views", []):
        view = str(view_entry["view"])
        clean_path = image_path_from_overlay_entry(root, view_entry, overlay_manifest_path.parent, "source_image_path")
        selector_value = view_entry.get("selector_path") or view_entry.get("overlay_path")
        selector_path = resolve_portable_path(root, selector_value, overlay_manifest_path.parent)
        if not selector_path.exists():
            raise FileNotFoundError(f"Selector/overlay image not found: {selector_path}")
        with Image.open(clean_path) as image:
            width, height = image.size
        if args.validate_only:
            coverage_by_view.append({"view": view, "validated": True})
            continue
        if args.dry_run:
            coverage = {
                "view": view,
                "expected_target_parts": expected_target_parts(row, part_plan),
                "visible_target_parts": [],
                "candidate_coverage": [],
                "coverage_status": "uncertain",
                "uncovered_target_parts": [],
                "missing_region_proposals": [],
                "should_trigger_missing_candidate": False,
                "reason": "dry_run",
                "raw_qwen3vl_response": {},
            }
        else:
            prompt = build_prompt(row, view, overlay_manifest.get("candidates", []), part_plan, (width, height))
            raw = run_qwen_json_two_images(model, processor, clean_path, selector_path, prompt, cfg)
            coverage = normalize_coverage(raw, view, width, height, args.max_missing_proposals_per_view)
        view_path = output_dir / f"{view}_coverage.json"
        write_json(view_path, coverage, args.overwrite)
        coverage_by_view.append(
            {
                **coverage,
                "coverage_path": relative_to_dataset(root, view_path),
            }
        )

    supplement_result: dict[str, Any] = {"added_candidates": []}
    if not args.validate_only and args.supplement_missing and any(item.get("should_trigger_missing_candidate") for item in coverage_by_view):
        supplement_result = supplement_missing_candidates(
            root=root,
            args=args,
            row=row,
            candidate_manifest_path=candidate_manifest_path,
            overlay_manifest_path=overlay_manifest_path,
            coverage_by_view=coverage_by_view,
            coverage_output_path=coverage_output_path,
        )

    combined = {
        "version": "v2.2",
        "pipeline": "vlm_coverage_check_and_missing_candidate_supplement",
        "pilot_id": pilot_id,
        "sample_id": row["sample_id"],
        "object_category": row.get("object_category", candidate_manifest.get("object_category", "")),
        "task": row.get("task", candidate_manifest.get("task", "")),
        "executor": row.get("executor", candidate_manifest.get("executor", "")),
        "candidate_manifest": relative_to_dataset(root, candidate_manifest_path),
        "overlay_manifest": relative_to_dataset(root, overlay_manifest_path),
        "part_plan": relative_to_dataset(root, resolve_path(root, args.part_plan_root) / pilot_id / "combined_part_plan.json")
        if (resolve_path(root, args.part_plan_root) / pilot_id / "combined_part_plan.json").exists()
        else None,
        "view_results": [
            {
                "view": item.get("view"),
                "coverage_path": item.get("coverage_path"),
                "coverage_status": item.get("coverage_status"),
                "should_trigger_missing_candidate": item.get("should_trigger_missing_candidate", False),
                "missing_proposals": len(item.get("missing_region_proposals", [])),
                "reason": item.get("reason", ""),
            }
            for item in coverage_by_view
        ],
        "coverage_by_view": coverage_by_view,
        "supplement_result": supplement_result,
        "notes": (
            "Coverage check only identifies candidate-pool holes. Supplemented candidates are proposals "
            "and must be rerendered, selected by VLM/rules, and human-reviewed."
        ),
    }
    write_json(coverage_output_path, combined, args.overwrite)
    return combined


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    cfg = load_yaml(resolve_path(root, args.config))
    rows = select_rows(root, args)
    model = None
    processor = None
    if not args.validate_only and not args.dry_run:
        model, processor = load_qwen_model(cfg, root)
    try:
        from tqdm import tqdm

        row_iter = tqdm(rows, desc="coverage", unit="row")
    except Exception:
        row_iter = rows
    print(f"[coverage] rows={len(rows)} estimated_view_calls={len(rows) * 8}", flush=True)
    outputs = [run_for_row(root, args, cfg, row, model, processor) for row in row_iter]
    print(
        json.dumps(
            {
                "rows": len(outputs),
                "validate_only": args.validate_only,
                "dry_run": args.dry_run,
                "supplement_missing": args.supplement_missing,
                "outputs": [
                    {
                        "pilot_id": item["pilot_id"],
                        "sample_id": item["sample_id"],
                        "coverage_check": f"processed/vlm_candidate_v2/coverage_check/{item['pilot_id']}/combined_coverage_check.json",
                        "added_candidates": item.get("supplement_result", {}).get("added_candidates", []),
                    }
                    for item in outputs
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
