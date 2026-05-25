#!/usr/bin/env python3
"""Probe Qwen3-VL on naturalized renders without touching v3 pipeline outputs.

This script is an isolated experiment. It reads natural_surface_vlm renders,
asks Qwen3-VL whether the task/end-effector target part is recognizable, and
optionally asks for coarse 2D boxes/points for diagnosis. It does not create
3D masks and does not modify v2/v3 candidate directories.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

DATASET_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = DATASET_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from path_utils import relative_to_dataset, resolve_portable_path  # noqa: E402
from run_qwen3vl_sam2_pilot import (  # noqa: E402
    EXECUTOR_DEFINITIONS,
    TASK_DEFINITIONS,
    extract_json,
    load_qwen_model,
    load_yaml,
    qwen_device,
)


EXECUTOR_RULES = {
    "gripper": (
        "parallel two-finger gripper: target opposing or potentially paired contact surfaces that can form a stable clamp."
    ),
    "suction": (
        "single suction cup: target continuous low-curvature sealable surfaces; reject holes, rods, handles, and narrow edges."
    ),
    "hook": (
        "single hook: target holes, rings, handle-loop inner rims, lips, or behind-flange structures that allow insertion, "
        "mechanical interlocking, and pulling/lifting force."
    ),
    "dexterous_hand": (
        "multi-finger hand: target task-related grasp, wrap, pinch, pull, press, or fine-manipulation regions."
    ),
}


CATEGORY_HINTS = {
    "bag": {
        "hook": {
            "target": ["bag handle loop", "top handle", "handle inner rim"],
            "reject": ["bag body panel", "flat bag surface", "ordinary side edge"],
        }
    },
    "scissors": {
        "hook": {
            "target": ["finger holes", "handle loops", "inner rim of handle rings"],
            "reject": ["blade", "cutting edge", "blade tip", "ordinary blade boundary"],
        }
    },
    "mug": {
        "hook": {"target": ["handle opening", "handle inner rim"], "reject": ["cup body", "cup wall", "rim only"]},
        "gripper": {"target": ["handle", "opposing cup side surfaces"], "reject": ["inside cavity"]},
        "suction": {"target": ["smooth outer cup wall"], "reject": ["handle", "rim", "inside cavity"]},
    },
    "keyboard": {
        "dexterous_hand": {"target": ["keys", "button tops"], "reject": ["keyboard frame", "side wall"]},
        "press_push": {"target": ["keys", "button tops"], "reject": ["keyboard frame", "side wall"]},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a natural-render VLM recognition/localization probe.")
    parser.add_argument("--dataset-root", default=".", help="Dataset root directory.")
    parser.add_argument("--config", default="configs/qwen3vl_sam2_pilot.yaml", help="Qwen config relative to dataset root.")
    parser.add_argument(
        "--pilot-csv",
        default="processed/metadata/vlm_pilot_samples_v0_1.csv",
        help="Pilot CSV relative to dataset root.",
    )
    parser.add_argument(
        "--renders-root",
        default="processed/natural_surface_vlm/renders",
        help="Natural render root relative to dataset root.",
    )
    parser.add_argument(
        "--output-root",
        default="processed/natural_surface_vlm/vlm_probe",
        help="Probe output root relative to dataset root.",
    )
    parser.add_argument("--pilot-id", required=True, help="Pilot row id to probe.")
    parser.add_argument("--manifest", default="", help="Optional explicit natural view_manifest.json path.")
    parser.add_argument(
        "--views",
        default="",
        help="Comma-separated view subset. Default uses all views in the natural render manifest.",
    )
    parser.add_argument(
        "--image-key",
        default="natural_render_path",
        choices=["natural_render_path", "panel_path", "confidence_image_path"],
        help="Which manifest image to send to Qwen3-VL. Use natural_render_path for the main test.",
    )
    parser.add_argument(
        "--probe-mode",
        choices=["semantic", "localize", "semantic_and_localize"],
        default="semantic_and_localize",
        help="Semantic tests target understanding; localize asks for rough 2D boxes/points.",
    )
    parser.add_argument(
        "--no-refine-localization",
        action="store_true",
        help="Disable foreground/upper-structure refinement for VLM boxes and points.",
    )
    parser.add_argument(
        "--refine-min-confidence",
        type=float,
        default=0.25,
        help="Minimum natural-render confidence for foreground refinement.",
    )
    parser.add_argument(
        "--refine-snap-radius",
        type=int,
        default=48,
        help="Maximum pixel distance for snapping VLM points to valid foreground.",
    )
    parser.add_argument(
        "--upper-margin",
        type=int,
        default=24,
        help="Extra pixels below detected body top included for upper handle/ring/loop refinement.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Write deterministic placeholder probe outputs.")
    parser.add_argument("--validate-only", action="store_true", help="Validate paths without loading Qwen.")
    return parser.parse_args()


def resolve_path(root: Path, value: str | Path, base_dir: Path | None = None) -> Path:
    return resolve_portable_path(root, value, base_dir=base_dir)


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


def selected_row(root: Path, args: argparse.Namespace) -> dict[str, str]:
    rows = read_csv(resolve_path(root, args.pilot_csv))
    matches = [row for row in rows if row.get("pilot_id") == args.pilot_id]
    if not matches:
        raise ValueError(f"Pilot id not found: {args.pilot_id}")
    return matches[0]


def category_hint(row: dict[str, str]) -> dict[str, list[str]]:
    category = row.get("object_category", "").lower()
    executor = row.get("executor", "").lower()
    task = row.get("task", "").lower()
    for key, hints in CATEGORY_HINTS.items():
        if key in category:
            if executor in hints:
                return hints[executor]
            if task in hints:
                return hints[task]
    return {"target": [], "reject": []}


def manifest_path_for_row(root: Path, args: argparse.Namespace, row: dict[str, str]) -> Path:
    if args.manifest:
        return resolve_path(root, args.manifest)
    return resolve_path(root, args.renders_root) / row["sample_id"] / "view_manifest.json"


def selected_view_entries(manifest: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    entries = list(manifest.get("views", []))
    if args.views:
        wanted = {item.strip().lower() for item in args.views.split(",") if item.strip()}
        entries = [entry for entry in entries if str(entry.get("view", "")).lower() in wanted]
    if not entries:
        raise ValueError("No view entries selected from natural render manifest.")
    return entries


def normalize_list(value: Any, max_count: int = 12) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        return []
    out: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
        if len(out) >= max_count:
            break
    return out


def normalize_box(box: Any, image_size: int) -> list[int] | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    except (TypeError, ValueError):
        return None
    x1, x2 = sorted((max(0, min(image_size - 1, x1)), max(0, min(image_size - 1, x2))))
    y1, y2 = sorted((max(0, min(image_size - 1, y1)), max(0, min(image_size - 1, y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def normalize_points(points: Any, image_size: int, max_count: int = 12) -> list[list[int]]:
    out: list[list[int]] = []
    if not isinstance(points, (list, tuple)):
        return out
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        try:
            x, y = int(round(float(point[0]))), int(round(float(point[1])))
        except (TypeError, ValueError):
            continue
        out.append([max(0, min(image_size - 1, x)), max(0, min(image_size - 1, y))])
        if len(out) >= max_count:
            break
    return out


def semantic_prompt(row: dict[str, str], view: str, image_size: int) -> str:
    executor = row.get("executor", "")
    task = row.get("task", "")
    hint = category_hint(row)
    return f"""
You are evaluating whether a naturalized render from a sparse 3D point cloud helps VLM-based affordance annotation.

Image: a single natural-like render, size {image_size}x{image_size}.
Object category: {row.get('object_category', '')}
Task: {task}
Task definition: {TASK_DEFINITIONS.get(task, task)}
End-effector: {executor}
End-effector definition: {EXECUTOR_DEFINITIONS.get(executor, executor)}
Executor mechanism: {EXECUTOR_RULES.get(executor, executor)}
View: {view}
Prior target hints for this category/executor: {hint.get('target', [])}
Prior reject hints for this category/executor: {hint.get('reject', [])}

Your job:
1. Determine whether the task-relevant functional target region is recognizable in this naturalized render.
2. Name the target part(s) that should be grounded.
3. Name reject parts that should not be positive.
4. Explain whether this view is useful for VLM-assisted labeling.

For hook, use the three-step mechanism test:
- Can the hook enter or go behind the structure?
- Can it catch/interlock instead of sliding off?
- Can it pull/lift along the task direction?

Return strict JSON only:
{{
  "view": "{view}",
  "visible_object_parts": ["short part names"],
  "target_positive_parts": ["task-related functional target part names"],
  "target_grounding_queries": ["short phrases suitable for later grounding"],
  "reject_negative_parts": ["visible parts that should not be positive"],
  "uncertain_parts": ["uncertain parts requiring human review"],
  "target_region_description": "plain-language description of the target region in this image",
  "mechanism_check": {{
    "can_enter_or_contact": true,
    "can_hold_or_constrain": true,
    "can_apply_task_force": true,
    "reason": "mechanism-level explanation"
  }},
  "view_usefulness": "good|partial|poor",
  "feasible": true,
  "confidence": 0.0,
  "notes": "short note"
}}

Rules:
- Do not mark all visible surfaces as target.
- Do not treat the whole object body as target unless the mechanism truly requires the body.
- If the right semantic part is visible but still hard to localize, set view_usefulness="partial".
- If no target is visible in this view, set feasible=false and explain why.
"""


def localization_prompt(row: dict[str, str], view: str, image_size: int, semantic_plan: dict[str, Any] | None) -> str:
    executor = row.get("executor", "")
    task = row.get("task", "")
    semantic_hint = semantic_plan or {}
    return f"""
You are testing rough 2D localization on a naturalized render from a sparse 3D point cloud.

Image size: {image_size}x{image_size}.
Object category: {row.get('object_category', '')}
Task: {task}
End-effector: {executor}
Mechanism: {EXECUTOR_RULES.get(executor, executor)}
View: {view}
Previously identified target parts: {semantic_hint.get('target_positive_parts', [])}
Target description: {semantic_hint.get('target_region_description', '')}

Return coarse localization of only the functional target region for this task and end-effector.
Coordinates must be pixel coordinates in the input image.

Return strict JSON only:
{{
  "view": "{view}",
  "can_localize": true,
  "target_region_description": "what region the coordinates refer to",
  "boxes": [[x1, y1, x2, y2]],
  "positive_points": [[x, y]],
  "negative_points": [[x, y]],
  "confidence": 0.0,
  "notes": "short note"
}}

Rules:
- Output at most 3 boxes and 8 positive points.
- For hook on a bag, target the handle loop / inner rim, not the flat bag body.
- Prefer a small tight region around the target part instead of the whole object.
- If the target is not visible or cannot be localized, set can_localize=false and use empty boxes/points.
"""


def run_qwen_json(model: Any, processor: Any, image_path: Path, prompt: str, cfg: dict[str, Any]) -> dict[str, Any]:
    import torch

    qwen_cfg = cfg.get("qwen3vl", {})
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
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
    gen_kwargs: dict[str, Any] = {"max_new_tokens": int(qwen_cfg.get("max_new_tokens", 512)), "do_sample": False}
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


def normalize_semantic(raw: dict[str, Any], view: str) -> dict[str, Any]:
    target = normalize_list(raw.get("target_positive_parts", raw.get("target_part_names", [])), max_count=12)
    queries = normalize_list(raw.get("target_grounding_queries", raw.get("grounding_queries", target)), max_count=12)
    reject = normalize_list(raw.get("reject_negative_parts", raw.get("reject_parts", [])), max_count=12)
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    feasible = bool(raw.get("feasible", bool(target or queries))) and bool(target or queries)
    mechanism = raw.get("mechanism_check", {})
    if not isinstance(mechanism, dict):
        mechanism = {"reason": str(mechanism)}
    return {
        "view": view,
        "visible_object_parts": normalize_list(raw.get("visible_object_parts", []), max_count=16),
        "target_positive_parts": target if feasible else [],
        "target_grounding_queries": queries if feasible else [],
        "reject_negative_parts": reject,
        "uncertain_parts": normalize_list(raw.get("uncertain_parts", []), max_count=12),
        "target_region_description": str(raw.get("target_region_description", "")),
        "mechanism_check": {
            "can_enter_or_contact": bool(mechanism.get("can_enter_or_contact", feasible)),
            "can_hold_or_constrain": bool(mechanism.get("can_hold_or_constrain", feasible)),
            "can_apply_task_force": bool(mechanism.get("can_apply_task_force", feasible)),
            "reason": str(mechanism.get("reason", "")),
        },
        "view_usefulness": str(raw.get("view_usefulness", "partial")),
        "feasible": feasible,
        "confidence": max(0.0, min(1.0, confidence)),
        "notes": str(raw.get("notes", "")),
        "raw_qwen3vl_response": raw,
    }


def normalize_localization(raw: dict[str, Any], view: str, image_size: int) -> dict[str, Any]:
    boxes = []
    for box in raw.get("boxes", []) if isinstance(raw.get("boxes", []), list) else []:
        norm = normalize_box(box, image_size)
        if norm is not None:
            boxes.append(norm)
        if len(boxes) >= 3:
            break
    points = normalize_points(raw.get("positive_points", []), image_size, max_count=8)
    neg = normalize_points(raw.get("negative_points", []), image_size, max_count=8)
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    can_localize = bool(raw.get("can_localize", bool(boxes or points))) and bool(boxes or points)
    return {
        "view": view,
        "can_localize": can_localize,
        "target_region_description": str(raw.get("target_region_description", "")),
        "boxes": boxes if can_localize else [],
        "positive_points": points if can_localize else [],
        "negative_points": neg if can_localize else [],
        "confidence": max(0.0, min(1.0, confidence)),
        "notes": str(raw.get("notes", "")),
        "raw_qwen3vl_response": raw,
    }


def load_refine_maps(root: Path, manifest_dir: Path, entry: dict[str, Any]) -> tuple[Any, Any, Any]:
    import numpy as np

    index_path = resolve_path(root, entry["point_index_path"], base_dir=manifest_dir)
    confidence_path = resolve_path(root, entry["confidence_path"], base_dir=manifest_dir)
    source_path = resolve_path(root, entry["source_path"], base_dir=manifest_dir)
    if not index_path.exists() or not confidence_path.exists() or not source_path.exists():
        raise FileNotFoundError(f"Missing natural render refinement maps for view: {entry.get('view')}")
    return np.load(index_path), np.load(confidence_path), np.load(source_path)


def target_prefers_upper_structure(semantic: dict[str, Any] | None, row: dict[str, str]) -> bool:
    if semantic is None:
        return False
    words = " ".join(
        normalize_list(semantic.get("target_positive_parts", []), 20)
        + normalize_list(semantic.get("target_grounding_queries", []), 20)
        + [str(semantic.get("target_region_description", ""))]
    ).lower()
    keywords = ("handle", "loop", "ring", "hole", "inner rim", "top", "upper")
    if any(key in words for key in keywords):
        return True
    executor = row.get("executor", "").lower()
    return executor == "hook" and any(key in words for key in ("catch", "interlock", "enter"))


def upper_structure_mask(valid: Any, margin: int) -> Any:
    import numpy as np

    row_counts = valid.sum(axis=1)
    if row_counts.size == 0 or row_counts.max() <= 0:
        return valid.copy()
    # Rows with many foreground pixels are likely the main body. Anything
    # clearly above that body top is treated as an upper/detached structure.
    threshold = max(12, int(row_counts.max() * 0.22))
    dense_rows = np.where(row_counts >= threshold)[0]
    if dense_rows.size == 0:
        return valid.copy()
    body_top = int(dense_rows.min())
    cutoff = min(valid.shape[0] - 1, body_top + max(0, int(margin)))
    mask = np.zeros_like(valid, dtype=bool)
    mask[: cutoff + 1, :] = valid[: cutoff + 1, :]
    if mask.sum() < 4:
        return valid.copy()
    return mask


def snap_point_to_mask(point: list[int], mask: Any, radius: int) -> list[int] | None:
    import numpy as np

    x, y = int(point[0]), int(point[1])
    h, w = mask.shape
    x0, x1 = max(0, x - radius), min(w - 1, x + radius)
    y0, y1 = max(0, y - radius), min(h - 1, y + radius)
    ys, xs = np.where(mask[y0 : y1 + 1, x0 : x1 + 1])
    if xs.size == 0:
        return None
    xs = xs + x0
    ys = ys + y0
    d2 = (xs - x) * (xs - x) + (ys - y) * (ys - y)
    best = int(np.argmin(d2))
    return [int(xs[best]), int(ys[best])]


def bbox_from_mask(mask: Any, padding: int = 4) -> list[int] | None:
    import numpy as np

    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    h, w = mask.shape
    return [
        max(0, int(xs.min()) - padding),
        max(0, int(ys.min()) - padding),
        min(w - 1, int(xs.max()) + padding),
        min(h - 1, int(ys.max()) + padding),
    ]


def refine_localization(
    root: Path,
    manifest_dir: Path,
    entry: dict[str, Any],
    localization: dict[str, Any],
    semantic: dict[str, Any] | None,
    row: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    import numpy as np

    if args.no_refine_localization or not localization.get("can_localize"):
        return localization
    index_map, confidence, source = load_refine_maps(root, manifest_dir, entry)
    valid = (index_map >= 0) & (source > 0) & (confidence >= float(args.refine_min_confidence))
    if not np.any(valid):
        localization["refine"] = {"enabled": True, "status": "no_valid_foreground"}
        return localization

    prefer_upper = target_prefers_upper_structure(semantic, row)
    target_mask = upper_structure_mask(valid, args.upper_margin) if prefer_upper else valid
    if target_mask.sum() < 4:
        target_mask = valid

    refined_boxes: list[list[int]] = []
    for box in localization.get("boxes", []):
        x1, y1, x2, y2 = box
        local = np.zeros_like(target_mask, dtype=bool)
        local[y1 : y2 + 1, x1 : x2 + 1] = target_mask[y1 : y2 + 1, x1 : x2 + 1]
        if local.sum() < 4:
            local[y1 : y2 + 1, x1 : x2 + 1] = valid[y1 : y2 + 1, x1 : x2 + 1]
        refined = bbox_from_mask(local)
        if refined is not None:
            refined_boxes.append(refined)

    refined_points: list[list[int]] = []
    for point in localization.get("positive_points", []):
        snapped = snap_point_to_mask(point, target_mask, int(args.refine_snap_radius))
        if snapped is None:
            snapped = snap_point_to_mask(point, valid, int(args.refine_snap_radius))
        if snapped is not None and snapped not in refined_points:
            refined_points.append(snapped)

    refined_negative: list[list[int]] = []
    for point in localization.get("negative_points", []):
        # Negative points are often meant to exclude body/background, so keep
        # them close to any foreground instead of forcing them into upper parts.
        snapped = snap_point_to_mask(point, valid, int(args.refine_snap_radius))
        if snapped is not None and snapped not in refined_negative:
            refined_negative.append(snapped)

    before = {
        "boxes": localization.get("boxes", []),
        "positive_points": localization.get("positive_points", []),
        "negative_points": localization.get("negative_points", []),
    }
    localization["boxes"] = refined_boxes or localization.get("boxes", [])
    localization["positive_points"] = refined_points or localization.get("positive_points", [])
    localization["negative_points"] = refined_negative or localization.get("negative_points", [])
    localization["refine"] = {
        "enabled": True,
        "prefer_upper_structure": bool(prefer_upper),
        "min_confidence": float(args.refine_min_confidence),
        "snap_radius": int(args.refine_snap_radius),
        "upper_margin": int(args.upper_margin),
        "valid_pixels": int(valid.sum()),
        "target_pixels": int(target_mask.sum()),
        "before": before,
        "after": {
            "boxes": localization.get("boxes", []),
            "positive_points": localization.get("positive_points", []),
            "negative_points": localization.get("negative_points", []),
        },
    }
    localization["notes"] = (localization.get("notes", "") + " | foreground_refined").strip(" |")
    return localization


def dry_semantic(row: dict[str, str], view: str) -> dict[str, Any]:
    hint = category_hint(row)
    return normalize_semantic(
        {
            "view": view,
            "visible_object_parts": [],
            "target_positive_parts": hint.get("target", []),
            "target_grounding_queries": hint.get("target", []),
            "reject_negative_parts": hint.get("reject", []),
            "target_region_description": "dry run semantic placeholder",
            "mechanism_check": {"reason": "dry run"},
            "view_usefulness": "partial",
            "feasible": bool(hint.get("target", [])),
            "confidence": 0.0,
            "notes": "dry_run",
        },
        view,
    )


def dry_localization(view: str) -> dict[str, Any]:
    return normalize_localization(
        {
            "view": view,
            "can_localize": False,
            "target_region_description": "dry run localization placeholder",
            "boxes": [],
            "positive_points": [],
            "negative_points": [],
            "confidence": 0.0,
            "notes": "dry_run",
        },
        view,
        768,
    )


def draw_overlay(image_path: Path, output_path: Path, localization: dict[str, Any]) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for idx, box in enumerate(localization.get("boxes", []), start=1):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=(255, 70, 70), width=3)
        draw.text((x1 + 4, max(0, y1 - 14)), f"B{idx}", fill=(255, 70, 70), font=font)
    for idx, point in enumerate(localization.get("positive_points", []), start=1):
        x, y = point
        r = 5
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(40, 220, 110), outline=(255, 255, 255), width=1)
        draw.text((x + 7, y + 4), f"P{idx}", fill=(40, 220, 110), font=font)
    for idx, point in enumerate(localization.get("negative_points", []), start=1):
        x, y = point
        r = 5
        draw.line([x - r, y - r, x + r, y + r], fill=(255, 210, 80), width=2)
        draw.line([x - r, y + r, x + r, y - r], fill=(255, 210, 80), width=2)
        draw.text((x + 7, y + 4), f"N{idx}", fill=(255, 210, 80), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def combine_summary(row: dict[str, str], results: list[dict[str, Any]]) -> dict[str, Any]:
    target_counter: Counter[str] = Counter()
    query_counter: Counter[str] = Counter()
    reject_counter: Counter[str] = Counter()
    useful_counter: Counter[str] = Counter()
    feasible_views = 0
    localizable_views = 0
    confidence_sum = 0.0
    for result in results:
        sem = result.get("semantic", {})
        loc = result.get("localization", {})
        if sem.get("feasible"):
            feasible_views += 1
        if loc.get("can_localize"):
            localizable_views += 1
        confidence_sum += float(sem.get("confidence", 0.0))
        useful_counter[str(sem.get("view_usefulness", "unknown"))] += 1
        target_counter.update(sem.get("target_positive_parts", []))
        query_counter.update(sem.get("target_grounding_queries", []))
        reject_counter.update(sem.get("reject_negative_parts", []))
    n = max(1, len(results))
    return {
        "pilot_id": row["pilot_id"],
        "sample_id": row["sample_id"],
        "object_category": row.get("object_category", ""),
        "task": row.get("task", ""),
        "executor": row.get("executor", ""),
        "views": len(results),
        "feasible_views": int(feasible_views),
        "localizable_views": int(localizable_views),
        "mean_semantic_confidence": float(confidence_sum / n),
        "view_usefulness_counts": dict(useful_counter),
        "ranked_target_parts": [{"part": k, "votes": int(v)} for k, v in target_counter.most_common()],
        "ranked_grounding_queries": [{"query": k, "votes": int(v)} for k, v in query_counter.most_common()],
        "ranked_reject_parts": [{"part": k, "votes": int(v)} for k, v in reject_counter.most_common()],
        "notes": "Natural-render VLM probe only. No v2/v3 pipeline output is modified.",
    }


def write_index_html(output_dir: Path, summary: dict[str, Any], results: list[dict[str, Any]], root: Path) -> None:
    cards = []
    for result in results:
        overlay = result.get("overlay_path") or result.get("image_path", "")
        overlay_path = resolve_path(root, overlay) if overlay else None
        rel_overlay = Path(overlay_path).name if overlay_path and overlay_path.exists() else ""
        sem = result.get("semantic", {})
        loc = result.get("localization", {})
        cards.append(
            f"""
<section class="card">
  <h2>{html.escape(result.get('view', ''))}</h2>
  {f'<img src="{html.escape(rel_overlay)}" />' if rel_overlay else ''}
  <p><b>semantic:</b> feasible={sem.get('feasible')} usefulness={html.escape(str(sem.get('view_usefulness', '')))} confidence={sem.get('confidence')}</p>
  <p><b>target:</b> {html.escape(', '.join(sem.get('target_positive_parts', [])))}</p>
  <p><b>description:</b> {html.escape(str(sem.get('target_region_description', '')))}</p>
  <p><b>localize:</b> can_localize={loc.get('can_localize')} boxes={len(loc.get('boxes', []))} points={len(loc.get('positive_points', []))}</p>
</section>
"""
        )
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>Natural Render VLM Probe</title>
<style>
body {{ margin: 0; background: #0e1117; color: #e8edf5; font-family: Arial, sans-serif; }}
header {{ padding: 20px 28px; border-bottom: 1px solid #2d3748; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; padding: 18px; }}
.card {{ border: 1px solid #2d3748; background: #111722; padding: 14px; }}
img {{ width: 100%; height: auto; display: block; background: #080b10; }}
p {{ line-height: 1.5; color: #cbd5e1; }}
code {{ color: #93c5fd; }}
</style>
</head>
<body>
<header>
  <h1>Natural Render VLM Probe</h1>
  <p><code>{html.escape(summary.get('pilot_id', ''))}</code> | {html.escape(summary.get('object_category', ''))} | {html.escape(summary.get('task', ''))} | {html.escape(summary.get('executor', ''))}</p>
  <p>feasible views: {summary.get('feasible_views')} / {summary.get('views')}; localizable views: {summary.get('localizable_views')} / {summary.get('views')}</p>
</header>
<main class="grid">
{''.join(cards)}
</main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")


def run_for_row(root: Path, args: argparse.Namespace, cfg: dict[str, Any], row: dict[str, str], model: Any, processor: Any) -> dict[str, Any]:
    manifest_path = manifest_path_for_row(root, args, row)
    manifest = read_json(manifest_path)
    manifest_dir = manifest_path.parent
    entries = selected_view_entries(manifest, args)
    output_dir = resolve_path(root, args.output_root) / row["pilot_id"]
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory exists. Use --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = int(manifest.get("image_size", 768))
    results: list[dict[str, Any]] = []
    for entry in entries:
        view = str(entry["view"])
        image_value = entry.get(args.image_key) or entry.get("natural_render_path")
        image_path = resolve_path(root, image_value, base_dir=manifest_dir)
        if not image_path.exists():
            raise FileNotFoundError(f"Probe image not found: {image_path}")
        result: dict[str, Any] = {
            "view": view,
            "image_path": relative_to_dataset(root, image_path),
            "image_key": args.image_key,
        }
        semantic = None
        localization = None
        if args.validate_only:
            result["status"] = "validated"
        else:
            if args.probe_mode in ("semantic", "semantic_and_localize"):
                if args.dry_run:
                    semantic = dry_semantic(row, view)
                else:
                    raw = run_qwen_json(model, processor, image_path, semantic_prompt(row, view, image_size), cfg)
                    semantic = normalize_semantic(raw, view)
                write_json(output_dir / f"{view}_semantic.json", semantic, args.overwrite)
                result["semantic"] = semantic
            if args.probe_mode in ("localize", "semantic_and_localize"):
                if args.dry_run:
                    localization = dry_localization(view)
                else:
                    raw = run_qwen_json(model, processor, image_path, localization_prompt(row, view, image_size, semantic), cfg)
                    localization = normalize_localization(raw, view, image_size)
                    localization = refine_localization(root, manifest_dir, entry, localization, semantic, row, args)
                write_json(output_dir / f"{view}_localization.json", localization, args.overwrite)
                overlay_path = output_dir / f"{view}_probe_overlay.png"
                draw_overlay(image_path, overlay_path, localization)
                result["localization"] = localization
                result["overlay_path"] = relative_to_dataset(root, overlay_path)
        results.append(result)

    summary = combine_summary(row, results) if not args.validate_only else {
        "pilot_id": row["pilot_id"],
        "sample_id": row["sample_id"],
        "validated_views": len(results),
        "notes": "Validation only.",
    }
    summary["manifest"] = relative_to_dataset(root, manifest_path)
    summary["output_dir"] = relative_to_dataset(root, output_dir)
    summary["probe_mode"] = args.probe_mode
    summary["image_key"] = args.image_key
    write_json(output_dir / "combined_probe_summary.json", summary, args.overwrite)
    if not args.validate_only:
        write_json(output_dir / "view_probe_results.json", {"summary": summary, "views": results}, args.overwrite)
        write_index_html(output_dir, summary, results, root)
    return summary


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    cfg = load_yaml(resolve_path(root, args.config))
    row = selected_row(root, args)
    model = None
    processor = None
    if not args.validate_only and not args.dry_run:
        model, processor = load_qwen_model(cfg, root)
    summary = run_for_row(root, args, cfg, row, model, processor)
    print(json.dumps({"validate_only": args.validate_only, "dry_run": args.dry_run, "summary": summary}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
