#!/usr/bin/env python3
"""Run Qwen3-VL + SAM2 on Multi-EE VLM pilot renders.

Pipeline:
  1. Qwen3-VL reads one rendered view at a time and emits box/point prompts.
  2. SAM2 converts those prompts into a 2D binary mask.
  3. Masks are saved in the same layout expected by build_vlm_pilot_candidates.py.

This script is written for the remote GPU server workflow:
local Codex edits -> GitHub push -> remote server git pull -> run this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from path_utils import resolve_portable_path


EXECUTOR_DEFINITIONS = {
    "gripper": (
        "Two-finger parallel gripper. Mark only areas that can be pinched from two sides "
        "and stably loaded, such as handle outer sides, narrow stems, and graspable edges. "
        "Do not mark broad flat centers, hole interiors, tiny unstable buttons, or surfaces "
        "that are only suitable for suction."
    ),
    "suction": (
        "Suction cup. Mark only locally flat, low-curvature, normal-consistent, sufficiently "
        "large surfaces such as panels, box tops, drawer fronts, or dish centers. Do not mark "
        "holes, edges, rods, handles, rings, perforated regions, or high-curvature areas."
    ),
    "hook": (
        "Hook end-effector. Mark only structures that can be hooked or hung, such as inner "
        "handle holes, rings, hole boundaries, and bag handles. Do not mark ordinary flat "
        "surfaces, buttons, or smooth regions that cannot mechanically catch a hook."
    ),
    "dexterous_hand": (
        "Dexterous multi-finger hand. Mark only regions where a human-like hand can stably "
        "wrap, grasp, press, rotate, or finely manipulate. Do not mark every touchable surface."
    ),
}

TASK_DEFINITIONS = {
    "pick_up": "Pick up the object; stable lifting contact matters.",
    "lift_carry": "Lift and carry the object; load-bearing stable contact matters.",
    "open_pull": "Open or pull an articulated part; usable pull handles, rings, or panels matter.",
    "press_push": "Press or push; buttons, switches, and pushable panels matter.",
}

DEFAULT_VIEWS = [
    "yaw000_elev20",
    "yaw045_elev20",
    "yaw090_elev20",
    "yaw135_elev20",
    "yaw180_elev20",
    "yaw225_elev20",
    "yaw270_elev20",
    "yaw315_elev20",
]


@dataclass
class QwenPrompt:
    feasible: bool
    boxes: list[list[int]]
    positive_points: list[list[int]]
    negative_points: list[list[int]]
    confidence: float
    notes: str
    raw: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3-VL + SAM2 on VLM pilot renders.")
    parser.add_argument("--dataset-root", default=".", help="Dataset root directory.")
    parser.add_argument(
        "--config",
        default="configs/qwen3vl_sam2_pilot.yaml",
        help="YAML config relative to dataset root.",
    )
    parser.add_argument("--pilot-id", default=None, help="Run only one pilot_id.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of pilot rows.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing masks/responses.")
    parser.add_argument("--dry-run", action="store_true", help="Create empty masks without loading Qwen/SAM2.")
    parser.add_argument("--validate-only", action="store_true", help="Validate config, pilot rows, and render files only.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def resolve_path(root: Path, value: str | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
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


def clamp_int(value: Any, low: int, high: int) -> int | None:
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(low, min(high, number))


def maybe_scaled(value: Any, scale: float) -> Any:
    try:
        return float(value) * scale
    except (TypeError, ValueError):
        return value


def normalize_point_list(items: Any, image_size: int, max_count: int) -> list[list[int]]:
    out: list[list[int]] = []
    if isinstance(items, dict):
        items = [items]
    elif isinstance(items, (list, tuple)) and len(items) >= 2 and not isinstance(items[0], (list, tuple, dict)):
        items = [items]
    if not isinstance(items, list):
        return out
    for item in items:
        if isinstance(item, dict):
            item = [item.get("x"), item.get("y")]
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        scale = image_size - 1 if all(isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0 for v in item[:2]) else 1.0
        x = clamp_int(maybe_scaled(item[0], scale), 0, image_size - 1)
        y = clamp_int(maybe_scaled(item[1], scale), 0, image_size - 1)
        if x is not None and y is not None:
            out.append([x, y])
        if len(out) >= max_count:
            break
    return out


def normalize_boxes(items: Any, image_size: int, max_count: int, min_area: int) -> list[list[int]]:
    out: list[list[int]] = []
    if isinstance(items, dict):
        items = [items]
    elif isinstance(items, (list, tuple)) and len(items) >= 4 and not isinstance(items[0], (list, tuple, dict)):
        items = [items]
    if not isinstance(items, list):
        return out
    for item in items:
        if isinstance(item, dict):
            if "box" in item or "bbox" in item:
                item = item.get("box", item.get("bbox", []))
            else:
                item = [item.get("x1"), item.get("y1"), item.get("x2"), item.get("y2")]
        if not isinstance(item, (list, tuple)) or len(item) < 4:
            continue
        scale = image_size - 1 if all(isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0 for v in item[:4]) else 1.0
        x1 = clamp_int(maybe_scaled(item[0], scale), 0, image_size - 1)
        y1 = clamp_int(maybe_scaled(item[1], scale), 0, image_size - 1)
        x2 = clamp_int(maybe_scaled(item[2], scale), 0, image_size - 1)
        y2 = clamp_int(maybe_scaled(item[3], scale), 0, image_size - 1)
        if None in (x1, y1, x2, y2):
            continue
        xa, xb = sorted([int(x1), int(x2)])
        ya, yb = sorted([int(y1), int(y2)])
        if (xb - xa) * (yb - ya) >= min_area:
            out.append([xa, ya, xb, yb])
        if len(out) >= max_count:
            break
    return out


def extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def build_qwen_prompt(row: dict[str, str], view: str, image_size: int, cfg: dict[str, Any]) -> str:
    executor = row["executor"]
    task = row["task"]
    seg_cfg = cfg.get("segmentation", {})
    return f"""
You are helping build a research dataset for object-level 3D affordance grounding.
You see one rendered point-cloud view of the object. Identify candidate image prompts
for SAM2 to segment the area usable by the current end-effector and task.
The image is a sparse rendered point cloud: colored pixels are object points, and
the dark background is empty space. Infer object parts from the visible geometry,
outline, protrusions, holes, handles, rings, and panel boundaries.

Object category: {row['object_category']}
Task: {task}
Task definition: {TASK_DEFINITIONS.get(task, task)}
End-effector: {executor}
End-effector definition: {EXECUTOR_DEFINITIONS[executor]}
View name: {view}
Image size: {image_size} x {image_size}
Manual-review issue: {row.get('issue_type', '')} / {row.get('decision', '')}
Pilot reason: {row.get('pilot_reason', '')}

Return strict JSON only:
{{
  "view": "{view}",
  "visible_object_parts": ["part name"],
  "target_region_description": "short description of the usable region",
  "feasible": true,
  "confidence": 0.0,
  "boxes": [[x1, y1, x2, y2]],
  "positive_points": [[x, y]],
  "negative_points": [[x, y]],
  "notes": "short reason"
}}

Rules:
1. Coordinates are pixels in [0, {image_size - 1}]. Do not use normalized coordinates unless unavoidable.
2. If feasible=true, provide at least one box or one positive point.
3. Every box and positive point must lie on, or tightly enclose, colored object pixels. Never place them in the dark background.
4. For handle/ring/hookable regions, points must be on the visible colored handle/ring pixels, not in the empty interior or nearby background.
5. Keep boxes tight around the target part only. Do not include broad object body surfaces unless they are the target region.
6. Provide at most {seg_cfg.get('max_boxes_per_view', 3)} boxes.
7. Provide at most {seg_cfg.get('max_positive_points_per_view', 8)} positive points.
8. Provide at most {seg_cfg.get('max_negative_points_per_view', 8)} negative points.
9. If the usable region is invisible or physically infeasible, set feasible=false and use empty boxes/points.
10. Be conservative. Do not mark ordinary contact surfaces as positive affordance.
11. For hook, only mark visible hookable holes/rings/inner handle boundaries.
12. For suction, avoid edges, handles, holes, and high-curvature regions.
13. Use visible_object_parts and target_region_description to explain the part-level reasoning before coordinates.
"""


def resolve_qwen_pretrained_source(root: Path, qwen_cfg: dict[str, Any]) -> str:
    model_path = qwen_cfg.get("model_path")
    if model_path:
        path = Path(str(model_path))
        resolved = path if path.is_absolute() else root / path
        if not resolved.exists():
            raise FileNotFoundError(
                f"Qwen3-VL local model_path does not exist: {resolved}. "
                "Download/transfer the model directory first, or use qwen3vl.model_id when the server has network access."
            )
        if not (resolved / "config.json").exists():
            raise FileNotFoundError(
                f"Qwen3-VL local model_path is missing config.json: {resolved}. "
                "Point model_path to the Hugging Face model directory or snapshot directory that contains config.json."
            )
        return str(resolved)
    return str(qwen_cfg.get("model_id", "Qwen/Qwen3-VL-8B-Instruct"))


def load_qwen_model(cfg: dict[str, Any], root: Path) -> tuple[Any, Any]:
    qwen_cfg = cfg.get("qwen3vl", {})
    model_source = resolve_qwen_pretrained_source(root, qwen_cfg)
    try:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except Exception as exc:
        raise RuntimeError(
            "Qwen3-VL dependencies are missing. Install latest transformers, torch, accelerate, and pillow."
        ) from exc

    load_kwargs: dict[str, Any] = {
        "dtype": qwen_cfg.get("dtype", "auto"),
    }
    shared_kwargs: dict[str, Any] = {}
    if qwen_cfg.get("cache_dir"):
        shared_kwargs["cache_dir"] = str(resolve_path(root, qwen_cfg["cache_dir"]))
    if qwen_cfg.get("revision"):
        shared_kwargs["revision"] = qwen_cfg["revision"]
    if qwen_cfg.get("trust_remote_code") is not None:
        shared_kwargs["trust_remote_code"] = bool(qwen_cfg["trust_remote_code"])
    if qwen_cfg.get("local_files_only") is not None:
        shared_kwargs["local_files_only"] = bool(qwen_cfg["local_files_only"])
    load_kwargs.update(shared_kwargs)
    device_map = qwen_cfg.get("device_map", "auto")
    if isinstance(device_map, str) and device_map.startswith("cuda:"):
        load_kwargs["device_map"] = {"": device_map}
    else:
        load_kwargs["device_map"] = device_map
    attn_impl = qwen_cfg.get("attn_implementation")
    if attn_impl:
        load_kwargs["attn_implementation"] = attn_impl

    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_source, **load_kwargs)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load Qwen3-VL. If the remote server cannot access Hugging Face, "
            "download/transfer Qwen/Qwen3-VL-8B-Instruct to a local directory, set "
            "qwen3vl.model_path to that directory, and set qwen3vl.local_files_only: true."
        ) from exc

    processor_kwargs: dict[str, Any] = dict(shared_kwargs)
    if qwen_cfg.get("min_pixels") is not None:
        processor_kwargs["min_pixels"] = int(qwen_cfg["min_pixels"])
    if qwen_cfg.get("max_pixels") is not None:
        processor_kwargs["max_pixels"] = int(qwen_cfg["max_pixels"])
    try:
        processor = AutoProcessor.from_pretrained(model_source, **processor_kwargs)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load Qwen3-VL processor. Make sure the local model directory contains "
            "processor/tokenizer files and config.json, or disable local_files_only when the server has network access."
        ) from exc
    model.eval()
    torch.cuda.empty_cache()
    return model, processor


def qwen_device(model: Any) -> Any:
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def run_qwen_for_view(model: Any, processor: Any, image_path: Path, prompt: str, cfg: dict[str, Any]) -> dict[str, Any]:
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


def normalize_qwen_output(raw: dict[str, Any], image_size: int, cfg: dict[str, Any]) -> QwenPrompt:
    seg_cfg = cfg.get("segmentation", {})
    boxes = normalize_boxes(
        raw.get("boxes", raw.get("bboxes", raw.get("box", raw.get("bbox", [])))),
        image_size,
        int(seg_cfg.get("max_boxes_per_view", 3)),
        int(seg_cfg.get("min_box_area", 16)),
    )
    pos = normalize_point_list(
        raw.get("positive_points", raw.get("positive_point", raw.get("points", raw.get("point", [])))),
        image_size,
        int(seg_cfg.get("max_positive_points_per_view", 8)),
    )
    neg = normalize_point_list(
        raw.get("negative_points", raw.get("negative_point", [])),
        image_size,
        int(seg_cfg.get("max_negative_points_per_view", 8)),
    )
    feasible = bool(raw.get("feasible", bool(boxes or pos))) and bool(boxes or pos)
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return QwenPrompt(
        feasible=feasible,
        boxes=boxes,
        positive_points=pos,
        negative_points=neg,
        confidence=max(0.0, min(1.0, confidence)),
        notes=str(raw.get("notes", "")),
        raw=raw,
    )


def prompt_to_dict(prompt: QwenPrompt) -> dict[str, Any]:
    return {
        "feasible": prompt.feasible,
        "boxes": prompt.boxes,
        "positive_points": prompt.positive_points,
        "negative_points": prompt.negative_points,
        "confidence": prompt.confidence,
        "notes": prompt.notes,
    }


def nearest_foreground_point(
    point: list[int],
    foreground: np.ndarray,
    max_distance: int,
) -> list[int] | None:
    h, w = foreground.shape
    x = max(0, min(w - 1, int(point[0])))
    y = max(0, min(h - 1, int(point[1])))
    if foreground[y, x]:
        return [x, y]

    radius = max(0, int(max_distance))
    y0, y1 = max(0, y - radius), min(h - 1, y + radius)
    x0, x1 = max(0, x - radius), min(w - 1, x + radius)
    candidates = np.argwhere(foreground[y0 : y1 + 1, x0 : x1 + 1])
    if candidates.size == 0:
        return None
    candidates[:, 0] += y0
    candidates[:, 1] += x0
    dist2 = (candidates[:, 1] - x) ** 2 + (candidates[:, 0] - y) ** 2
    best_idx = int(np.argmin(dist2))
    if float(dist2[best_idx]) > float(radius * radius):
        return None
    return [int(candidates[best_idx, 1]), int(candidates[best_idx, 0])]


def dilate_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    if radius <= 0:
        return mask.astype(bool)
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    radius2 = radius * radius
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius2:
                continue
            src_y0 = max(0, -dy)
            src_y1 = min(h, h - dy)
            src_x0 = max(0, -dx)
            src_x1 = min(w, w - dx)
            dst_y0 = max(0, dy)
            dst_y1 = min(h, h + dy)
            dst_x0 = max(0, dx)
            dst_x1 = min(w, w + dx)
            out[dst_y0:dst_y1, dst_x0:dst_x1] |= mask[src_y0:src_y1, src_x0:src_x1]
    return out


def refine_box_to_foreground(
    box: list[int],
    foreground: np.ndarray,
    max_distance: int,
    padding: int,
    min_pixels: int,
) -> list[int] | None:
    h, w = foreground.shape
    x1, y1, x2, y2 = box
    x1, x2 = sorted([max(0, min(w - 1, int(x1))), max(0, min(w - 1, int(x2)))])
    y1, y2 = sorted([max(0, min(h - 1, int(y1))), max(0, min(h - 1, int(y2)))])

    ys, xs = np.where(foreground[y1 : y2 + 1, x1 : x2 + 1])
    if len(xs) >= min_pixels:
        xs = xs + x1
        ys = ys + y1
    else:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        radius = max(0, int(max_distance))
        sx0, sx1 = max(0, cx - radius), min(w - 1, cx + radius)
        sy0, sy1 = max(0, cy - radius), min(h - 1, cy + radius)
        ys, xs = np.where(foreground[sy0 : sy1 + 1, sx0 : sx1 + 1])
        if len(xs) < min_pixels:
            return None
        xs = xs + sx0
        ys = ys + sy0

    pad = max(0, int(padding))
    rx1 = max(0, int(xs.min()) - pad)
    ry1 = max(0, int(ys.min()) - pad)
    rx2 = min(w - 1, int(xs.max()) + pad)
    ry2 = min(h - 1, int(ys.max()) + pad)
    if (rx2 - rx1) * (ry2 - ry1) <= 0:
        return None
    return [rx1, ry1, rx2, ry2]


def refine_prompt_to_foreground(prompt: QwenPrompt, index_map: np.ndarray, cfg: dict[str, Any]) -> QwenPrompt:
    seg_cfg = cfg.get("segmentation", {})
    if not bool(seg_cfg.get("foreground_prompt_refine", True)):
        return prompt

    index_foreground = index_map >= 0
    if index_foreground.ndim != 2 or not np.any(index_foreground):
        return prompt

    max_point_dist = int(seg_cfg.get("max_point_snap_distance", 24))
    max_box_dist = int(seg_cfg.get("max_box_snap_distance", 48))
    box_padding = int(seg_cfg.get("box_padding", 4))
    min_box_pixels = int(seg_cfg.get("min_box_foreground_pixels", 4))
    snap_negative = bool(seg_cfg.get("snap_negative_points", False))
    snap_radius = int(seg_cfg.get("foreground_snap_radius", 0))
    foreground = dilate_bool(index_foreground, snap_radius)

    boxes: list[list[int]] = []
    for box in prompt.boxes:
        refined = refine_box_to_foreground(box, foreground, max_box_dist, box_padding, min_box_pixels)
        if refined is not None and refined not in boxes:
            boxes.append(refined)

    positive_points: list[list[int]] = []
    for point in prompt.positive_points:
        refined = nearest_foreground_point(point, foreground, max_point_dist)
        if refined is not None and refined not in positive_points:
            positive_points.append(refined)

    negative_points: list[list[int]] = []
    for point in prompt.negative_points:
        if snap_negative:
            refined = nearest_foreground_point(point, foreground, max_point_dist)
            if refined is not None and refined not in negative_points:
                negative_points.append(refined)
        else:
            x, y = int(point[0]), int(point[1])
            if 0 <= y < index_foreground.shape[0] and 0 <= x < index_foreground.shape[1] and index_foreground[y, x]:
                negative_points.append([x, y])

    feasible = prompt.feasible and bool(boxes or positive_points)
    return QwenPrompt(
        feasible=feasible,
        boxes=boxes,
        positive_points=positive_points,
        negative_points=negative_points,
        confidence=prompt.confidence,
        notes=prompt.notes,
        raw=prompt.raw,
    )


def prefer_hook_points_over_boxes(prompt: QwenPrompt, executor: str, cfg: dict[str, Any]) -> QwenPrompt:
    seg_cfg = cfg.get("segmentation", {})
    if executor != "hook" or not bool(seg_cfg.get("hook_prefer_points_over_boxes", True)):
        return prompt
    if not prompt.positive_points:
        return prompt
    return QwenPrompt(
        feasible=prompt.feasible,
        boxes=[],
        positive_points=prompt.positive_points,
        negative_points=prompt.negative_points,
        confidence=prompt.confidence,
        notes=prompt.notes,
        raw=prompt.raw,
    )


def load_sam2_predictor(cfg: dict[str, Any], root: Path) -> Any:
    sam_cfg = cfg.get("sam2", {})
    try:
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except Exception as exc:
        raise RuntimeError("SAM2 is missing. Install facebookresearch/sam2 in the remote environment.") from exc

    hf_model_id = sam_cfg.get("hf_model_id")
    if hf_model_id:
        predictor = SAM2ImagePredictor.from_pretrained(hf_model_id, device=sam_cfg.get("device", "cuda:0"))
    else:
        from sam2.build_sam import build_sam2

        checkpoint = sam_cfg.get("checkpoint")
        model_cfg = sam_cfg.get("model_cfg")
        if not checkpoint or not model_cfg:
            raise ValueError("SAM2 requires either hf_model_id or checkpoint + model_cfg.")
        checkpoint_path = resolve_path(root, str(checkpoint))
        assert checkpoint_path is not None
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"SAM2 checkpoint not found: {checkpoint_path}. "
                "Relative sam2.checkpoint paths are resolved from --dataset-root."
            )
        predictor = SAM2ImagePredictor(
            build_sam2(model_cfg, str(checkpoint_path), device=sam_cfg.get("device", "cuda:0"), mode="eval")
        )
    torch.cuda.empty_cache()
    return predictor


def choose_best_mask(masks: np.ndarray, scores: np.ndarray | None) -> np.ndarray:
    if masks.ndim == 2:
        return masks > 0
    if masks.ndim != 3:
        raise ValueError(f"Unexpected SAM2 mask shape: {masks.shape}")
    if scores is None or len(scores) != masks.shape[0]:
        return masks[0] > 0
    return masks[int(np.argmax(scores))] > 0


def sam2_segment(image: np.ndarray, prompt: QwenPrompt, predictor: Any, cfg: dict[str, Any]) -> np.ndarray:
    import torch

    sam_cfg = cfg.get("sam2", {})
    device_type = "cuda" if str(sam_cfg.get("device", "cuda:0")).startswith("cuda") else "cpu"
    dtype = torch.bfloat16 if sam_cfg.get("dtype", "bfloat16") == "bfloat16" else torch.float32
    multimask = bool(sam_cfg.get("multimask_output", True))
    combined = np.zeros(image.shape[:2], dtype=bool)
    if not prompt.feasible or (not prompt.boxes and not prompt.positive_points):
        return combined

    point_coords = None
    point_labels = None
    if prompt.positive_points or prompt.negative_points:
        coords = prompt.positive_points + prompt.negative_points
        labels = [1] * len(prompt.positive_points) + [0] * len(prompt.negative_points)
        point_coords = np.asarray(coords, dtype=np.float32)
        point_labels = np.asarray(labels, dtype=np.int32)

    autocast_enabled = device_type == "cuda"
    with torch.inference_mode(), torch.autocast(device_type, dtype=dtype, enabled=autocast_enabled):
        predictor.set_image(image)
        if prompt.boxes:
            for box in prompt.boxes:
                masks, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=np.asarray(box, dtype=np.float32),
                    multimask_output=multimask,
                )
                combined |= choose_best_mask(np.asarray(masks), np.asarray(scores) if scores is not None else None)
        elif point_coords is not None:
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=multimask,
            )
            combined |= choose_best_mask(np.asarray(masks), np.asarray(scores) if scores is not None else None)
    return combined


def save_binary_mask(mask: np.ndarray, npy_path: Path, png_path: Path, overwrite: bool) -> None:
    if (npy_path.exists() or png_path.exists()) and not overwrite:
        raise FileExistsError(f"Mask exists. Use --overwrite: {npy_path}")
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    arr = (mask > 0).astype(np.uint8)
    np.save(npy_path, arr)
    image = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    image.save(png_path)


def save_empty_mask(image_size: int, npy_path: Path, png_path: Path, overwrite: bool) -> None:
    save_binary_mask(np.zeros((image_size, image_size), dtype=np.uint8), npy_path, png_path, overwrite)


def maybe_draw_prompt_overlay(image: Image.Image, prompt: QwenPrompt, output_path: Path) -> None:
    overlay = image.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    for box in prompt.boxes:
        draw.rectangle(box, outline=(255, 60, 60), width=2)
    for x, y in prompt.positive_points:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(40, 220, 80))
    for x, y in prompt.negative_points:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(60, 120, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    cfg_path = resolve_path(root, args.config)
    assert cfg_path is not None
    cfg = load_yaml(cfg_path)
    if args.overwrite:
        cfg.setdefault("runtime", {})["overwrite"] = True
    if args.dry_run:
        cfg.setdefault("runtime", {})["dry_run"] = True

    pilot_cfg = cfg.get("pilot", {})
    pilot_csv = resolve_path(root, pilot_cfg.get("pilot_csv", "processed/metadata/vlm_pilot_samples_v0_1.csv"))
    renders_root = resolve_path(root, pilot_cfg.get("renders_root", "processed/vlm_pilot/renders"))
    output_mask_root = resolve_path(root, pilot_cfg.get("output_mask_root", "processed/vlm_pilot/vlm_2d_masks"))
    response_root = resolve_path(root, pilot_cfg.get("output_response_root", "processed/vlm_pilot/qwen3vl_sam2_responses"))
    assert pilot_csv and renders_root and output_mask_root and response_root
    views = pilot_cfg.get("views", DEFAULT_VIEWS)
    overwrite = bool(cfg.get("runtime", {}).get("overwrite", False))
    dry_run = bool(cfg.get("runtime", {}).get("dry_run", False))

    rows = read_csv(pilot_csv)
    if args.pilot_id:
        rows = [row for row in rows if row.get("pilot_id") == args.pilot_id]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No pilot rows selected.")

    if args.validate_only:
        checked_views = 0
        for row in rows:
            sample_id = row["sample_id"]
            manifest = read_json(renders_root / sample_id / "view_manifest.json")
            manifest_views = {entry["view"]: entry for entry in manifest.get("views", [])}
            for view in views:
                if view not in manifest_views:
                    raise KeyError(f"View {view} missing from manifest for {sample_id}")
                image_path = resolve_portable_path(root, manifest_views[view].get("render_path", ""), renders_root / sample_id)
                if not image_path.exists():
                    raise FileNotFoundError(f"Render image not found: {image_path}")
                checked_views += 1
        print(json.dumps({"validate_only": True, "pilot_rows": len(rows), "views_checked": checked_views}, indent=2))
        return 0

    model = processor = predictor = None
    if not dry_run:
        model, processor = load_qwen_model(cfg, root)
        predictor = load_sam2_predictor(cfg, root)

    summary: list[dict[str, Any]] = []
    for row in rows:
        pilot_id = row["pilot_id"]
        sample_id = row["sample_id"]
        executor = row["executor"]
        manifest = read_json(renders_root / sample_id / "view_manifest.json")
        image_size = int(manifest.get("image_size", pilot_cfg.get("image_size", 384)))
        manifest_views = {entry["view"]: entry for entry in manifest.get("views", [])}
        pilot_response_dir = response_root / pilot_id
        mask_dir = output_mask_root / sample_id / executor
        view_results: list[dict[str, Any]] = []

        for view in views:
            if view not in manifest_views:
                raise KeyError(f"View {view} missing from manifest for {sample_id}")
            image_path = resolve_portable_path(root, manifest_views[view]["render_path"], renders_root / sample_id)
            npy_path = mask_dir / f"{view}.npy"
            png_path = mask_dir / f"{view}.png"
            response_path = pilot_response_dir / f"{view}_qwen3vl.json"
            if npy_path.exists() and response_path.exists() and not overwrite:
                view_results.append({"view": view, "status": "skipped", "mask_path": str(npy_path)})
                continue

            if dry_run:
                raw = {"view": view, "feasible": False, "confidence": 0.0, "boxes": [], "positive_points": [], "negative_points": [], "notes": "dry-run"}
                prompt = normalize_qwen_output(raw, image_size, cfg)
                prompt_before_refine = prompt_to_dict(prompt)
                save_empty_mask(image_size, npy_path, png_path, overwrite=True)
            else:
                text_prompt = build_qwen_prompt(row, view, image_size, cfg)
                raw = run_qwen_for_view(model, processor, image_path, text_prompt, cfg)
                prompt = normalize_qwen_output(raw, image_size, cfg)
                prompt_before_refine = prompt_to_dict(prompt)
                index_map_path = resolve_portable_path(root, manifest_views[view]["point_index_path"], renders_root / sample_id)
                index_map = np.load(index_map_path)
                prompt = refine_prompt_to_foreground(prompt, index_map, cfg)
                prompt = prefer_hook_points_over_boxes(prompt, executor, cfg)
                image_pil = Image.open(image_path).convert("RGB")
                image_np = np.asarray(image_pil)
                mask = sam2_segment(image_np, prompt, predictor, cfg)
                save_binary_mask(mask, npy_path, png_path, overwrite=overwrite)
                if cfg.get("segmentation", {}).get("save_prompt_overlay", False):
                    maybe_draw_prompt_overlay(image_pil, prompt, pilot_response_dir / f"{view}_prompt_overlay.png")

            write_json(
                response_path,
                {
                    "pilot_id": pilot_id,
                    "sample_id": sample_id,
                    "executor": executor,
                    "view": view,
                    "prompt_before_foreground_refine": prompt_before_refine,
                    "normalized_prompt": {
                        "feasible": prompt.feasible,
                        "boxes": prompt.boxes,
                        "positive_points": prompt.positive_points,
                        "negative_points": prompt.negative_points,
                        "confidence": prompt.confidence,
                        "notes": prompt.notes,
                    },
                    "raw_qwen3vl_response": prompt.raw,
                    "mask_path": str(npy_path),
                },
                overwrite=True,
            )
            view_results.append(
                {
                    "view": view,
                    "status": "done",
                    "mask_path": str(npy_path),
                    "positive_pixels": int(np.load(npy_path).sum()),
                    "feasible": prompt.feasible,
                    "confidence": prompt.confidence,
                }
            )

        combined_response = {
            "pilot_id": pilot_id,
            "sample_id": sample_id,
            "object_category": row.get("object_category", ""),
            "task": row.get("task", ""),
            "executor": executor,
            "views": view_results,
        }
        write_json(pilot_response_dir / "combined_response.json", combined_response, overwrite=True)
        summary.append(combined_response)

    response_root.mkdir(parents=True, exist_ok=True)
    write_json(
        response_root / "run_summary.json",
        {"rows": summary, "dry_run": dry_run, "config": str(cfg_path)},
        overwrite=True,
    )
    print(json.dumps({"pilot_rows": len(summary), "dry_run": dry_run, "response_root": str(response_root)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
