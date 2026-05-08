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

DEFAULT_VIEWS = ["front", "back", "left", "right", "top", "iso"]


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


def normalize_point_list(items: Any, image_size: int, max_count: int) -> list[list[int]]:
    out: list[list[int]] = []
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        x = clamp_int(item[0], 0, image_size - 1)
        y = clamp_int(item[1], 0, image_size - 1)
        if x is not None and y is not None:
            out.append([x, y])
        if len(out) >= max_count:
            break
    return out


def normalize_boxes(items: Any, image_size: int, max_count: int, min_area: int) -> list[list[int]]:
    out: list[list[int]] = []
    if not isinstance(items, list):
        return out
    for item in items:
        if isinstance(item, dict):
            item = item.get("box", item.get("bbox", []))
        if not isinstance(item, (list, tuple)) or len(item) < 4:
            continue
        x1 = clamp_int(item[0], 0, image_size - 1)
        y1 = clamp_int(item[1], 0, image_size - 1)
        x2 = clamp_int(item[2], 0, image_size - 1)
        y2 = clamp_int(item[3], 0, image_size - 1)
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
  "feasible": true,
  "confidence": 0.0,
  "boxes": [[x1, y1, x2, y2]],
  "positive_points": [[x, y]],
  "negative_points": [[x, y]],
  "notes": "short reason"
}}

Rules:
1. Coordinates are pixels in [0, {image_size - 1}].
2. Provide at most {seg_cfg.get('max_boxes_per_view', 3)} boxes.
3. Provide at most {seg_cfg.get('max_positive_points_per_view', 8)} positive points.
4. Provide at most {seg_cfg.get('max_negative_points_per_view', 8)} negative points.
5. If the usable region is invisible or physically infeasible, set feasible=false and use empty boxes/points.
6. Be conservative. Do not mark ordinary contact surfaces as positive affordance.
7. For hook, only mark visible hookable holes/rings/inner handle boundaries.
8. For suction, avoid edges, handles, holes, and high-curvature regions.
"""


def load_qwen_model(cfg: dict[str, Any]) -> tuple[Any, Any]:
    qwen_cfg = cfg.get("qwen3vl", {})
    model_id = qwen_cfg.get("model_id", "Qwen/Qwen3-VL-8B-Instruct")
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
    device_map = qwen_cfg.get("device_map", "auto")
    if isinstance(device_map, str) and device_map.startswith("cuda:"):
        load_kwargs["device_map"] = {"": device_map}
    else:
        load_kwargs["device_map"] = device_map
    attn_impl = qwen_cfg.get("attn_implementation")
    if attn_impl:
        load_kwargs["attn_implementation"] = attn_impl

    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
    processor_kwargs = {}
    if qwen_cfg.get("min_pixels") is not None:
        processor_kwargs["min_pixels"] = int(qwen_cfg["min_pixels"])
    if qwen_cfg.get("max_pixels") is not None:
        processor_kwargs["max_pixels"] = int(qwen_cfg["max_pixels"])
    processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)
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
    return extract_json(text)


def normalize_qwen_output(raw: dict[str, Any], image_size: int, cfg: dict[str, Any]) -> QwenPrompt:
    seg_cfg = cfg.get("segmentation", {})
    boxes = normalize_boxes(
        raw.get("boxes", raw.get("bboxes", [])),
        image_size,
        int(seg_cfg.get("max_boxes_per_view", 3)),
        int(seg_cfg.get("min_box_area", 16)),
    )
    pos = normalize_point_list(
        raw.get("positive_points", raw.get("points", [])),
        image_size,
        int(seg_cfg.get("max_positive_points_per_view", 8)),
    )
    neg = normalize_point_list(
        raw.get("negative_points", []),
        image_size,
        int(seg_cfg.get("max_negative_points_per_view", 8)),
    )
    feasible = bool(raw.get("feasible", bool(boxes or pos)))
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


def load_sam2_predictor(cfg: dict[str, Any]) -> Any:
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
        predictor = SAM2ImagePredictor(
            build_sam2(model_cfg, checkpoint, device=sam_cfg.get("device", "cuda:0"), mode="eval")
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
                image_path = Path(manifest_views[view].get("render_path", ""))
                if not image_path.exists():
                    raise FileNotFoundError(f"Render image not found: {image_path}")
                checked_views += 1
        print(json.dumps({"validate_only": True, "pilot_rows": len(rows), "views_checked": checked_views}, indent=2))
        return 0

    model = processor = predictor = None
    if not dry_run:
        model, processor = load_qwen_model(cfg)
        predictor = load_sam2_predictor(cfg)

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
            image_path = Path(manifest_views[view]["render_path"])
            npy_path = mask_dir / f"{view}.npy"
            png_path = mask_dir / f"{view}.png"
            response_path = pilot_response_dir / f"{view}_qwen3vl.json"
            if npy_path.exists() and response_path.exists() and not overwrite:
                view_results.append({"view": view, "status": "skipped", "mask_path": str(npy_path)})
                continue

            if dry_run:
                raw = {"view": view, "feasible": False, "confidence": 0.0, "boxes": [], "positive_points": [], "negative_points": [], "notes": "dry-run"}
                prompt = normalize_qwen_output(raw, image_size, cfg)
                save_empty_mask(image_size, npy_path, png_path, overwrite=True)
            else:
                text_prompt = build_qwen_prompt(row, view, image_size, cfg)
                raw = run_qwen_for_view(model, processor, image_path, text_prompt, cfg)
                prompt = normalize_qwen_output(raw, image_size, cfg)
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
