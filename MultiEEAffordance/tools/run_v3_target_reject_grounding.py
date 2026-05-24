#!/usr/bin/env python3
"""Ground v3 semantic target/reject parts into rough 2D boxes and points.

This script asks Qwen3-VL to localize two different roles on clean renders:

- target: candidate-positive functional parts
- reject: hard-veto parts that must not be included in positive candidates

The output is not a final mask. It is a structured 2D seed proposal that is
projected back to 3D by project_v3_grounding_to_3d.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

from PIL import Image

from path_utils import resolve_portable_path
from run_qwen3vl_sam2_pilot import EXECUTOR_DEFINITIONS, TASK_DEFINITIONS, load_qwen_model, load_yaml, qwen_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ground v3 target/reject semantic parts with Qwen3-VL.")
    parser.add_argument("--dataset-root", default=".", help="Dataset root directory.")
    parser.add_argument("--config", default="configs/qwen3vl_sam2_pilot.yaml", help="YAML config relative to dataset root.")
    parser.add_argument("--pilot-csv", default="processed/metadata/vlm_pilot_samples_v0_1.csv")
    parser.add_argument("--renders-root", default="processed/vlm_semantic_part/renders")
    parser.add_argument("--semantic-plan-root", default="processed/vlm_candidate_v3/semantic_plans")
    parser.add_argument("--output-root", default="processed/vlm_candidate_v3/target_reject_grounding")
    parser.add_argument("--pilot-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--max-regions-per-role", type=int, default=4)
    parser.add_argument(
        "--max-target-box-area-fraction",
        type=float,
        default=0.35,
        help="Target boxes larger than this image fraction are discarded when positive points exist.",
    )
    parser.add_argument(
        "--strict-json",
        action="store_true",
        help="Abort on malformed Qwen JSON instead of recording the failed view and continuing.",
    )
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


def extract_json_block(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    start = stripped.find("{")
    if start < 0:
        raise json.JSONDecodeError("No JSON object start found", stripped, 0)
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(stripped)):
        ch = stripped[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : idx + 1]
    block = stripped[start:]
    if depth > 0:
        block += "}" * depth
    return block


def repair_common_json_issues(text: str) -> str:
    repaired = text.strip()
    repaired = repaired.replace("\u201c", '"').replace("\u201d", '"')
    repaired = repaired.replace("\u2018", "'").replace("\u2019", "'")
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = re.sub(r"\bTrue\b", "true", repaired)
    repaired = re.sub(r"\bFalse\b", "false", repaired)
    repaired = re.sub(r"\bNone\b", "null", repaired)
    return repaired


def safe_extract_json(text: str, strict: bool) -> dict[str, Any]:
    block = extract_json_block(text)
    attempts = [block, repair_common_json_issues(block)]
    errors: list[str] = []
    for candidate in attempts:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
    if strict:
        raise json.JSONDecodeError(errors[-1], attempts[-1], 0)
    return {
        "target_regions": [],
        "reject_regions": [],
        "coverage_warning": "qwen_json_parse_failed",
        "_parse_error": " | ".join(errors),
        "_raw_text": text,
        "_json_block": block[:4000],
    }


def selected_rows(root: Path, args: argparse.Namespace) -> list[dict[str, str]]:
    rows = read_csv(resolve_path(root, args.pilot_csv))
    if args.pilot_id:
        rows = [row for row in rows if row.get("pilot_id") == args.pilot_id]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No pilot rows selected.")
    return rows


def render_image_path(root: Path, entry: dict[str, Any], base_dir: Path) -> Path:
    for key in ("dense_render_path", "selector_path", "silhouette_path", "render_path"):
        value = entry.get(key)
        if value:
            path = resolve_portable_path(root, value, base_dir)
            if path.exists():
                return path
    raise FileNotFoundError(f"No usable image path in view entry: {entry}")


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
        point = [int(max(0, min(width - 1, round(x)))), int(max(0, min(height - 1, round(y))))]
        if point not in out:
            out.append(point)
    return out


def normalize_regions(
    value: Any,
    role: str,
    width: int,
    height: int,
    max_regions: int,
    max_target_box_area_fraction: float,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        box = normalize_box(item.get("box"), width, height)
        points = normalize_points(item.get("positive_points", item.get("points", [])), width, height)
        refine_action = ""
        box_before_refine = box
        if role == "target" and box is not None and points:
            box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
            area_fraction = box_area / max(1, width * height)
            if area_fraction > max(0.01, float(max_target_box_area_fraction)):
                box = None
                refine_action = "drop_oversized_target_box_keep_points"
        if box is None and not points:
            continue
        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        region = {
            "role": role,
            "part_name": str(item.get("part_name", item.get("label", role))).strip() or role,
            "box": box,
            "positive_points": points,
            "confidence": max(0.0, min(1.0, confidence)),
            "reason": str(item.get("reason", "")),
        }
        if refine_action:
            region["box_before_refine"] = box_before_refine
            region["refine_action"] = refine_action
        out.append(region)
        if len(out) >= max(1, int(max_regions)):
            break
    return out


def build_prompt(row: dict[str, str], plan: dict[str, Any], view: str, width: int, height: int) -> str:
    executor = row.get("executor", "")
    task = row.get("task", "")
    return f"""
You are grounding semantic target/reject parts on a sparse point-cloud render.

Image coordinates: width={width}, height={height}. Output coordinates in this image.
Use tight boxes around visible parts. Do not box the whole object unless the whole object is truly the target.

Object category: {row.get('object_category', '')}
Task: {task}
Task definition: {TASK_DEFINITIONS.get(task, task)}
End-effector: {executor}
End-effector definition: {EXECUTOR_DEFINITIONS.get(executor, executor)}
View: {view}

Semantic plan:
target_positive_parts: {plan.get('target_positive_parts', [])}
target_grounding_queries: {plan.get('target_grounding_queries', [])}
reject_negative_parts: {plan.get('reject_negative_parts', [])}
reject_grounding_queries: {plan.get('reject_grounding_queries', [])}

Grounding rules:
1. target_regions are only task-related positive functional parts.
2. reject_regions are hard veto areas that must be excluded from target candidates.
3. For scissors + hook, target handle/finger holes and handle-ring inner rims; reject blades, cutting edges, tips, and long blade boundaries.
4. For hook, do not mark ordinary long edges unless the hook can enter/catch/interlock.
5. If a target part is visible but sparse, use points plus a tight box around the sparse points.
6. If a reject part is visible, mark it even when no target is visible.

Return strict JSON only:
{{
  "view": "{view}",
  "target_regions": [
    {{
      "part_name": "handle loop",
      "box": [x1, y1, x2, y2],
      "positive_points": [[x, y]],
      "confidence": 0.0,
      "reason": "why this is a target"
    }}
  ],
  "reject_regions": [
    {{
      "part_name": "blade",
      "box": [x1, y1, x2, y2],
      "positive_points": [[x, y]],
      "confidence": 0.0,
      "reason": "why this must be rejected"
    }}
  ],
  "coverage_warning": "short note if the target is not visible or ambiguous",
  "confidence": 0.0
}}
"""


def run_qwen_json(model: Any, processor: Any, image_path: Path, prompt: str, cfg: dict[str, Any], strict_json: bool) -> dict[str, Any]:
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
    raw = safe_extract_json(text, strict_json)
    raw.setdefault("_raw_text", text)
    return raw


def normalize_grounding(
    raw: dict[str, Any],
    view: str,
    width: int,
    height: int,
    max_regions: int,
    max_target_box_area_fraction: float,
) -> dict[str, Any]:
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "view": view,
        "target_regions": normalize_regions(
            raw.get("target_regions", []),
            "target",
            width,
            height,
            max_regions,
            max_target_box_area_fraction,
        ),
        "reject_regions": normalize_regions(
            raw.get("reject_regions", []),
            "reject",
            width,
            height,
            max_regions,
            max_target_box_area_fraction,
        ),
        "coverage_warning": str(raw.get("coverage_warning", raw.get("warning", ""))),
        "confidence": max(0.0, min(1.0, confidence)),
        "raw_qwen3vl_response": raw,
    }


def run_for_row(root: Path, args: argparse.Namespace, cfg: dict[str, Any], row: dict[str, str], model: Any | None, processor: Any | None) -> dict[str, Any]:
    pilot_id = row["pilot_id"]
    sample_id = row["sample_id"]
    plan_path = resolve_path(root, args.semantic_plan_root) / pilot_id / "combined_semantic_plan.json"
    plan = read_json(plan_path)
    render_manifest_path = resolve_path(root, args.renders_root) / sample_id / "view_manifest.json"
    render_manifest = read_json(render_manifest_path)
    output_dir = resolve_path(root, args.output_root) / pilot_id
    output_dir.mkdir(parents=True, exist_ok=True)
    views: list[dict[str, Any]] = []

    for entry in render_manifest.get("views", []):
        view = str(entry["view"])
        image_path = render_image_path(root, entry, render_manifest_path.parent)
        with Image.open(image_path) as img:
            width, height = img.size
        if args.validate_only:
            views.append({"view": view, "validated": True})
            continue
        if args.dry_run:
            grounding = {
                "view": view,
                "target_regions": [],
                "reject_regions": [],
                "coverage_warning": "dry_run",
                "confidence": 0.0,
                "raw_qwen3vl_response": {},
            }
        else:
            raw = run_qwen_json(model, processor, image_path, build_prompt(row, plan, view, width, height), cfg, args.strict_json)
            grounding = normalize_grounding(
                raw,
                view,
                width,
                height,
                args.max_regions_per_role,
                args.max_target_box_area_fraction,
            )
        out_path = output_dir / f"{view}_target_reject_grounding.json"
        write_json(out_path, grounding, args.overwrite)
        views.append(
            {
                "view": view,
                "grounding_path": str(out_path.relative_to(root).as_posix()),
                "target_regions": len(grounding.get("target_regions", [])),
                "reject_regions": len(grounding.get("reject_regions", [])),
                "coverage_warning": grounding.get("coverage_warning", ""),
                "parse_error": grounding.get("raw_qwen3vl_response", {}).get("_parse_error", ""),
            }
        )

    combined = {
        "version": "v3",
        "pipeline": "target_reject_2d_grounding",
        "pilot_id": pilot_id,
        "sample_id": sample_id,
        "object_category": row.get("object_category", ""),
        "task": row.get("task", ""),
        "executor": row.get("executor", ""),
        "semantic_plan": str(plan_path.relative_to(root).as_posix()),
        "render_manifest": str(render_manifest_path.relative_to(root).as_posix()),
        "views": views,
        "notes": "2D target/reject seeds only; project to 3D before candidate growth.",
    }
    write_json(output_dir / "combined_target_reject_grounding.json", combined, args.overwrite)
    return combined


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    cfg = load_yaml(resolve_path(root, args.config))
    rows = selected_rows(root, args)
    model = None
    processor = None
    if not args.validate_only and not args.dry_run:
        model, processor = load_qwen_model(cfg, root)
    outputs = [run_for_row(root, args, cfg, row, model, processor) for row in rows]
    print(json.dumps({"rows": len(outputs), "validate_only": args.validate_only, "dry_run": args.dry_run, "outputs": outputs}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
