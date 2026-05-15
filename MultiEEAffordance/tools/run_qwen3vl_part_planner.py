#!/usr/bin/env python3
"""Use Qwen3-VL as a semantic part planner for affordance annotation.

The planner does not output boxes, points, or masks. It outputs target part
names and grounding queries that can later be passed to GroundingDINO /
Florence-2 + SAM2.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from path_utils import relative_to_dataset, resolve_portable_path
from run_qwen3vl_sam2_pilot import EXECUTOR_DEFINITIONS, TASK_DEFINITIONS, extract_json, load_qwen_model, load_yaml, qwen_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3-VL semantic part planner.")
    parser.add_argument("--dataset-root", default=".", help="Dataset root directory.")
    parser.add_argument("--config", default="configs/qwen3vl_sam2_pilot.yaml", help="YAML config relative to dataset root.")
    parser.add_argument(
        "--pilot-csv",
        default="processed/metadata/vlm_pilot_samples_v0_1.csv",
        help="Pilot CSV relative to dataset root.",
    )
    parser.add_argument(
        "--renders-root",
        default="processed/vlm_semantic_part/renders",
        help="VLM-friendly render root relative to dataset root.",
    )
    parser.add_argument(
        "--output-root",
        default="processed/vlm_semantic_part/part_plans",
        help="Output part-plan root relative to dataset root.",
    )
    parser.add_argument("--pilot-id", default=None, help="Run only one pilot row.")
    parser.add_argument("--limit", type=int, default=None, help="Limit selected pilot rows.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing part plans.")
    parser.add_argument("--dry-run", action="store_true", help="Write deterministic placeholder part plans without loading Qwen.")
    parser.add_argument("--validate-only", action="store_true", help="Validate inputs without loading Qwen.")
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


def selected_rows(root: Path, args: argparse.Namespace) -> list[dict[str, str]]:
    rows = read_csv(resolve_path(root, args.pilot_csv))
    if args.pilot_id:
        rows = [row for row in rows if row.get("pilot_id") == args.pilot_id]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No pilot rows selected.")
    return rows


def build_part_prompt(row: dict[str, str], view: str) -> str:
    executor = row["executor"]
    task = row["task"]
    return f"""
You are helping build a research dataset for object-level 3D affordance grounding.

You see a VLM-friendly rendering panel created from a sparse 3D point cloud.
The panel may include a dense point render, a silhouette, and a zoomed region.
Your task is NOT to draw a box or a mask.
Your task is to identify which object part should be grounded by another model.

Object category: {row.get('object_category', '')}
Task: {task}
Task definition: {TASK_DEFINITIONS.get(task, task)}
End-effector: {executor}
End-effector definition: {EXECUTOR_DEFINITIONS.get(executor, executor)}
View name: {view}
Pilot issue: {row.get('issue_type', '')}
Pilot reason: {row.get('pilot_reason', '')}

Return strict JSON only:
{{
  "view": "{view}",
  "visible_object_parts": ["part name"],
  "target_part_names": ["part to ground"],
  "grounding_queries": ["short open-vocabulary phrase"],
  "mechanism_reason": "why these parts can support the task and end-effector mechanism",
  "reject_parts": ["parts that should not be positive"],
  "uncertain_parts": ["parts that need human review"],
  "feasible": true,
  "confidence": 0.0,
  "notes": "short note"
}}

Rules:
1. Do not output pixel coordinates, boxes, points, or masks.
2. Identify task-related operative contact parts, not all touchable surfaces.
3. For hook, prefer handle loops, rings, holes, inner boundaries, or lips that a hook can enter or catch.
4. For suction, prefer flat, continuous, low-curvature surfaces; reject handles, holes, rods, and edges.
5. For gripper, prefer paired or potentially paired opposing contact surfaces.
6. For dexterous_hand, prefer task-related grasp, wrap, press, pinch, or fine manipulation regions.
7. If the target part is very sparse in a point-cloud render, still name it if the object semantics and geometry indicate it.
8. If no plausible target part is visible, use feasible=false and empty target_part_names / grounding_queries.
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


def normalize_string_list(value: Any, max_count: int = 8) -> list[str]:
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


def normalize_plan(raw: dict[str, Any], view: str) -> dict[str, Any]:
    target_parts = normalize_string_list(raw.get("target_part_names", raw.get("target_parts", [])))
    queries = normalize_string_list(raw.get("grounding_queries", raw.get("queries", target_parts)))
    feasible = bool(raw.get("feasible", bool(target_parts or queries))) and bool(target_parts or queries)
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "view": view,
        "visible_object_parts": normalize_string_list(raw.get("visible_object_parts", []), max_count=12),
        "target_part_names": target_parts if feasible else [],
        "grounding_queries": queries if feasible else [],
        "mechanism_reason": str(raw.get("mechanism_reason", "")),
        "reject_parts": normalize_string_list(raw.get("reject_parts", []), max_count=12),
        "uncertain_parts": normalize_string_list(raw.get("uncertain_parts", []), max_count=12),
        "feasible": feasible,
        "confidence": max(0.0, min(1.0, confidence)),
        "notes": str(raw.get("notes", "")),
        "raw_qwen3vl_response": raw,
    }


def dry_run_plan(row: dict[str, str], view: str) -> dict[str, Any]:
    executor = row.get("executor", "")
    category = row.get("object_category", "").lower()
    task = row.get("task", "")
    if executor == "hook" and "bag" in category:
        queries = ["bag handle", "top handle loop", "hookable handle loop"]
        parts = ["bag handle", "top handle loop"]
        reason = "Dry run: bag handle is the expected hookable load-bearing part for lift/carry."
    else:
        queries = []
        parts = []
        reason = "Dry run placeholder; no semantic model was loaded."
    return {
        "view": view,
        "visible_object_parts": [],
        "target_part_names": parts,
        "grounding_queries": queries,
        "mechanism_reason": reason,
        "reject_parts": ["ordinary surface", "flat body panel"],
        "uncertain_parts": [],
        "feasible": bool(queries),
        "confidence": 0.0,
        "notes": "dry_run",
        "raw_qwen3vl_response": {},
    }


def combine_plans(row: dict[str, str], view_plans: list[dict[str, Any]]) -> dict[str, Any]:
    query_counter: Counter[str] = Counter()
    part_counter: Counter[str] = Counter()
    confidence_sum: Counter[str] = Counter()
    for plan in view_plans:
        confidence = float(plan.get("confidence", 0.0))
        for query in plan.get("grounding_queries", []):
            key = str(query).strip()
            if key:
                query_counter[key] += 1
                confidence_sum[key] += confidence
        for part in plan.get("target_part_names", []):
            key = str(part).strip()
            if key:
                part_counter[key] += 1

    ranked_queries = [
        {
            "query": query,
            "votes": int(votes),
            "mean_confidence": float(confidence_sum[query] / max(1, votes)),
        }
        for query, votes in query_counter.most_common()
    ]
    return {
        "pilot_id": row["pilot_id"],
        "sample_id": row["sample_id"],
        "object_category": row.get("object_category", ""),
        "task": row.get("task", ""),
        "executor": row.get("executor", ""),
        "target_part_names": [part for part, _ in part_counter.most_common()],
        "grounding_queries": [item["query"] for item in ranked_queries],
        "ranked_queries": ranked_queries,
        "view_plans": [
            {
                "view": plan["view"],
                "target_part_names": plan["target_part_names"],
                "grounding_queries": plan["grounding_queries"],
                "feasible": plan["feasible"],
                "confidence": plan["confidence"],
                "notes": plan["notes"],
            }
            for plan in view_plans
        ],
        "notes": "Semantic part plan only. It must be grounded, projected, fused, and reviewed.",
    }


def run_for_row(root: Path, args: argparse.Namespace, cfg: dict[str, Any], row: dict[str, str], model: Any, processor: Any) -> dict[str, Any]:
    pilot_id = row["pilot_id"]
    sample_id = row["sample_id"]
    render_manifest_path = resolve_path(root, args.renders_root) / sample_id / "view_manifest.json"
    manifest = read_json(render_manifest_path)
    output_dir = resolve_path(root, args.output_root) / pilot_id
    output_dir.mkdir(parents=True, exist_ok=True)

    view_plans: list[dict[str, Any]] = []
    for entry in manifest.get("views", []):
        view = entry["view"]
        image_value = entry.get("selector_path") or entry.get("dense_render_path")
        image_path = resolve_portable_path(root, image_value, render_manifest_path.parent)
        if not image_path.exists():
            raise FileNotFoundError(f"Planner image not found: {image_path}")
        if args.validate_only:
            continue
        if args.dry_run:
            plan = dry_run_plan(row, view)
        else:
            raw = run_qwen_json(model, processor, image_path, build_part_prompt(row, view), cfg)
            plan = normalize_plan(raw, view)
        plan_path = output_dir / f"{view}_part_plan.json"
        write_json(plan_path, plan, args.overwrite)
        view_plans.append(plan)

    if args.validate_only:
        return {"pilot_id": pilot_id, "sample_id": sample_id, "validated_views": len(manifest.get("views", []))}

    combined = combine_plans(row, view_plans)
    combined["render_manifest"] = relative_to_dataset(root, render_manifest_path)
    write_json(output_dir / "combined_part_plan.json", combined, args.overwrite)
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
