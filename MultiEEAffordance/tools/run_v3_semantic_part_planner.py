#!/usr/bin/env python3
"""Build v3 semantic affordance part plans with explicit target/reject roles.

v3 changes the pipeline objective:

  object + task + end-effector + multi-view renders
      -> semantic target parts
      -> explicit reject/veto parts
      -> target/reject grounding
      -> reject-aware 3D candidate growth

The planner still does not create masks. It produces a structured plan that
later stages use to avoid geometry-only mistakes such as treating a scissor
blade as a hookable part only because it is thin and high-curvature.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from path_utils import relative_to_dataset, resolve_portable_path
from run_qwen3vl_sam2_pilot import (
    EXECUTOR_DEFINITIONS,
    TASK_DEFINITIONS,
    extract_json,
    load_qwen_model,
    load_yaml,
    qwen_device,
)


EXECUTOR_RULES = {
    "gripper": (
        "parallel two-finger gripper: select opposing or potentially paired contact surfaces; "
        "reject surfaces that are merely visible but cannot form a stable clamp."
    ),
    "suction": (
        "single suction cup: select continuous low-curvature, sealable, normally accessible surfaces; "
        "reject holes, handles, rods, narrow rims, porous-looking gaps, and ordinary sharp edges."
    ),
    "hook": (
        "single hook: select holes, rings, handle-loop inner rims, lips, or back-side flanges that support "
        "insertion, mechanical interlocking, and pulling/lifting constraint; reject blades, cutting edges, "
        "tips, ordinary long edges, flat panels, and surfaces that cannot catch a hook."
    ),
    "dexterous_hand": (
        "multi-finger hand: select task-related grasp, wrap, pinch, press, pull, or fine-manipulation regions; "
        "reject generic touchable surfaces not needed for the specified task."
    ),
}


CATEGORY_PRIORS = {
    "scissors": {
        "hook": {
            "target": ["finger holes", "handle loops", "inner rim of handle rings"],
            "reject": ["blade", "cutting edge", "blade tip", "long blade boundary", "ordinary outer contour"],
        },
        "gripper": {
            "target": ["handle loops", "handle neck", "safe blunt handle region"],
            "reject": ["sharp blade", "cutting edge", "blade tip"],
        },
        "dexterous_hand": {
            "target": ["finger holes", "handle loops", "pivot-safe handle area"],
            "reject": ["cutting edge", "blade tip"],
        },
    },
    "knife": {
        "hook": {"target": ["handle hole", "lanyard hole"], "reject": ["blade", "cutting edge", "tip"]},
        "gripper": {"target": ["handle"], "reject": ["blade", "cutting edge", "tip"]},
        "dexterous_hand": {"target": ["handle"], "reject": ["blade", "cutting edge", "tip"]},
    },
    "bag": {
        "hook": {"target": ["handle loop", "bag handle", "handle inner rim"], "reject": ["bag body panel", "flat fabric surface"]},
        "gripper": {"target": ["handle", "side graspable region"], "reject": ["thin flexible surface without opposing support"]},
    },
    "mug": {
        "hook": {"target": ["handle opening", "handle inner rim"], "reject": ["cup body", "rim edge without hook retention"]},
        "suction": {"target": ["smooth side wall", "flat outer wall"], "reject": ["handle", "rim", "inside cavity"]},
        "gripper": {"target": ["handle", "opposing cup side surfaces"], "reject": ["inside cavity"]},
    },
    "door": {
        "open_pull": {"target": ["handle", "pull plate"], "reject": ["flat door panel edge", "hinge side"]},
        "suction": {"target": ["flat door panel"], "reject": ["door handle", "door gap", "edge seam"]},
    },
    "keyboard": {
        "dexterous_hand": {"target": ["keys", "button tops"], "reject": ["keyboard side wall", "base underside"]},
        "press_push": {"target": ["keys", "button tops"], "reject": ["keyboard frame", "side wall"]},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v3 semantic target/reject part planner.")
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
        default="processed/vlm_candidate_v3/semantic_plans",
        help="Output semantic plan root relative to dataset root.",
    )
    parser.add_argument("--pilot-id", default=None, help="Run only one pilot row.")
    parser.add_argument("--limit", type=int, default=None, help="Limit selected pilot rows.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing part plans.")
    parser.add_argument("--dry-run", action="store_true", help="Write deterministic placeholder plans without loading Qwen.")
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


def progress_rows(rows: list[dict[str, str]], desc: str):
    try:
        from tqdm import tqdm

        return tqdm(rows, desc=desc, unit="row")
    except Exception:
        return rows


def normalize_string_list(value: Any, max_count: int = 12) -> list[str]:
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


def category_prior(row: dict[str, str]) -> dict[str, list[str]]:
    category = row.get("object_category", "").lower()
    executor = row.get("executor", "").lower()
    task = row.get("task", "").lower()
    for key, priors in CATEGORY_PRIORS.items():
        if key in category:
            if executor in priors:
                return priors[executor]
            if task in priors:
                return priors[task]
    return {"target": [], "reject": []}


def build_prompt(row: dict[str, str], view: str) -> str:
    executor = row.get("executor", "")
    task = row.get("task", "")
    prior = category_prior(row)
    return f"""
You are designing annotation targets for a 3D affordance dataset.

You see a VLM-friendly render made from a sparse 3D point cloud. The render may be incomplete or sparse.
Your task is to produce a semantic annotation plan, not masks.

Input-data limitations:
- The render is not a real RGB photo. It is a visual proxy created from sparse 3D points.
- Thin structures, handles, loops, rings, holes, switches, buttons, and small parts may be broken, noisy, or only partially visible.
- Do not conclude that a target part is absent only because the sparse render does not show a clean continuous surface.
- If a part is semantically expected for the object/task/executor and is plausible from the visible geometry, include it with lower confidence or put it into uncertain_parts.
- This planning stage should maximize recall for downstream human review while still listing clear negative/reject parts.

Object category: {row.get('object_category', '')}
Task: {task}
Task definition: {TASK_DEFINITIONS.get(task, task)}
End-effector: {executor}
End-effector definition: {EXECUTOR_DEFINITIONS.get(executor, executor)}
Executor mechanism rule: {EXECUTOR_RULES.get(executor, executor)}
View: {view}
Pilot issue: {row.get('issue_type', '')}
Pilot reason: {row.get('pilot_reason', '')}
Category prior target parts: {prior.get('target', [])}
Category prior reject parts: {prior.get('reject', [])}

Important distinction:
- target_positive_parts are task-related functional contact/constraint parts for this end-effector.
- reject_negative_parts are parts that may look geometrically plausible but must not become positive.
- uncertain_parts are plausible but need human review.

For hook, use the three-step test:
1. Can the hook enter or go behind this structure?
2. Can it catch/interlock rather than slide off?
3. Can it pull/lift along the task direction?

Return strict JSON only:
{{
  "view": "{view}",
  "visible_object_parts": ["short part names"],
  "target_positive_parts": ["functional target part names"],
  "target_grounding_queries": ["short phrases for target grounding"],
  "reject_negative_parts": ["parts that must be vetoed"],
  "reject_grounding_queries": ["short phrases for reject grounding"],
  "uncertain_parts": ["parts requiring human review"],
  "mechanism_check": {{
    "can_enter_or_contact": true,
    "can_hold_or_constrain": true,
    "can_apply_task_force": true,
    "reason": "mechanism-level explanation"
  }},
  "feasible": true,
  "confidence": 0.0,
  "notes": "short note"
}}

Rules:
1. Do not output pixel coordinates, boxes, points, or masks.
2. Do not mark all visible surfaces as target parts.
3. Explicitly list reject parts that should be used as a veto layer.
4. For scissors + hook, target handle/finger holes and inner rings; reject blade, cutting edge, tip, and ordinary blade boundary.
5. If the target part is semantically expected but hard to see in a sparse point cloud, still name it and mark lower confidence.
6. Use uncertain_parts for plausible but weakly visible or partially resolved target parts instead of dropping them.
7. Set feasible=false only when the object/task/executor combination is genuinely implausible or no task-related target part is semantically expected.
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


def normalize_plan(raw: dict[str, Any], row: dict[str, str], view: str) -> dict[str, Any]:
    prior = category_prior(row)
    target_parts = normalize_string_list(raw.get("target_positive_parts", raw.get("target_part_names", [])))
    target_queries = normalize_string_list(raw.get("target_grounding_queries", raw.get("grounding_queries", target_parts)))
    reject_parts = normalize_string_list(raw.get("reject_negative_parts", raw.get("reject_parts", [])))
    reject_queries = normalize_string_list(raw.get("reject_grounding_queries", reject_parts))
    for text in prior.get("target", []):
        if text not in target_parts:
            target_parts.append(text)
        if text not in target_queries:
            target_queries.append(text)
    for text in prior.get("reject", []):
        if text not in reject_parts:
            reject_parts.append(text)
        if text not in reject_queries:
            reject_queries.append(text)
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    feasible = bool(raw.get("feasible", bool(target_parts))) and bool(target_parts)
    mechanism = raw.get("mechanism_check", {})
    if not isinstance(mechanism, dict):
        mechanism = {"reason": str(mechanism)}
    return {
        "view": view,
        "visible_object_parts": normalize_string_list(raw.get("visible_object_parts", []), max_count=16),
        "target_positive_parts": target_parts if feasible else [],
        "target_grounding_queries": target_queries if feasible else [],
        "reject_negative_parts": reject_parts,
        "reject_grounding_queries": reject_queries,
        "uncertain_parts": normalize_string_list(raw.get("uncertain_parts", []), max_count=16),
        "mechanism_check": {
            "can_enter_or_contact": bool(mechanism.get("can_enter_or_contact", feasible)),
            "can_hold_or_constrain": bool(mechanism.get("can_hold_or_constrain", feasible)),
            "can_apply_task_force": bool(mechanism.get("can_apply_task_force", feasible)),
            "reason": str(mechanism.get("reason", raw.get("mechanism_reason", ""))),
        },
        "feasible": feasible,
        "confidence": max(0.0, min(1.0, confidence)),
        "notes": str(raw.get("notes", "")),
        "raw_qwen3vl_response": raw,
    }


def dry_run_plan(row: dict[str, str], view: str) -> dict[str, Any]:
    raw = {
        "view": view,
        "visible_object_parts": [],
        "target_positive_parts": [],
        "target_grounding_queries": [],
        "reject_negative_parts": [],
        "reject_grounding_queries": [],
        "uncertain_parts": [],
        "mechanism_check": {"reason": "dry_run"},
        "feasible": False,
        "confidence": 0.0,
        "notes": "dry_run",
    }
    return normalize_plan(raw, row, view)


def combine_plans(root: Path, row: dict[str, str], view_plans: list[dict[str, Any]], render_manifest: Path) -> dict[str, Any]:
    target_counter: Counter[str] = Counter()
    query_counter: Counter[str] = Counter()
    reject_counter: Counter[str] = Counter()
    reject_query_counter: Counter[str] = Counter()
    confidence_by_query: defaultdict[str, float] = defaultdict(float)
    for plan in view_plans:
        confidence = float(plan.get("confidence", 0.0))
        for item in plan.get("target_positive_parts", []):
            target_counter[item] += 1
        for item in plan.get("target_grounding_queries", []):
            query_counter[item] += 1
            confidence_by_query[item] += confidence
        for item in plan.get("reject_negative_parts", []):
            reject_counter[item] += 1
        for item in plan.get("reject_grounding_queries", []):
            reject_query_counter[item] += 1
    ranked_queries = [
        {
            "query": query,
            "votes": int(votes),
            "mean_confidence": float(confidence_by_query[query] / max(1, votes)),
        }
        for query, votes in query_counter.most_common()
    ]
    return {
        "version": "v3",
        "pipeline": "semantic_target_reject_part_plan",
        "pilot_id": row["pilot_id"],
        "sample_id": row["sample_id"],
        "object_category": row.get("object_category", ""),
        "task": row.get("task", ""),
        "executor": row.get("executor", ""),
        "target_positive_parts": [part for part, _ in target_counter.most_common()],
        "target_grounding_queries": [item["query"] for item in ranked_queries],
        "ranked_target_queries": ranked_queries,
        "reject_negative_parts": [part for part, _ in reject_counter.most_common()],
        "reject_grounding_queries": [query for query, _ in reject_query_counter.most_common()],
        "view_plans": view_plans,
        "render_manifest": relative_to_dataset(root, render_manifest),
        "notes": (
            "v3 semantic plan only. It defines target and reject semantics; later stages ground, project, "
            "grow candidates, and require human review."
        ),
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
    sample_id = row["sample_id"]
    render_manifest_path = resolve_path(root, args.renders_root) / sample_id / "view_manifest.json"
    manifest = read_json(render_manifest_path)
    output_dir = resolve_path(root, args.output_root) / pilot_id
    output_dir.mkdir(parents=True, exist_ok=True)
    view_plans: list[dict[str, Any]] = []
    for entry in manifest.get("views", []):
        view = str(entry["view"])
        image_value = entry.get("selector_path") or entry.get("dense_render_path") or entry.get("render_path")
        image_path = resolve_portable_path(root, image_value, render_manifest_path.parent)
        if not image_path.exists():
            raise FileNotFoundError(f"Planner image not found: {image_path}")
        if args.validate_only:
            continue
        if args.dry_run:
            plan = dry_run_plan(row, view)
        else:
            raw = run_qwen_json(model, processor, image_path, build_prompt(row, view), cfg)
            plan = normalize_plan(raw, row, view)
        plan_path = output_dir / f"{view}_semantic_plan.json"
        write_json(plan_path, plan, args.overwrite)
        view_plans.append(plan)
    if args.validate_only:
        return {"pilot_id": pilot_id, "sample_id": sample_id, "validated_views": len(manifest.get("views", []))}
    combined = combine_plans(root, row, view_plans, render_manifest_path)
    write_json(output_dir / "combined_semantic_plan.json", combined, args.overwrite)
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
    estimated_calls = sum(8 for _ in rows)
    print(f"[v3-plan] rows={len(rows)} estimated_view_calls={estimated_calls}", flush=True)
    outputs = [run_for_row(root, args, cfg, row, model, processor) for row in progress_rows(rows, "v3 plan")]
    print(json.dumps({"rows": len(outputs), "validate_only": args.validate_only, "dry_run": args.dry_run, "outputs": outputs}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
