#!/usr/bin/env python3
"""Use Qwen3-VL to select labeled v2 candidate regions.

The model sees candidate overlays and returns candidate ids, not pixel boxes.
This keeps VLM in its stronger role: semantic-functional judging.
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
from run_qwen3vl_sam2_pilot import (
    EXECUTOR_DEFINITIONS,
    TASK_DEFINITIONS,
    extract_json,
    load_qwen_model,
    load_yaml,
    qwen_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3-VL candidate selection for v2 candidate regions.")
    parser.add_argument("--dataset-root", default=".", help="Dataset root directory.")
    parser.add_argument("--config", default="configs/qwen3vl_sam2_pilot.yaml", help="Qwen config relative to root.")
    parser.add_argument(
        "--pilot-csv",
        default="processed/metadata/vlm_pilot_samples_v0_1.csv",
        help="Pilot CSV relative to dataset root.",
    )
    parser.add_argument(
        "--overlay-root",
        default="processed/vlm_candidate_v2/candidate_overlays",
        help="Overlay root relative to dataset root.",
    )
    parser.add_argument(
        "--selection-root",
        default="processed/vlm_candidate_v2/vlm_selection",
        help="Selection output root relative to dataset root.",
    )
    parser.add_argument(
        "--part-plan-root",
        default="processed/vlm_semantic_part/part_plans",
        help="Optional Qwen3-VL part-plan root relative to dataset root.",
    )
    parser.add_argument("--pilot-id", default=None, help="Run only one pilot row.")
    parser.add_argument("--limit", type=int, default=None, help="Limit selected rows.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and write empty selections without loading Qwen.")
    parser.add_argument("--validate-only", action="store_true", help="Validate inputs only.")
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


def build_prompt(row: dict[str, str], view: str, candidates: list[dict[str, Any]], part_plan: dict[str, Any] | None) -> str:
    executor = row.get("executor", "")
    task = row.get("task", "")
    candidate_lines = "\n".join(
        "- {id}: {name} | family={family} | points={points} | suggested={suggested} | {desc}".format(
            id=item["candidate_id"],
            name=item["candidate_name"],
            family=item["candidate_family"],
            points=item.get("point_count", ""),
            suggested=",".join(item.get("recommended_executors", [])),
            desc=item.get("description", ""),
        )
        for item in candidates
    )
    part_context = "No previous semantic part plan is available."
    if part_plan:
        target_parts = part_plan.get("target_part_names", [])
        queries = part_plan.get("grounding_queries", [])
        ranked = part_plan.get("ranked_queries", [])
        part_context = (
            f"Previous Qwen3-VL semantic part plan target parts: {target_parts}\n"
            f"Previous grounding queries: {queries}\n"
            f"Ranked semantic queries: {ranked}\n"
            "Use this as semantic context. The current colored candidates may have generic geometry names, "
            "but they can still correspond to these target parts spatially."
        )
    return f"""
You are helping build a multi-label 3D affordance dataset for heterogeneous end-effectors.

You will see a rendered 3D point-cloud image with colored candidate regions.
The left panel is usually the full object view; the right panel is a zoomed crop.
Your job is to select candidate IDs, not to draw boxes or masks.

Object category: {row.get('object_category', '')}
Task: {task}
Task definition: {TASK_DEFINITIONS.get(task, task)}
End-effector: {executor}
End-effector definition: {EXECUTOR_DEFINITIONS.get(executor, executor)}
View: {view}
Pilot issue type: {row.get('issue_type', '')}
Pilot reason: {row.get('pilot_reason', '')}

Semantic part-planner context:
{part_context}

Candidate list:
{candidate_lines}

Decision principles:
1. Select only candidates that are task-related functional contact regions, not all touchable surfaces.
2. A selected region must satisfy the mechanism of the specified end-effector.
3. Candidate families are proposals, not labels. You may reject a geometrically plausible candidate if it is not semantically functional.
4. Do not reject a candidate only because its name is generic, such as "thin_structure" or "existing_gripper_weak_mask".
5. If a candidate spatially overlaps the semantic target part from the part plan, select it or mark it uncertain even if its original weak-label source came from another executor.
6. If the view is ambiguous, put the candidate into uncertain_candidates instead of selected_candidates.
7. Reject broad fallback/body regions unless they are clearly the functional operation area for this task.
8. Do not output pixel coordinates, boxes, or segmentation masks.

Return strict JSON only:
{{
  "view": "{view}",
  "visible_functional_parts": ["short part names"],
  "selected_candidates": ["A"],
  "uncertain_candidates": [],
  "rejected_candidates": ["B"],
  "confidence": 0.0,
  "reason": "short mechanism-based explanation"
}}
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


def normalize_id_list(value: Any, valid_ids: set[str]) -> list[str]:
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
        candidate_id = str(item).strip().upper()
        if candidate_id in valid_ids and candidate_id not in out:
            out.append(candidate_id)
    return out


def normalize_selection(raw: dict[str, Any], valid_ids: set[str], view: str) -> dict[str, Any]:
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "view": view,
        "visible_functional_parts": raw.get("visible_functional_parts", []),
        "selected_candidates": normalize_id_list(raw.get("selected_candidates", []), valid_ids),
        "uncertain_candidates": normalize_id_list(raw.get("uncertain_candidates", []), valid_ids),
        "rejected_candidates": normalize_id_list(raw.get("rejected_candidates", []), valid_ids),
        "confidence": max(0.0, min(1.0, confidence)),
        "reason": str(raw.get("reason", raw.get("notes", ""))),
        "raw_qwen3vl_response": raw,
    }


def load_part_plan(root: Path, args: argparse.Namespace, pilot_id: str) -> dict[str, Any] | None:
    path = resolve_path(root, args.part_plan_root) / pilot_id / "combined_part_plan.json"
    if not path.exists():
        return None
    return read_json(path)


def select_rows(root: Path, args: argparse.Namespace) -> list[dict[str, str]]:
    rows = read_csv(resolve_path(root, args.pilot_csv))
    if args.pilot_id:
        rows = [row for row in rows if row.get("pilot_id") == args.pilot_id]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No pilot rows selected.")
    return rows


def run_for_row(
    root: Path,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    row: dict[str, str],
    model: Any | None,
    processor: Any | None,
) -> dict[str, Any]:
    pilot_id = row["pilot_id"]
    overlay_manifest_path = resolve_path(root, args.overlay_root) / pilot_id / "overlay_manifest.json"
    overlay_manifest = read_json(overlay_manifest_path)
    part_plan = load_part_plan(root, args, pilot_id)
    candidates = overlay_manifest["candidates"]
    valid_ids = {str(item["candidate_id"]) for item in candidates}
    output_dir = resolve_path(root, args.selection_root) / pilot_id
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_counter: Counter[str] = Counter()
    uncertain_counter: Counter[str] = Counter()
    confidence_sum: Counter[str] = Counter()
    view_results: list[dict[str, Any]] = []

    for view_entry in overlay_manifest.get("views", []):
        view = str(view_entry["view"])
        selector_value = view_entry.get("selector_path") or view_entry.get("overlay_path")
        selector_path = resolve_portable_path(root, selector_value, overlay_manifest_path.parent)
        if not selector_path.exists():
            raise FileNotFoundError(f"Selector image not found: {selector_path}")
        if args.validate_only:
            continue
        if args.dry_run:
            selection = {
                "view": view,
                "visible_functional_parts": ["dry_run"],
                "selected_candidates": [],
                "uncertain_candidates": [],
                "rejected_candidates": sorted(valid_ids),
                "confidence": 0.0,
                "reason": "dry_run",
                "raw_qwen3vl_response": {},
            }
        else:
            prompt = build_prompt(row, view, candidates, part_plan)
            raw = run_qwen_json(model, processor, selector_path, prompt, cfg)
            selection = normalize_selection(raw, valid_ids, view)
        for cid in selection["selected_candidates"]:
            selected_counter[cid] += 1
            confidence_sum[cid] += float(selection["confidence"])
        for cid in selection["uncertain_candidates"]:
            uncertain_counter[cid] += 1
        selection_path = output_dir / f"{view}_selection.json"
        write_json(selection_path, selection, args.overwrite)
        view_results.append(
            {
                "view": view,
                "selection_path": relative_to_dataset(root, selection_path),
                "selected_candidates": selection["selected_candidates"],
                "uncertain_candidates": selection["uncertain_candidates"],
                "confidence": selection["confidence"],
                "reason": selection["reason"],
            }
        )

    if args.validate_only:
        return {
            "pilot_id": pilot_id,
            "sample_id": row["sample_id"],
            "validated_views": len(overlay_manifest.get("views", [])),
        }

    ranked = []
    for cid in sorted(valid_ids):
        votes = int(selected_counter[cid])
        uncertain_votes = int(uncertain_counter[cid])
        if votes or uncertain_votes:
            ranked.append(
                {
                    "candidate_id": cid,
                    "selected_votes": votes,
                    "uncertain_votes": uncertain_votes,
                    "mean_confidence": float(confidence_sum[cid] / max(1, votes)),
                }
            )
    ranked.sort(key=lambda item: (item["selected_votes"], item["mean_confidence"], item["uncertain_votes"]), reverse=True)
    combined = {
        "version": "v2",
        "pipeline": "vlm_guided_candidate_selection",
        "pilot_id": pilot_id,
        "sample_id": row["sample_id"],
        "object_category": row.get("object_category", ""),
        "task": row.get("task", ""),
        "executor": row.get("executor", ""),
        "overlay_manifest": relative_to_dataset(root, overlay_manifest_path),
        "candidate_ids": sorted(valid_ids),
        "view_results": view_results,
        "ranked_candidates": ranked,
        "selected_candidates": [item["candidate_id"] for item in ranked if item["selected_votes"] > 0],
        "uncertain_candidates": [item["candidate_id"] for item in ranked if item["selected_votes"] == 0 and item["uncertain_votes"] > 0],
        "notes": "VLM selection is a proposal. It must be checked by executor rules and human review.",
    }
    write_json(output_dir / "combined_selection.json", combined, args.overwrite)
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
    outputs = [run_for_row(root, args, cfg, row, model, processor) for row in rows]
    print(json.dumps({"rows": len(outputs), "validate_only": args.validate_only, "dry_run": args.dry_run, "outputs": outputs}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
