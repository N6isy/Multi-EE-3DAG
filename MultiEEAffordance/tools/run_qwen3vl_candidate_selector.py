#!/usr/bin/env python3
"""Use Qwen3-VL to select precomputed hook candidates from overlay images.

Unlike run_qwen3vl_sam2_pilot.py, this script does not ask Qwen3-VL to output
pixel boxes or points. The model only chooses labeled geometry candidates
such as A/B/C, which is much more stable for sparse point-cloud renders.
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
from run_qwen3vl_sam2_pilot import TASK_DEFINITIONS, extract_json, load_qwen_model, load_yaml, qwen_device


HOOK_DEFINITION = (
    "Hook end-effector. The usable region must allow insertion, mechanical "
    "interlocking, and pulling/lifting force. For lift_carry, prefer load-bearing "
    "handle loops, rings, holes, or back-side lips. Reject broad object surfaces, "
    "ordinary top edges, flat panels, and contact-only regions."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3-VL candidate selection for hook pilot rows.")
    parser.add_argument("--dataset-root", default=".", help="Dataset root directory.")
    parser.add_argument(
        "--config",
        default="configs/qwen3vl_sam2_pilot.yaml",
        help="YAML config relative to dataset root.",
    )
    parser.add_argument(
        "--pilot-csv",
        default="processed/metadata/vlm_pilot_samples_v0_1.csv",
        help="Pilot CSV relative to dataset root.",
    )
    parser.add_argument(
        "--overlay-root",
        default="processed/vlm_pilot/hook_candidate_overlays",
        help="Overlay root relative to dataset root.",
    )
    parser.add_argument(
        "--candidate-root",
        default="processed/vlm_pilot/hook_candidates",
        help="Candidate root relative to dataset root.",
    )
    parser.add_argument(
        "--output-root",
        default="processed/vlm_pilot/hook_candidate_selection",
        help="Selection output root relative to dataset root.",
    )
    parser.add_argument("--pilot-id", default=None, help="Run only one pilot row.")
    parser.add_argument("--limit", type=int, default=None, help="Limit selected hook pilot rows.")
    parser.add_argument("--executor", default="hook", help="Executor to process. Default: hook.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing selection outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Validate files and emit empty selections without loading Qwen.")
    parser.add_argument("--validate-only", action="store_true", help="Validate files only.")
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


def build_selector_prompt(
    row: dict[str, str],
    view: str,
    candidate_ids: list[str],
    candidate_names: list[str],
    candidate_descriptions: list[str],
) -> str:
    candidate_lines = "\n".join(
        f"- {candidate_ids[i]} / {candidate_names[i]}: {candidate_descriptions[i]}"
        for i in range(len(candidate_ids))
    )
    task = row.get("task", "")
    return f"""
You are selecting hook affordance candidates for a 3D object dataset.

You see a rendered point-cloud view with colored candidate overlays.
The colored candidate labels are:
{candidate_lines}

Object category: {row.get('object_category', '')}
Task: {task}
Task definition: {TASK_DEFINITIONS.get(task, task)}
End-effector: hook
End-effector definition: {HOOK_DEFINITION}
View name: {view}
Pilot issue: {row.get('issue_type', '')}
Pilot reason: {row.get('pilot_reason', '')}

Choose which labeled candidates are physically plausible hook regions.
Do not output pixel coordinates, boxes, or segmentation masks.

Decision rules:
1. Select a candidate only if it corresponds to a handle loop, hole boundary, ring, or back-side lip that a hook can enter or catch.
2. For lift_carry, the selected region must plausibly bear load while lifting/carrying.
3. Reject broad upper bands, flat bag/body surfaces, ordinary silhouette edges, and contact-only regions.
4. If the view is ambiguous but a candidate clearly highlights the visible handle/loop, select it with lower confidence.
5. If none of the candidates are hookable in this view, return an empty selected_candidates list.

Return strict JSON only:
{{
  "view": "{view}",
  "visible_hookable_structure": "short phrase or none",
  "selected_candidates": ["A"],
  "rejected_candidates": ["B", "C"],
  "confidence": 0.0,
  "reason": "short reason"
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


def normalize_candidate_list(value: Any, valid_ids: set[str]) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        return []
    normalized: list[str] = []
    for item in items:
        candidate_id = str(item).strip().upper()
        if candidate_id in valid_ids and candidate_id not in normalized:
            normalized.append(candidate_id)
    return normalized


def normalize_selection(raw: dict[str, Any], valid_ids: set[str]) -> dict[str, Any]:
    selected = normalize_candidate_list(
        raw.get("selected_candidates", raw.get("selected_candidate", raw.get("candidates", []))),
        valid_ids,
    )
    rejected = normalize_candidate_list(raw.get("rejected_candidates", []), valid_ids)
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "view": str(raw.get("view", "")),
        "visible_hookable_structure": str(raw.get("visible_hookable_structure", "")),
        "selected_candidates": selected,
        "rejected_candidates": rejected,
        "confidence": max(0.0, min(1.0, confidence)),
        "reason": str(raw.get("reason", raw.get("notes", ""))),
        "raw_qwen3vl_response": raw,
    }


def select_rows(args: argparse.Namespace, root: Path) -> list[dict[str, str]]:
    rows = read_csv(resolve_path(root, args.pilot_csv))
    rows = [row for row in rows if row.get("executor") == args.executor]
    if args.pilot_id:
        rows = [row for row in rows if row.get("pilot_id") == args.pilot_id]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No hook pilot rows selected.")
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
    candidate_manifest_path = resolve_path(root, args.candidate_root) / pilot_id / "candidate_manifest.json"
    candidate_manifest = read_json(candidate_manifest_path)
    candidate_ids = [str(item) for item in candidate_manifest["candidate_ids"]]
    candidate_names = [str(item) for item in candidate_manifest["candidate_names"]]
    candidate_descriptions = [str(item) for item in candidate_manifest["candidate_descriptions"]]
    valid_ids = set(candidate_ids)

    output_dir = resolve_path(root, args.output_root) / pilot_id
    output_dir.mkdir(parents=True, exist_ok=True)
    view_results: list[dict[str, Any]] = []
    vote_counter: Counter[str] = Counter()
    confidence_sum: Counter[str] = Counter()

    for view_entry in overlay_manifest.get("views", []):
        view = view_entry["view"]
        overlay_path = resolve_portable_path(root, view_entry["overlay_path"], overlay_manifest_path.parent)
        if not overlay_path.exists():
            raise FileNotFoundError(f"Overlay image not found: {overlay_path}")

        if args.validate_only:
            continue
        if args.dry_run:
            selection = {
                "view": view,
                "visible_hookable_structure": "dry_run",
                "selected_candidates": [],
                "rejected_candidates": candidate_ids,
                "confidence": 0.0,
                "reason": "dry_run",
                "raw_qwen3vl_response": {},
            }
        else:
            prompt = build_selector_prompt(row, view, candidate_ids, candidate_names, candidate_descriptions)
            raw = run_qwen_json(model, processor, overlay_path, prompt, cfg)
            selection = normalize_selection(raw, valid_ids)
            selection["view"] = view

        for candidate_id in selection["selected_candidates"]:
            vote_counter[candidate_id] += 1
            confidence_sum[candidate_id] += float(selection["confidence"])
        selection_path = output_dir / f"{view}_selection.json"
        write_json(selection_path, selection, args.overwrite)
        view_results.append(
            {
                "view": view,
                "selection_path": relative_to_dataset(root, selection_path),
                "selected_candidates": selection["selected_candidates"],
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

    ranked = sorted(
        (
            {
                "candidate_id": candidate_id,
                "votes": int(votes),
                "mean_confidence": float(confidence_sum[candidate_id] / max(1, votes)),
            }
            for candidate_id, votes in vote_counter.items()
        ),
        key=lambda item: (item["votes"], item["mean_confidence"]),
        reverse=True,
    )
    combined = {
        "pilot_id": pilot_id,
        "sample_id": row["sample_id"],
        "object_category": row.get("object_category", ""),
        "task": row.get("task", ""),
        "executor": row.get("executor", args.executor),
        "candidate_manifest": relative_to_dataset(root, candidate_manifest_path),
        "overlay_manifest": relative_to_dataset(root, overlay_manifest_path),
        "candidate_ids": candidate_ids,
        "view_results": view_results,
        "ranked_candidates": ranked,
        "selected_candidates": [item["candidate_id"] for item in ranked if item["votes"] > 0],
        "notes": "Qwen3-VL candidate selection only; selected masks still require rule check and human review.",
    }
    write_json(output_dir / "combined_selection.json", combined, args.overwrite)
    return combined


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    cfg_path = resolve_path(root, args.config)
    cfg = load_yaml(cfg_path)
    rows = select_rows(args, root)

    model = None
    processor = None
    if not args.validate_only and not args.dry_run:
        model, processor = load_qwen_model(cfg, root)

    outputs = [run_for_row(root, args, cfg, row, model, processor) for row in rows]
    print(
        json.dumps(
            {
                "rows": len(outputs),
                "validate_only": bool(args.validate_only),
                "dry_run": bool(args.dry_run),
                "outputs": outputs,
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
