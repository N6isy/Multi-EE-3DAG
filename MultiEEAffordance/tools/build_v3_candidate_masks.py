#!/usr/bin/env python3
"""Build reviewable four-channel masks from v3 reject-aware candidates."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from path_utils import relative_to_dataset, resolve_portable_path
from utils.task_taxonomy import ALL_TASKS, EXECUTOR_ORDER, LEGACY_DEFAULT_ACTIVE_TASKS


KNOWN_TASKS = set(ALL_TASKS)
DEFAULT_ACTIVE_TASKS = set(LEGACY_DEFAULT_ACTIVE_TASKS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v3 candidate four-channel masks for human review.")
    parser.add_argument("--dataset-root", default=".", help="Dataset root directory.")
    parser.add_argument("--pilot-csv", default="processed/metadata/vlm_pilot_samples_v0_1.csv")
    parser.add_argument("--samples", default="processed/metadata/samples_checked_v0_1.jsonl")
    parser.add_argument("--candidate-root", default="processed/vlm_candidate_v3/3d_candidates")
    parser.add_argument(
        "--metadata-root",
        default="",
        help=(
            "Optional root used to resolve and write portable generated paths. "
            "Use the mirrored dataset root under external storage for review packages."
        ),
    )
    parser.add_argument("--empty-pilot-csv", default="", help="Optional CSV of empty-review rows to append.")
    parser.add_argument("--output-mask-root", default="processed/vlm_candidate_v3/fused_masks")
    parser.add_argument("--output-samples", default="processed/metadata/v3_candidate_samples_v0_1.jsonl")
    parser.add_argument("--output-split-dir", default="splits_v3_candidates")
    parser.add_argument("--summary-json", default="processed/metadata/v3_candidate_summary_v0_1.json")
    parser.add_argument("--pilot-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--include-tasks",
        default=",".join(sorted(DEFAULT_ACTIVE_TASKS)),
        help=(
            "Comma-separated tasks to keep, or 'all'. The v3 candidate-mask default "
            "keeps legacy proposal tasks pick_up,open_pull,press_push and excludes lift_carry."
        ),
    )
    parser.add_argument(
        "--exclude-tasks",
        default="",
        help="Comma-separated tasks to drop after include filtering.",
    )
    parser.add_argument("--selected-candidates", default="", help="Comma-separated manual candidate ids.")
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_task_filter(value: str, allow_all: bool) -> set[str] | None:
    raw = str(value or "").strip()
    if not raw:
        return set()
    if raw.lower() == "all":
        if allow_all:
            return None
        raise ValueError("'all' is only valid for --include-tasks")
    tasks = {item.strip() for item in raw.split(",") if item.strip()}
    unknown = sorted(tasks.difference(KNOWN_TASKS))
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}. Known tasks: {sorted(KNOWN_TASKS)}")
    return tasks


def resolve_path(root: Path, value: str | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def metadata_root(root: Path, args: argparse.Namespace) -> Path:
    return resolve_path(root, args.metadata_root) if args.metadata_root else root


def metadata_path(root: Path, args: argparse.Namespace, path: Path) -> str:
    return relative_to_dataset(metadata_root(root, args), path)


def resolve_generated_path(root: Path, args: argparse.Namespace, value: str | Path, base_dir: Path | None = None) -> Path:
    raw = str(value).strip()
    if not raw:
        return Path("")
    path = Path(raw)
    if path.is_absolute() and path.exists():
        return path
    if base_dir is not None and not path.is_absolute():
        candidate = base_dir / raw
        if candidate.exists():
            return candidate
    if args.metadata_root and not path.is_absolute():
        candidate = metadata_root(root, args) / raw
        if candidate.exists():
            return candidate
    return resolve_portable_path(root, raw, base_dir)


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Samples JSONL not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def write_json(path: Path, data: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def selected_rows(root: Path, args: argparse.Namespace) -> list[dict[str, str]]:
    rows = read_csv(resolve_path(root, args.pilot_csv))
    include_tasks = parse_task_filter(args.include_tasks, allow_all=True)
    exclude_tasks = parse_task_filter(args.exclude_tasks, allow_all=False) or set()
    if include_tasks is not None:
        rows = [row for row in rows if row.get("task") in include_tasks]
    if exclude_tasks:
        rows = [row for row in rows if row.get("task") not in exclude_tasks]
    if args.pilot_id:
        rows = [row for row in rows if row.get("pilot_id") == args.pilot_id]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No pilot rows selected.")
    return rows


def parse_id_list(value: str) -> list[str]:
    out: list[str] = []
    for item in str(value or "").split(","):
        cid = item.strip().upper()
        if cid and cid not in out:
            out.append(cid)
    return out


def safe_dict(value: Any) -> dict[str, Any]:
    """Return a defensive shallow dict for metadata fields.

    Large-scale converted queues may carry fields such as ``negative_reason`` as
    a plain string in the CSV row.  Calling ``dict("reason")`` raises
    ``dictionary update sequence element #0 has length 1; 2 is required`` and
    used to stop the whole build.  Non-object metadata is ignored here and kept
    in per-row notes/summaries instead of being treated as a mapping.
    """
    if isinstance(value, dict):
        return dict(value)
    if value in (None, ""):
        return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    try:
        candidate = dict(value)
    except (TypeError, ValueError):
        return {}
    return candidate


def merge_sample(row: dict[str, str], sample_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sample = dict(sample_by_id.get(row.get("sample_id", ""), {}))
    for key, value in row.items():
        if value not in (None, ""):
            sample[key] = value
    return sample


def build_for_row(root: Path, args: argparse.Namespace, row: dict[str, str], sample_by_id: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    pilot_id = row["pilot_id"]
    sample_id = row["sample_id"]
    executor = row.get("executor", "")
    if executor not in EXECUTOR_ORDER:
        raise ValueError(f"Unknown executor '{executor}' for {pilot_id}")
    sample = merge_sample(row, sample_by_id)

    manifest_path = resolve_path(root, args.candidate_root) / pilot_id / "candidate_manifest.json"
    manifest = read_json(manifest_path)
    npz_path = resolve_generated_path(root, args, manifest["candidate_npz"], manifest_path.parent)
    data = np.load(npz_path, allow_pickle=True)
    candidate_ids = [str(item).upper() for item in data["candidate_ids"].tolist()]
    candidate_masks = data["candidate_masks"].astype(np.uint8)

    selected_ids = parse_id_list(args.selected_candidates) or [str(item).upper() for item in manifest.get("default_selected_candidates", [])]
    selected_indices = [candidate_ids.index(cid) for cid in selected_ids if cid in candidate_ids]
    n = int(candidate_masks.shape[1]) if candidate_masks.ndim == 2 else 0
    target_mask = np.zeros((n,), dtype=np.uint8)
    if selected_indices:
        target_mask = (candidate_masks[selected_indices].sum(axis=0) > 0).astype(np.uint8)
    if int(target_mask.sum()) == 0 and not args.allow_empty:
        raise ValueError(f"v3 target candidate mask is empty for {pilot_id}. Use --allow-empty if expected.")

    base_mask_value = row.get("checked_mask_path") or sample.get("multi_channel_mask_path") or row.get("source_mask_path")
    base_mask_path = resolve_portable_path(root, base_mask_value)
    if not base_mask_path.exists():
        raise FileNotFoundError(f"Base mask not found: {base_mask_path}")
    base_mask = np.load(base_mask_path)
    if base_mask.ndim != 2 or base_mask.shape[1] != len(EXECUTOR_ORDER):
        raise ValueError(f"Expected base mask shape [N,4], got {base_mask.shape}: {base_mask_path}")
    if base_mask.shape[0] != target_mask.shape[0]:
        raise ValueError(f"Candidate length {target_mask.shape[0]} does not match base mask {base_mask.shape[0]}")

    out_mask = base_mask.astype(np.uint8).copy()
    channel = EXECUTOR_ORDER.index(executor)
    out_mask[:, channel] = target_mask
    out_mask_path = resolve_path(root, args.output_mask_root) / f"{sample_id}_{pilot_id}_v3_candidate.npy"
    if out_mask_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output mask exists. Use --overwrite: {out_mask_path}")
    out_mask_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_mask_path, out_mask)

    feasibility = safe_dict(sample.get("feasibility"))
    label_source = safe_dict(sample.get("label_source"))
    negative_reason = safe_dict(sample.get("negative_reason"))
    feasibility[executor] = bool(int(target_mask.sum()) > 0)
    label_source[executor] = "mixed"
    negative_reason[executor] = None if feasibility[executor] else f"v3_no_selected_{executor}_candidate"

    v3_update = {
        "pilot_id": pilot_id,
        "executor": executor,
        "source": "v3_semantic_target_reject_candidate",
        "provenance": [
            "qwen3vl_semantic_target_reject_plan",
            "qwen3vl_target_reject_grounding",
            "point_index_projection",
            "reject_aware_3d_candidate_growth",
        ],
        "candidate_manifest": metadata_path(root, args, manifest_path),
        "selected_candidates": selected_ids,
        "default_selected_candidates": manifest.get("default_selected_candidates", []),
        "positive_points": int(target_mask.sum()),
        "target_seed_points": int(manifest.get("target_seed_points", 0)),
        "reject_veto_points": int(manifest.get("reject_veto_points", 0)),
        "requires_human_review": True,
    }
    sample["multi_channel_mask_path"] = metadata_path(root, args, out_mask_path)
    sample["executor_order"] = EXECUTOR_ORDER
    sample["feasibility"] = feasibility
    sample["label_source"] = label_source
    sample["negative_reason"] = negative_reason
    sample["quality_flag"] = "weak"
    sample["split"] = "val"
    sample["v3_candidate_update"] = v3_update
    # Keep compatibility with serve_v2_annotation_app.py while the v3 review UI
    # is still sharing the same point-level editor.
    sample["v2_candidate_update"] = {
        "pilot_id": pilot_id,
        "executor": executor,
        "source": "v3_semantic_target_reject_candidate",
        "candidate_manifest": metadata_path(root, args, manifest_path),
        "rule_filter_path": "",
        "selected_candidates": selected_ids,
        "positive_points": int(target_mask.sum()),
        "requires_human_review": True,
    }
    sample["notes"] = (
        str(sample.get("notes", ""))
        + f" | v3_candidate: {executor} channel generated from semantic target seeds with reject-veto growth; requires human review."
    )
    summary = {
        "pilot_id": pilot_id,
        "sample_id": sample_id,
        "executor": executor,
        "selected_candidates": selected_ids,
        "positive_points": int(target_mask.sum()),
        "target_seed_points": int(manifest.get("target_seed_points", 0)),
        "reject_veto_points": int(manifest.get("reject_veto_points", 0)),
        "output_mask_path": metadata_path(root, args, out_mask_path),
    }
    return sample, summary


def build_empty_for_row(root: Path, args: argparse.Namespace, row: dict[str, str], sample_by_id: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    pilot_id = row["pilot_id"]
    sample_id = row["sample_id"]
    executor = row.get("executor", "")
    if executor not in EXECUTOR_ORDER:
        raise ValueError(f"Unknown executor '{executor}' for empty review row {pilot_id}")
    sample = merge_sample(row, sample_by_id)
    base_mask_value = row.get("checked_mask_path") or sample.get("multi_channel_mask_path") or row.get("source_mask_path")
    base_mask_path = resolve_portable_path(root, base_mask_value)
    if not base_mask_path.exists():
        raise FileNotFoundError(f"Base mask not found for empty review row {pilot_id}: {base_mask_path}")
    base_mask = np.load(base_mask_path)
    if base_mask.ndim != 2 or base_mask.shape[1] != len(EXECUTOR_ORDER):
        raise ValueError(f"Expected base mask shape [N,4], got {base_mask.shape}: {base_mask_path}")
    out_mask = base_mask.astype(np.uint8).copy()
    channel = EXECUTOR_ORDER.index(executor)
    out_mask[:, channel] = 0
    out_mask_path = resolve_path(root, args.output_mask_root) / f"{sample_id}_{pilot_id}_v3_empty_review.npy"
    if out_mask_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output mask exists. Use --overwrite: {out_mask_path}")
    out_mask_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_mask_path, out_mask)

    feasibility = safe_dict(sample.get("feasibility"))
    label_source = safe_dict(sample.get("label_source"))
    negative_reason = safe_dict(sample.get("negative_reason"))
    feasibility[executor] = False
    label_source[executor] = "unavailable"
    negative_reason[executor] = row.get("negative_reason") or negative_reason.get(executor) or f"no_{executor}_feasible_region"

    update = {
        "pilot_id": pilot_id,
        "executor": executor,
        "source": "v3_empty_review_required",
        "candidate_manifest": "",
        "rule_filter_path": "",
        "selected_candidates": [],
        "positive_points": 0,
        "requires_human_review": True,
        "review_mode": "confirm_empty",
        "negative_reason": negative_reason[executor],
    }
    sample["multi_channel_mask_path"] = metadata_path(root, args, out_mask_path)
    sample["executor_order"] = EXECUTOR_ORDER
    sample["feasibility"] = feasibility
    sample["label_source"] = label_source
    sample["negative_reason"] = negative_reason
    sample["quality_flag"] = "weak"
    sample["split"] = "val"
    sample["v3_candidate_update"] = dict(update)
    sample["v2_candidate_update"] = dict(update)
    sample["notes"] = (
        str(sample.get("notes", ""))
        + f" | v3_empty_review: {executor} channel is expected empty and requires human confirmation."
    )
    summary = {
        "pilot_id": pilot_id,
        "sample_id": sample_id,
        "executor": executor,
        "selected_candidates": [],
        "positive_points": 0,
        "output_mask_path": metadata_path(root, args, out_mask_path),
        "review_mode": "confirm_empty",
    }
    return sample, summary


def build_failed_candidate_row(
    root: Path,
    args: argparse.Namespace,
    row: dict[str, str],
    sample_by_id: dict[str, dict[str, Any]],
    exc: Exception,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pilot_id = row["pilot_id"]
    sample_id = row["sample_id"]
    executor = row.get("executor", "")
    if executor not in EXECUTOR_ORDER:
        raise ValueError(f"Unknown executor '{executor}' for failed candidate row {pilot_id}") from exc
    sample = merge_sample(row, sample_by_id)
    base_mask_value = row.get("checked_mask_path") or sample.get("multi_channel_mask_path") or row.get("source_mask_path")
    base_mask_path = resolve_portable_path(root, base_mask_value)
    if not base_mask_path.exists():
        raise FileNotFoundError(f"Base mask not found for failed candidate row {pilot_id}: {base_mask_path}") from exc
    base_mask = np.load(base_mask_path)
    if base_mask.ndim != 2 or base_mask.shape[1] != len(EXECUTOR_ORDER):
        raise ValueError(f"Expected base mask shape [N,4], got {base_mask.shape}: {base_mask_path}") from exc
    out_mask = base_mask.astype(np.uint8).copy()
    channel = EXECUTOR_ORDER.index(executor)
    out_mask[:, channel] = 0
    out_mask_path = resolve_path(root, args.output_mask_root) / f"{sample_id}_{pilot_id}_v3_candidate_build_error.npy"
    if out_mask_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output mask exists. Use --overwrite: {out_mask_path}") from exc
    out_mask_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_mask_path, out_mask)

    error_text = f"{type(exc).__name__}: {exc}"
    feasibility = safe_dict(sample.get("feasibility"))
    label_source = safe_dict(sample.get("label_source"))
    negative_reason = safe_dict(sample.get("negative_reason"))
    feasibility[executor] = False
    label_source[executor] = "unavailable"
    negative_reason[executor] = error_text
    update = {
        "pilot_id": pilot_id,
        "executor": executor,
        "source": "v3_candidate_build_failed_empty_placeholder",
        "candidate_manifest": "",
        "rule_filter_path": "",
        "selected_candidates": [],
        "positive_points": 0,
        "requires_human_review": True,
        "review_mode": "point_refine",
        "build_error": error_text,
    }
    sample["multi_channel_mask_path"] = metadata_path(root, args, out_mask_path)
    sample["executor_order"] = EXECUTOR_ORDER
    sample["feasibility"] = feasibility
    sample["label_source"] = label_source
    sample["negative_reason"] = negative_reason
    sample["quality_flag"] = "weak"
    sample["split"] = "val"
    sample["v3_candidate_update"] = update
    sample["v2_candidate_update"] = dict(update)
    sample["notes"] = (
        str(sample.get("notes", ""))
        + f" | v3_candidate_build_error: no automatic candidate was available for {executor}; human review should inspect manually."
    )
    summary = {
        "pilot_id": pilot_id,
        "sample_id": sample_id,
        "executor": executor,
        "selected_candidates": [],
        "positive_points": 0,
        "output_mask_path": metadata_path(root, args, out_mask_path),
        "review_mode": "point_refine",
        "build_error": error_text,
    }
    return sample, summary


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    try:
        rows = selected_rows(root, args)
    except ValueError:
        if args.empty_pilot_csv:
            rows = []
        else:
            raise
    empty_rows: list[dict[str, str]] = []
    if args.empty_pilot_csv:
        empty_path = resolve_path(root, args.empty_pilot_csv)
        if empty_path and empty_path.exists():
            empty_rows = read_csv(empty_path)
    checked_samples = read_jsonl(resolve_path(root, args.samples))
    sample_by_id = {str(row.get("sample_id")): row for row in checked_samples}
    output_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    build_errors: list[dict[str, Any]] = []
    try:
        from tqdm import tqdm

        row_iter = tqdm(rows, desc="build candidates", unit="row")
        empty_iter = tqdm(empty_rows, desc="build empty", unit="row") if empty_rows else []
    except Exception:
        row_iter = rows
        empty_iter = empty_rows
    for row in row_iter:
        try:
            sample, summary = build_for_row(root, args, row, sample_by_id)
        except Exception as exc:
            if not args.allow_empty:
                raise
            sample, summary = build_failed_candidate_row(root, args, row, sample_by_id, exc)
            build_errors.append(
                {
                    "pilot_id": row.get("pilot_id", ""),
                    "sample_id": row.get("sample_id", ""),
                    "executor": row.get("executor", ""),
                    "stage": "build_candidates",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        output_rows.append(sample)
        summaries.append(summary)
    for row in empty_iter:
        try:
            sample, summary = build_empty_for_row(root, args, row, sample_by_id)
        except Exception as exc:
            if not args.allow_empty:
                raise
            sample, summary = build_failed_candidate_row(root, args, row, sample_by_id, exc)
            summary["review_mode"] = "failed_needs_review"
            sample["v3_candidate_update"]["review_mode"] = "failed_needs_review"
            sample["v2_candidate_update"]["review_mode"] = "failed_needs_review"
            build_errors.append(
                {
                    "pilot_id": row.get("pilot_id", ""),
                    "sample_id": row.get("sample_id", ""),
                    "executor": row.get("executor", ""),
                    "stage": "build_empty",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        output_rows.append(sample)
        summaries.append(summary)

    output_samples = resolve_path(root, args.output_samples)
    write_jsonl(output_samples, output_rows, args.overwrite)
    split_dir = resolve_path(root, args.output_split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test", "contrast_test"]:
        split_path = split_dir / f"{split}.txt"
        if split_path.exists() and not args.overwrite:
            raise FileExistsError(f"Split file exists. Use --overwrite: {split_path}")
        ids = [row["sample_id"] for row in output_rows] if split == "val" else []
        split_path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
    summary = {
        "version": "v3",
        "rows": len(summaries),
        "output_samples": metadata_path(root, args, output_samples),
        "summaries": summaries,
        "build_errors": build_errors,
        "notes": "v3 candidate masks are review candidates, not checked ground truth.",
    }
    write_json(resolve_path(root, args.summary_json), summary, args.overwrite)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
