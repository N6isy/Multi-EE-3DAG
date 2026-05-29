#!/usr/bin/env python3
"""Run the v3 semantic target/reject affordance candidate pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.task_taxonomy import ALL_TASKS, LEGACY_DEFAULT_ACTIVE_TASKS


DEFAULT_STAGES = ["views", "plan", "part_propose", "render", "part_select", "part_filter", "build"]
GROUNDING_STAGES = ["views", "plan", "ground", "project", "grow", "render", "build"]
ALL_STAGES = [
    "views",
    "plan",
    "part_propose",
    "render",
    "part_select",
    "part_filter",
    "build",
    "ground",
    "project",
    "grow",
    "coverage",
    "visualize",
]
KNOWN_TASKS = list(ALL_TASKS)
DEFAULT_ACTIVE_TASKS = list(LEGACY_DEFAULT_ACTIVE_TASKS)
EMPTY_DECISIONS = {"empty", "empty_review_required", "confirm_empty", "skip_vlm_empty"}
DEFAULT_V3_OUTPUT_ROOT = "processed/vlm_candidate_v3"
DEFAULT_FILTERED_PILOT_CSV = "processed/vlm_candidate_v3/pipeline_runs/filtered_pilot_rows_latest.csv"
DEFAULT_EMPTY_REVIEW_CSV = "processed/vlm_candidate_v3/pipeline_runs/empty_review_rows_latest.csv"
DEFAULT_RUN_MANIFEST = "processed/vlm_candidate_v3/pipeline_runs/latest_run_manifest.json"
DEFAULT_REVIEW_OUTPUT_SAMPLES = "processed/metadata/v3_candidate_samples_v0_1.jsonl"
DEFAULT_REVIEW_SUMMARY_JSON = "processed/metadata/v3_candidate_summary_v0_1.json"
DEFAULT_REVIEW_SPLIT_DIR = "splits_v3_candidates"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v3 target/reject candidate pipeline.")
    parser.add_argument("--dataset-root", default="MultiEEAffordance", help="Dataset root directory.")
    parser.add_argument("--config", default="configs/qwen3vl_sam2_pilot.yaml", help="Qwen config relative to dataset root.")
    parser.add_argument("--pilot-csv", default="processed/metadata/vlm_pilot_samples_v0_1.csv")
    parser.add_argument(
        "--samples",
        default="",
        help="Samples JSONL used by render/build stages. If omitted, infer it from queue rows or queue summary when possible.",
    )
    parser.add_argument(
        "--include-tasks",
        default=",".join(DEFAULT_ACTIVE_TASKS),
        help=(
            "Comma-separated tasks to keep, or 'all'. The v3 candidate-generation default "
            "keeps legacy proposal tasks pick_up,open_pull,press_push and excludes lift_carry."
        ),
    )
    parser.add_argument(
        "--exclude-tasks",
        default="",
        help="Comma-separated tasks to drop after include filtering.",
    )
    parser.add_argument(
        "--filtered-pilot-csv",
        default=DEFAULT_FILTERED_PILOT_CSV,
        help="Filtered pilot CSV written relative to dataset root when task filtering is active.",
    )
    parser.add_argument(
        "--empty-review-csv",
        default=DEFAULT_EMPTY_REVIEW_CSV,
        help="Task-filtered empty-review rows written relative to dataset root.",
    )
    parser.add_argument(
        "--include-decisions",
        default="non_empty",
        help="Decision filter for VLM/candidate stages: non_empty, all, or comma-separated decision values.",
    )
    parser.add_argument("--pilot-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stages", default=",".join(DEFAULT_STAGES), help=f"Comma-separated stages: {','.join(ALL_STAGES)}")
    parser.add_argument(
        "--candidate-source",
        choices=["partseg", "grounding"],
        default="partseg",
        help="partseg uses 3D part proposals plus VLM candidate selection; grounding keeps the old VLM coordinate path.",
    )
    parser.add_argument(
        "--data-storage-root",
        default="",
        help=(
            "Optional external data storage root, e.g. /home/lzq/data. When set, v3 intermediate files, "
            "review input samples, and run manifests are written under <data-storage-root>/<dataset-root-name>/ "
            "while metadata paths stay portable for reviewer packages."
        ),
    )
    parser.add_argument(
        "--v3-output-root",
        default=DEFAULT_V3_OUTPUT_ROOT,
        help="Root for v3 generated intermediates such as 3d_candidates/fused_masks/pipeline_runs.",
    )
    parser.add_argument(
        "--metadata-root",
        default="",
        help=(
            "Path root used when writing portable paths into generated manifests/samples. "
            "Normally auto-filled to <data-storage-root>/<dataset-root-name>."
        ),
    )
    parser.add_argument(
        "--review-output-samples",
        default=DEFAULT_REVIEW_OUTPUT_SAMPLES,
        help="Review input JSONL produced by build stage.",
    )
    parser.add_argument(
        "--review-summary-json",
        default=DEFAULT_REVIEW_SUMMARY_JSON,
        help="Summary JSON produced by build stage.",
    )
    parser.add_argument(
        "--review-split-dir",
        default=DEFAULT_REVIEW_SPLIT_DIR,
        help="Split directory produced by build stage.",
    )
    parser.add_argument("--renders-root", default="processed/vlm_semantic_part/renders", help="VLM-friendly view render root.")
    parser.add_argument(
        "--part-proposal-backend",
        choices=["high_recall", "geometry", "partslippp"],
        default="high_recall",
        help="3D part proposal backend. high_recall is the default model-free candidate generator.",
    )
    parser.add_argument(
        "--partslippp-root",
        default="external/partslippp/outputs",
        help="Root containing external PartSLIP++ predictions for --part-proposal-backend partslippp.",
    )
    parser.add_argument(
        "--partslippp-path",
        default="",
        help="Optional explicit PartSLIP++ prediction path template.",
    )
    parser.add_argument(
        "--partslippp-fallback",
        choices=["error", "geometry"],
        default="error",
        help="Fallback when PartSLIP++ output is missing or malformed.",
    )
    parser.add_argument("--selected-candidates", default="", help="Optional manual candidate ids for build/visualize.")
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--box-shrink-ratio", type=float, default=0.0, help="Pass to 2D-to-3D projection.")
    parser.add_argument("--point-radius", type=int, default=4, help="Positive-point radius for projection.")
    parser.add_argument("--box-dilate-radius", type=int, default=2, help="Box dilation radius for projection.")
    parser.add_argument(
        "--max-target-box-area-fraction",
        type=float,
        default=0.35,
        help="Drop oversized target boxes in grounding when point prompts exist.",
    )
    parser.add_argument("--target-score-threshold", type=float, default=0.20, help="Target score threshold for v3 growth.")
    parser.add_argument("--reject-score-threshold", type=float, default=0.10, help="Reject veto score threshold for v3 growth.")
    parser.add_argument("--min-target-votes", type=float, default=1.0, help="Minimum target votes for v3 growth.")
    parser.add_argument("--min-reject-votes", type=float, default=1.0, help="Minimum reject votes for v3 veto.")
    parser.add_argument("--expand-hops", type=int, default=1, help="kNN expansion hops after target seed projection.")
    parser.add_argument("--k-neighbors", type=int, default=24, help="kNN neighborhood size for v3 growth.")
    parser.add_argument("--min-points", type=int, default=4, help="Minimum points per grown candidate.")
    parser.add_argument("--max-candidates", type=int, default=12, help="Maximum candidates shown in overlay rendering.")
    parser.add_argument("--proposal-max-candidates", type=int, default=64, help="Maximum candidates generated before overlay/top-k display.")
    parser.add_argument("--part-top-k", type=int, default=5, help="Top-k candidates retained per high-recall part group.")
    parser.add_argument("--dedupe-iou", type=float, default=0.985, help="Near-duplicate IoU threshold for high-recall candidates.")
    parser.add_argument("--small-part-max-fraction", type=float, default=0.10, help="Maximum object fraction for small-part proposals.")
    parser.add_argument("--min-selected-votes", type=int, default=1, help="Minimum VLM selected votes for part_filter.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--planner-dry-run", action="store_true", help="Run VLM-dependent stages as dry-run placeholders.")
    parser.add_argument(
        "--run-manifest",
        default=DEFAULT_RUN_MANIFEST,
        help="Run manifest path relative to dataset root.",
    )
    return parser.parse_args()


def join_path(root: str, *parts: str) -> str:
    return (Path(root) / Path(*parts)).as_posix()


def apply_storage_defaults(args: argparse.Namespace, dataset_root: Path) -> None:
    """Route generated review inputs/intermediates to external storage when requested."""
    if args.data_storage_root:
        storage_dataset_root = Path(args.data_storage_root) / dataset_root.name
        if args.v3_output_root == DEFAULT_V3_OUTPUT_ROOT:
            args.v3_output_root = (storage_dataset_root / DEFAULT_V3_OUTPUT_ROOT).as_posix()
        if args.review_output_samples == DEFAULT_REVIEW_OUTPUT_SAMPLES:
            args.review_output_samples = (storage_dataset_root / DEFAULT_REVIEW_OUTPUT_SAMPLES).as_posix()
        if args.review_summary_json == DEFAULT_REVIEW_SUMMARY_JSON:
            args.review_summary_json = (storage_dataset_root / DEFAULT_REVIEW_SUMMARY_JSON).as_posix()
        if args.review_split_dir == DEFAULT_REVIEW_SPLIT_DIR:
            args.review_split_dir = (storage_dataset_root / DEFAULT_REVIEW_SPLIT_DIR).as_posix()
        if not args.metadata_root:
            args.metadata_root = storage_dataset_root.as_posix()

    if args.filtered_pilot_csv == DEFAULT_FILTERED_PILOT_CSV:
        args.filtered_pilot_csv = join_path(args.v3_output_root, "pipeline_runs", "filtered_pilot_rows_latest.csv")
    if args.empty_review_csv == DEFAULT_EMPTY_REVIEW_CSV:
        args.empty_review_csv = join_path(args.v3_output_root, "pipeline_runs", "empty_review_rows_latest.csv")
    if args.run_manifest == DEFAULT_RUN_MANIFEST:
        args.run_manifest = join_path(args.v3_output_root, "pipeline_runs", "latest_run_manifest.json")


def v3_path(args: argparse.Namespace, *parts: str) -> str:
    return join_path(args.v3_output_root, *parts)


def stage_list(value: str) -> list[str]:
    allowed = ALL_STAGES
    stages = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in stages if item not in allowed]
    if invalid:
        raise ValueError(f"Invalid v3 stages: {invalid}. Available: {allowed}")
    return stages


def effective_stages(args: argparse.Namespace) -> list[str]:
    default_text = ",".join(DEFAULT_STAGES)
    if args.candidate_source == "grounding" and str(args.stages or "") == default_text:
        return list(GROUNDING_STAGES)
    return stage_list(args.stages)


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
        raise ValueError(f"Unknown tasks: {unknown}. Known tasks: {KNOWN_TASKS}")
    return tasks


def decision_selected(row: dict[str, str], include_decisions: str) -> bool:
    raw = str(include_decisions or "non_empty").strip()
    decision = str(row.get("decision") or "").strip()
    review_mode = str(row.get("review_mode") or "").strip()
    if raw.lower() == "all":
        return True
    if raw.lower() == "non_empty":
        return decision not in EMPTY_DECISIONS and review_mode != "confirm_empty"
    allowed = {item.strip() for item in raw.split(",") if item.strip()}
    if not allowed:
        return decision not in EMPTY_DECISIONS and review_mode != "confirm_empty"
    return decision in allowed


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Pilot CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def relative_to_dataset(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_samples_path(args: argparse.Namespace, dataset_root: Path, input_csv: Path, rows: list[dict[str, str]]) -> str:
    if str(args.samples or "").strip():
        return args.samples
    for row in rows:
        for key in ("samples_path", "source_samples_path", "input_samples"):
            value = str(row.get(key) or "").strip()
            if value:
                return value
    input_rel = relative_to_dataset(dataset_root, input_csv)
    for summary_path in sorted(input_csv.parent.glob("*summary*.json")):
        try:
            summary = read_json(summary_path)
        except Exception:
            continue
        output_csv = str(summary.get("output_csv") or "")
        if output_csv == input_rel or Path(output_csv).name == input_csv.name:
            input_samples = str(summary.get("input_samples") or "")
            if input_samples:
                return input_samples
    fallback = dataset_root / "processed/metadata/samples_v3_large_batch_v0_1.jsonl"
    if fallback.exists():
        return relative_to_dataset(dataset_root, fallback)
    return "processed/metadata/samples_checked_v0_1.jsonl"


def prepare_pilot_csv(args: argparse.Namespace, dataset_root: Path) -> dict[str, Any]:
    include_tasks = parse_task_filter(args.include_tasks, allow_all=True)
    exclude_tasks = parse_task_filter(args.exclude_tasks, allow_all=False) or set()
    input_csv = resolve_path(dataset_root, args.pilot_csv)
    rows = read_csv(input_csv)
    args.effective_samples = infer_samples_path(args, dataset_root, input_csv, rows)
    if include_tasks is None and not exclude_tasks and str(args.include_decisions).lower() == "all":
        args.effective_pilot_csv = args.pilot_csv
        args.effective_empty_review_csv = ""
        return {
            "input_pilot_csv": relative_to_dataset(dataset_root, input_csv),
            "effective_pilot_csv": args.pilot_csv,
            "empty_review_csv": "",
            "samples": args.effective_samples,
            "task_filter_active": False,
            "decision_filter_active": False,
            "include_tasks": "all",
            "exclude_tasks": [],
            "rows_before_task_filter": len(rows),
            "rows_after_task_filter": len(rows),
            "rows_for_vlm_candidate_stages": len(rows),
            "rows_for_empty_review": 0,
        }

    task_filtered = []
    for row in rows:
        task = str(row.get("task", ""))
        if include_tasks is not None and task not in include_tasks:
            continue
        if task in exclude_tasks:
            continue
        task_filtered.append(row)
    if not task_filtered:
        raise ValueError(
            "No pilot rows remain after task filtering. "
            f"include_tasks={sorted(include_tasks) if include_tasks is not None else 'all'}, "
            f"exclude_tasks={sorted(exclude_tasks)}"
        )
    filtered = [row for row in task_filtered if decision_selected(row, args.include_decisions)]
    empty_rows = [row for row in task_filtered if not decision_selected(row, args.include_decisions)]
    stages = effective_stages(args)
    if not filtered and any(stage in stages for stage in ("views", "plan", "ground", "project", "grow", "part_propose", "render", "part_select", "part_filter", "coverage")):
        raise ValueError(
            "No non-empty rows remain for VLM/candidate stages after decision filtering. "
            "Empty-review rows are passed to build only."
        )
    out_csv = resolve_path(dataset_root, args.filtered_pilot_csv)
    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(out_csv, filtered, fieldnames)
    empty_csv = resolve_path(dataset_root, args.empty_review_csv)
    if empty_rows:
        write_csv(empty_csv, empty_rows, fieldnames)
        args.effective_empty_review_csv = relative_to_dataset(dataset_root, empty_csv)
    else:
        args.effective_empty_review_csv = ""
    args.effective_pilot_csv = relative_to_dataset(dataset_root, out_csv)
    return {
        "input_pilot_csv": relative_to_dataset(dataset_root, input_csv),
        "effective_pilot_csv": args.effective_pilot_csv,
        "empty_review_csv": args.effective_empty_review_csv,
        "samples": args.effective_samples,
        "task_filter_active": True,
        "decision_filter_active": str(args.include_decisions).lower() != "all",
        "include_decisions": args.include_decisions,
        "include_tasks": sorted(include_tasks) if include_tasks is not None else "all",
        "exclude_tasks": sorted(exclude_tasks),
        "rows_before_task_filter": len(rows),
        "rows_after_task_filter": len(task_filtered),
        "rows_for_vlm_candidate_stages": len(filtered),
        "rows_for_empty_review": len(empty_rows),
    }


def script(root: Path, name: str) -> str:
    return str(root / "tools" / name)


def add_common(cmd: list[str], args: argparse.Namespace) -> list[str]:
    cmd.extend(["--dataset-root", args.dataset_root])
    cmd.extend(["--pilot-csv", getattr(args, "effective_pilot_csv", args.pilot_csv)])
    if args.pilot_id:
        cmd.extend(["--pilot-id", args.pilot_id])
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def build_commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    root = Path(args.dataset_root)
    stages = effective_stages(args)
    commands: list[tuple[str, list[str]]] = []
    if "views" in stages or any(stage in stages for stage in ("plan", "ground", "project", "grow", "render", "part_select", "coverage")):
        cmd = add_common([sys.executable, script(root, "render_vlm_friendly_views.py"), "--output-root", args.renders_root], args)
        if getattr(args, "effective_samples", ""):
            cmd.extend(["--samples", args.effective_samples])
        commands.append(("views", cmd))
    if "plan" in stages:
        cmd = add_common(
            [
                sys.executable,
                script(root, "run_v3_semantic_part_planner.py"),
                "--config",
                args.config,
                "--renders-root",
                args.renders_root,
            ],
            args,
        )
        if args.planner_dry_run:
            cmd.append("--dry-run")
        commands.append(("plan", cmd))
    if "ground" in stages:
        cmd = add_common(
            [
                sys.executable,
                script(root, "run_v3_target_reject_grounding.py"),
                "--config",
                args.config,
                "--renders-root",
                args.renders_root,
                "--max-target-box-area-fraction",
                str(args.max_target_box_area_fraction),
            ],
            args,
        )
        if args.planner_dry_run:
            cmd.append("--dry-run")
        commands.append(("ground", cmd))
    if "project" in stages:
        commands.append(
            (
                "project",
                add_common(
                    [
                        sys.executable,
                        script(root, "project_v3_grounding_to_3d.py"),
                        "--renders-root",
                        args.renders_root,
                        "--box-shrink-ratio",
                        str(args.box_shrink_ratio),
                        "--point-radius",
                        str(args.point_radius),
                        "--box-dilate-radius",
                        str(args.box_dilate_radius),
                    ],
                    args,
                ),
            )
        )
    if "grow" in stages:
        commands.append(
            (
                "grow",
                add_common(
                    [
                        sys.executable,
                        script(root, "grow_v3_reject_aware_candidates.py"),
                        "--target-score-threshold",
                        str(args.target_score_threshold),
                        "--reject-score-threshold",
                        str(args.reject_score_threshold),
                        "--min-target-votes",
                        str(args.min_target_votes),
                        "--min-reject-votes",
                        str(args.min_reject_votes),
                        "--expand-hops",
                        str(args.expand_hops),
                        "--k-neighbors",
                        str(args.k_neighbors),
                        "--min-points",
                        str(args.min_points),
                    ],
                    args,
                ),
            )
        )
    if "part_propose" in stages:
        cmd = add_common(
            [
                sys.executable,
                script(root, "propose_v3_part_candidates.py"),
                "--output-root",
                v3_path(args, "3d_candidates"),
                "--backend",
                args.part_proposal_backend,
                "--renders-root",
                args.renders_root,
                "--k-neighbors",
                str(args.k_neighbors),
                "--min-points",
                str(args.min_points),
                "--max-candidates",
                str(max(args.proposal_max_candidates, args.max_candidates)),
                "--seed-expand-hops",
                str(args.expand_hops),
                "--part-top-k",
                str(args.part_top_k),
                "--dedupe-iou",
                str(args.dedupe_iou),
                "--small-part-max-fraction",
                str(args.small_part_max_fraction),
                "--partslippp-root",
                args.partslippp_root,
                "--partslippp-path",
                args.partslippp_path,
                "--partslippp-fallback",
                args.partslippp_fallback,
            ],
            args,
        )
        if args.metadata_root:
            cmd.extend(["--metadata-root", args.metadata_root])
        if getattr(args, "effective_samples", ""):
            cmd.extend(["--samples", args.effective_samples])
        commands.append(("part_propose", cmd))
    if "render" in stages:
        cmd = add_common(
            [
                sys.executable,
                script(root, "render_candidate_overlays_v2.py"),
                "--candidate-root",
                v3_path(args, "3d_candidates"),
                "--renders-root",
                args.renders_root,
                "--fallback-renders-root",
                args.renders_root,
                "--output-root",
                v3_path(args, "candidate_overlays"),
                "--max-candidates",
                str(args.max_candidates),
            ],
            args,
        )
        commands.append(("render", cmd))
    if "part_select" in stages:
        cmd = add_common(
            [
                sys.executable,
                script(root, "run_vlm_candidate_selection_v2.py"),
                "--config",
                args.config,
                "--overlay-root",
                v3_path(args, "candidate_overlays"),
                "--selection-root",
                v3_path(args, "vlm_selection"),
                "--part-plan-root",
                v3_path(args, "semantic_plans"),
                "--update-candidate-manifest",
            ],
            args,
        )
        if args.planner_dry_run:
            cmd.append("--dry-run")
        commands.append(("part_select", cmd))
    if "part_filter" in stages:
        cmd = add_common(
            [
                sys.executable,
                script(root, "filter_candidates_by_executor_rules.py"),
                "--candidate-root",
                v3_path(args, "3d_candidates"),
                "--selection-root",
                v3_path(args, "vlm_selection"),
                "--output-root",
                v3_path(args, "rule_filter"),
                "--min-selected-votes",
                str(args.min_selected_votes),
                "--update-candidate-manifest",
            ],
            args,
        )
        commands.append(("part_filter", cmd))
    if "coverage" in stages:
        cmd = add_common(
            [
                sys.executable,
                script(root, "run_vlm_coverage_check_v2.py"),
                "--config",
                args.config,
                "--candidate-root",
                v3_path(args, "3d_candidates"),
                "--overlay-root",
                v3_path(args, "candidate_overlays"),
                "--part-plan-root",
                v3_path(args, "semantic_plans"),
                "--output-root",
                v3_path(args, "coverage_check"),
            ],
            args,
        )
        if args.planner_dry_run:
            cmd.append("--dry-run")
        commands.append(("coverage", cmd))
    if "build" in stages:
        cmd = add_common([sys.executable, script(root, "build_v3_candidate_masks.py")], args)
        cmd.extend(
            [
                "--include-tasks",
                args.include_tasks,
                "--exclude-tasks",
                args.exclude_tasks,
                "--candidate-root",
                v3_path(args, "3d_candidates"),
                "--output-mask-root",
                v3_path(args, "fused_masks"),
                "--output-samples",
                args.review_output_samples,
                "--summary-json",
                args.review_summary_json,
                "--output-split-dir",
                args.review_split_dir,
            ]
        )
        if args.metadata_root:
            cmd.extend(["--metadata-root", args.metadata_root])
        if getattr(args, "effective_samples", ""):
            cmd.extend(["--samples", args.effective_samples])
        if getattr(args, "effective_empty_review_csv", ""):
            cmd.extend(["--empty-pilot-csv", args.effective_empty_review_csv])
        if args.selected_candidates:
            cmd.extend(["--selected-candidates", args.selected_candidates])
        if args.allow_empty:
            cmd.append("--allow-empty")
        commands.append(("build", cmd))
    if "visualize" in stages:
        if not args.pilot_id:
            raise ValueError("visualize stage requires --pilot-id.")
        cmd = add_common(
            [
                sys.executable,
                script(root, "visualize_v2_candidates.py"),
                "--candidate-root",
                v3_path(args, "3d_candidates"),
                "--output-root",
                v3_path(args, "review_visualizations"),
            ],
            args,
        )
        if args.selected_candidates:
            cmd.extend(["--selected-candidates", args.selected_candidates])
        commands.append(("visualize", cmd))
    return commands


def manifest_path(dataset_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else dataset_root / path


def write_manifest(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    apply_storage_defaults(args, dataset_root)
    pilot_filter = prepare_pilot_csv(args, dataset_root)
    commands = build_commands(args)
    run_log: list[dict[str, Any]] = []
    for stage, cmd in commands:
        print(f"\n[stage:{stage}] {' '.join(cmd)}", flush=True)
        record: dict[str, Any] = {"stage": stage, "command": cmd}
        if args.dry_run:
            record["returncode"] = None
            record["status"] = "dry_run"
            run_log.append(record)
            continue
        completed = subprocess.run(cmd, cwd=Path.cwd(), check=False)
        record["returncode"] = completed.returncode
        record["status"] = "ok" if completed.returncode == 0 else "failed"
        run_log.append(record)
        if completed.returncode != 0:
            write_manifest(
                manifest_path(dataset_root, args.run_manifest),
                {
                    "version": "v3",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "dataset_root": str(dataset_root),
                    "pilot_id": args.pilot_id,
                    "stages": effective_stages(args),
                    "pilot_filter": pilot_filter,
                    "status": "failed",
                    "runs": run_log,
                },
            )
            return completed.returncode
    manifest = {
        "version": "v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "pilot_id": args.pilot_id,
        "stages": effective_stages(args),
        "pilot_filter": pilot_filter,
        "selected_candidates": args.selected_candidates,
        "planner_dry_run": args.planner_dry_run,
        "parameters": {
            "box_shrink_ratio": args.box_shrink_ratio,
            "point_radius": args.point_radius,
            "box_dilate_radius": args.box_dilate_radius,
            "max_target_box_area_fraction": args.max_target_box_area_fraction,
            "target_score_threshold": args.target_score_threshold,
            "reject_score_threshold": args.reject_score_threshold,
            "min_target_votes": args.min_target_votes,
            "min_reject_votes": args.min_reject_votes,
            "expand_hops": args.expand_hops,
            "k_neighbors": args.k_neighbors,
            "min_points": args.min_points,
            "max_candidates": args.max_candidates,
            "proposal_max_candidates": args.proposal_max_candidates,
            "part_top_k": args.part_top_k,
            "dedupe_iou": args.dedupe_iou,
            "small_part_max_fraction": args.small_part_max_fraction,
            "candidate_source": args.candidate_source,
            "renders_root": args.renders_root,
            "part_proposal_backend": args.part_proposal_backend,
            "partslippp_root": args.partslippp_root,
            "partslippp_path": args.partslippp_path,
            "partslippp_fallback": args.partslippp_fallback,
            "min_selected_votes": args.min_selected_votes,
        },
        "status": "dry_run" if args.dry_run else "ok",
        "runs": run_log,
    }
    out_path = manifest_path(dataset_root, args.run_manifest)
    write_manifest(out_path, manifest)
    print(json.dumps({"status": manifest["status"], "run_manifest": str(out_path)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
