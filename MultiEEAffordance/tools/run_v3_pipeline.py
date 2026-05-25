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


DEFAULT_STAGES = ["views", "plan", "ground", "project", "grow", "render", "build"]
KNOWN_TASKS = ["pick_up", "lift_carry", "open_pull", "press_push"]
DEFAULT_ACTIVE_TASKS = ["pick_up", "open_pull", "press_push"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v3 target/reject candidate pipeline.")
    parser.add_argument("--dataset-root", default="MultiEEAffordance", help="Dataset root directory.")
    parser.add_argument("--config", default="configs/qwen3vl_sam2_pilot.yaml", help="Qwen config relative to dataset root.")
    parser.add_argument("--pilot-csv", default="processed/metadata/vlm_pilot_samples_v0_1.csv")
    parser.add_argument(
        "--include-tasks",
        default=",".join(DEFAULT_ACTIVE_TASKS),
        help="Comma-separated tasks to keep, or 'all'. Default excludes lift_carry.",
    )
    parser.add_argument(
        "--exclude-tasks",
        default="",
        help="Comma-separated tasks to drop after include filtering.",
    )
    parser.add_argument(
        "--filtered-pilot-csv",
        default="processed/vlm_candidate_v3/pipeline_runs/filtered_pilot_rows_latest.csv",
        help="Filtered pilot CSV written relative to dataset root when task filtering is active.",
    )
    parser.add_argument("--pilot-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stages", default=",".join(DEFAULT_STAGES), help=f"Comma-separated stages: {','.join(DEFAULT_STAGES + ['visualize'])}")
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
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--planner-dry-run", action="store_true", help="Run VLM-dependent stages as dry-run placeholders.")
    parser.add_argument(
        "--run-manifest",
        default="processed/vlm_candidate_v3/pipeline_runs/latest_run_manifest.json",
        help="Run manifest path relative to dataset root.",
    )
    return parser.parse_args()


def stage_list(value: str) -> list[str]:
    allowed = DEFAULT_STAGES + ["visualize"]
    stages = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in stages if item not in allowed]
    if invalid:
        raise ValueError(f"Invalid v3 stages: {invalid}. Available: {allowed}")
    return stages


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


def prepare_pilot_csv(args: argparse.Namespace, dataset_root: Path) -> dict[str, Any]:
    include_tasks = parse_task_filter(args.include_tasks, allow_all=True)
    exclude_tasks = parse_task_filter(args.exclude_tasks, allow_all=False) or set()
    input_csv = resolve_path(dataset_root, args.pilot_csv)
    rows = read_csv(input_csv)
    if include_tasks is None and not exclude_tasks:
        args.effective_pilot_csv = args.pilot_csv
        return {
            "input_pilot_csv": relative_to_dataset(dataset_root, input_csv),
            "effective_pilot_csv": args.pilot_csv,
            "task_filter_active": False,
            "include_tasks": "all",
            "exclude_tasks": [],
            "rows_before_task_filter": len(rows),
            "rows_after_task_filter": len(rows),
        }

    filtered = []
    for row in rows:
        task = str(row.get("task", ""))
        if include_tasks is not None and task not in include_tasks:
            continue
        if task in exclude_tasks:
            continue
        filtered.append(row)
    if not filtered:
        raise ValueError(
            "No pilot rows remain after task filtering. "
            f"include_tasks={sorted(include_tasks) if include_tasks is not None else 'all'}, "
            f"exclude_tasks={sorted(exclude_tasks)}"
        )
    out_csv = resolve_path(dataset_root, args.filtered_pilot_csv)
    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(out_csv, filtered, fieldnames)
    args.effective_pilot_csv = relative_to_dataset(dataset_root, out_csv)
    return {
        "input_pilot_csv": relative_to_dataset(dataset_root, input_csv),
        "effective_pilot_csv": args.effective_pilot_csv,
        "task_filter_active": True,
        "include_tasks": sorted(include_tasks) if include_tasks is not None else "all",
        "exclude_tasks": sorted(exclude_tasks),
        "rows_before_task_filter": len(rows),
        "rows_after_task_filter": len(filtered),
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
    stages = stage_list(args.stages)
    commands: list[tuple[str, list[str]]] = []
    if "views" in stages or any(stage in stages for stage in ("plan", "ground", "project", "grow", "render", "build")):
        commands.append(("views", add_common([sys.executable, script(root, "render_vlm_friendly_views.py")], args)))
    if "plan" in stages:
        cmd = add_common([sys.executable, script(root, "run_v3_semantic_part_planner.py"), "--config", args.config], args)
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
    if "render" in stages:
        cmd = add_common(
            [
                sys.executable,
                script(root, "render_candidate_overlays_v2.py"),
                "--candidate-root",
                "processed/vlm_candidate_v3/3d_candidates",
                "--output-root",
                "processed/vlm_candidate_v3/candidate_overlays",
                "--max-candidates",
                str(args.max_candidates),
            ],
            args,
        )
        commands.append(("render", cmd))
    if "build" in stages:
        cmd = add_common([sys.executable, script(root, "build_v3_candidate_masks.py")], args)
        cmd.extend(["--include-tasks", args.include_tasks, "--exclude-tasks", args.exclude_tasks])
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
                "processed/vlm_candidate_v3/3d_candidates",
                "--output-root",
                "processed/vlm_candidate_v3/review_visualizations",
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
                    "stages": stage_list(args.stages),
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
        "stages": stage_list(args.stages),
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
