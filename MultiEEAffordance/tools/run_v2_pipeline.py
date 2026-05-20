#!/usr/bin/env python3
"""Run the modular v2 candidate-labeling pipeline.

This wrapper keeps the candidate-generation, VLM-selection, rule-filtering,
mask-building, and review-visualization steps in one reproducible command.
It does not replace the individual scripts; it records and runs them in order.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_STAGES = ["generate", "render", "select", "filter", "build", "visualize"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run modular v2 candidate pipeline.")
    parser.add_argument("--dataset-root", default="MultiEEAffordance", help="Dataset root directory.")
    parser.add_argument("--config", default="configs/qwen3vl_sam2_pilot.yaml", help="Qwen config relative to dataset root.")
    parser.add_argument("--pilot-id", default=None, help="Run only one pilot id.")
    parser.add_argument("--limit", type=int, default=None, help="Limit pilot rows.")
    parser.add_argument(
        "--stages",
        default=",".join(DEFAULT_STAGES),
        help=f"Comma-separated stages. Available: {','.join(DEFAULT_STAGES)}.",
    )
    parser.add_argument("--min-selected-votes", type=int, default=8, help="Rule-filter VLM vote threshold.")
    parser.add_argument("--selected-candidates", default="", help="Manual candidate ids for mask building/review visualization.")
    parser.add_argument("--include-uncertain", action="store_true", help="Include uncertain candidates when building masks.")
    parser.add_argument("--allow-empty", action="store_true", help="Allow empty candidate mask.")
    parser.add_argument("--overwrite", action="store_true", help="Pass --overwrite to child scripts.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument(
        "--run-manifest",
        default="processed/vlm_candidate_v2/pipeline_runs/latest_run_manifest.json",
        help="Run manifest path relative to dataset root.",
    )
    return parser.parse_args()


def stage_list(value: str) -> list[str]:
    stages = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in stages if item not in DEFAULT_STAGES]
    if invalid:
        raise ValueError(f"Invalid stages: {invalid}. Available: {DEFAULT_STAGES}")
    return stages


def script(root: Path, name: str) -> str:
    return str(root / "tools" / name)


def add_common(cmd: list[str], args: argparse.Namespace) -> list[str]:
    cmd.extend(["--dataset-root", args.dataset_root])
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
    if "generate" in stages:
        commands.append(("generate", add_common([sys.executable, script(root, "generate_3d_candidate_regions.py")], args)))
    if "render" in stages:
        commands.append(("render", add_common([sys.executable, script(root, "render_candidate_overlays_v2.py")], args)))
    if "select" in stages:
        cmd = add_common([sys.executable, script(root, "run_vlm_candidate_selection_v2.py"), "--config", args.config], args)
        commands.append(("select", cmd))
    if "filter" in stages:
        cmd = add_common(
            [
                sys.executable,
                script(root, "filter_candidates_by_executor_rules.py"),
                "--min-selected-votes",
                str(args.min_selected_votes),
            ],
            args,
        )
        commands.append(("filter", cmd))
    if "build" in stages:
        cmd = add_common([sys.executable, script(root, "build_v2_candidate_masks.py")], args)
        if args.selected_candidates:
            cmd.extend(["--selected-candidates", args.selected_candidates])
        if args.include_uncertain:
            cmd.append("--include-uncertain")
        if args.allow_empty:
            cmd.append("--allow-empty")
        commands.append(("build", cmd))
    if "visualize" in stages:
        cmd = add_common([sys.executable, script(root, "visualize_v2_candidates.py")], args)
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
                    "version": "v2",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "dataset_root": str(dataset_root),
                    "pilot_id": args.pilot_id,
                    "stages": stage_list(args.stages),
                    "status": "failed",
                    "runs": run_log,
                },
            )
            return completed.returncode
    manifest = {
        "version": "v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "pilot_id": args.pilot_id,
        "stages": stage_list(args.stages),
        "selected_candidates": args.selected_candidates,
        "min_selected_votes": args.min_selected_votes,
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
