#!/usr/bin/env python3
"""Calibrate thresholds and evaluate five-task Multi-EE models.

This script keeps the trained model unchanged. It uses the validation split to
choose executor-wise mask and feasibility thresholds, then evaluates the test
split with optional feasibility-gated mask prediction.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from .constants import EXECUTORS
from .metrics import MetricAccumulator
from .model_factory import build_model as build_config_model
from .model_factory import load_model_state
from .train import apply_enabled_executors, load_config, make_loader, resolve, save_json


def parse_grid(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("Threshold grid is empty.")
    for threshold in values:
        if threshold <= 0.0 or threshold >= 1.0:
            raise ValueError(f"Threshold must be inside (0, 1), got {threshold}.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate thresholds and evaluate a five-task baseline.")
    parser.add_argument("--config", required=True, help="Training config JSON.")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint, usually best.pt.")
    parser.add_argument("--dataset-root", default="", help="Override dataset_root from config.")
    parser.add_argument("--val-manifest", default="", help="Optional validation manifest override.")
    parser.add_argument("--test-manifest", default="", help="Optional test manifest override.")
    parser.add_argument("--output-json", required=True, help="Output calibrated metrics JSON.")
    parser.add_argument("--output-matrix-csv", default="", help="Optional 5x4 task-executor matrix CSV.")
    parser.add_argument("--thresholds-json", default="", help="Optional threshold JSON to load or write.")
    parser.add_argument("--reuse-thresholds", action="store_true", help="Load --thresholds-json instead of recalibrating.")
    parser.add_argument("--mask-threshold-grid", default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90")
    parser.add_argument("--feasibility-threshold-grid", default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90")
    parser.add_argument("--feasibility-gate", action="store_true", help="Force predicted masks to zero when predicted infeasible.")
    parser.add_argument("--small-part-max-fraction", type=float, default=0.02)
    parser.add_argument("--device", default="")
    return parser.parse_args()


def build_loaded_model(config: dict[str, Any], checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = build_config_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    load_model_state(model, state)
    model.eval()
    return model


def collect_logits(
    model: torch.nn.Module,
    loader,
    *,
    device: torch.device,
    config: dict[str, Any],
) -> list[dict[str, torch.Tensor]]:
    rows: list[dict[str, torch.Tensor]] = []
    with torch.no_grad():
        for batch in loader:
            points = batch["points"].to(device)
            task_id = batch["task_id"].to(device)
            tensor_batch = {
                "masks": batch["masks"].to(device),
                "channel_supervision": batch["channel_supervision"].to(device),
                "feasibility": batch["feasibility"].to(device),
            }
            apply_enabled_executors(tensor_batch, config)
            outputs = model(points, task_id)
            rows.append(
                {
                    "mask_prob": torch.sigmoid(outputs["mask_logits"]).detach().cpu(),
                    "mask": tensor_batch["masks"].detach().cpu(),
                    "feasibility_prob": torch.sigmoid(outputs["feasibility_logits"]).detach().cpu(),
                    "feasibility": tensor_batch["feasibility"].detach().cpu(),
                    "supervision": tensor_batch["channel_supervision"].detach().cpu(),
                }
            )
    return rows


def _f1_for_threshold(scores: torch.Tensor, labels: torch.Tensor, threshold: float) -> float:
    pred = scores >= threshold
    target = labels >= 0.5
    tp = float((pred & target).sum())
    fp = float((pred & ~target).sum())
    fn = float((~pred & target).sum())
    denom = 2.0 * tp + fp + fn
    return 0.0 if denom <= 0.0 else (2.0 * tp / denom)


def _global_iou_for_threshold(scores: torch.Tensor, labels: torch.Tensor, threshold: float) -> float:
    pred = scores >= threshold
    target = labels >= 0.5
    intersection = float((pred & target).sum())
    union = float((pred | target).sum())
    return 0.0 if union <= 0.0 else intersection / union


def calibrate_thresholds(
    rows: list[dict[str, torch.Tensor]],
    *,
    mask_grid: list[float],
    feasibility_grid: list[float],
) -> dict[str, Any]:
    mask_thresholds: list[float] = []
    feasibility_thresholds: list[float] = []
    per_executor: dict[str, Any] = {}
    for executor_index, executor in enumerate(EXECUTORS):
        supervised_scores = []
        supervised_labels = []
        feasible_mask_scores = []
        feasible_mask_labels = []
        for row in rows:
            supervised = row["supervision"][:, executor_index] >= 0.5
            if bool(supervised.any()):
                supervised_scores.append(row["feasibility_prob"][:, executor_index][supervised])
                supervised_labels.append(row["feasibility"][:, executor_index][supervised])
            feasible = supervised & (row["feasibility"][:, executor_index] >= 0.5)
            if bool(feasible.any()):
                feasible_mask_scores.append(row["mask_prob"][:, :, executor_index][feasible])
                feasible_mask_labels.append(row["mask"][:, :, executor_index][feasible])

        if supervised_scores:
            f_scores = torch.cat(supervised_scores)
            f_labels = torch.cat(supervised_labels)
            best_f_threshold, best_f_score = max(
                ((threshold, _f1_for_threshold(f_scores, f_labels, threshold)) for threshold in feasibility_grid),
                key=lambda item: (item[1], -abs(item[0] - 0.5)),
            )
        else:
            best_f_threshold, best_f_score = 0.5, 0.0

        if feasible_mask_scores:
            m_scores = torch.cat(feasible_mask_scores, dim=0)
            m_labels = torch.cat(feasible_mask_labels, dim=0)
            best_m_threshold, best_m_score = max(
                ((threshold, _global_iou_for_threshold(m_scores, m_labels, threshold)) for threshold in mask_grid),
                key=lambda item: (item[1], -abs(item[0] - 0.5)),
            )
        else:
            best_m_threshold, best_m_score = 0.5, 0.0

        mask_thresholds.append(float(best_m_threshold))
        feasibility_thresholds.append(float(best_f_threshold))
        per_executor[executor] = {
            "mask_threshold": float(best_m_threshold),
            "mask_calibration_iou": float(best_m_score),
            "feasibility_threshold": float(best_f_threshold),
            "feasibility_calibration_f1": float(best_f_score),
            "supervised_samples": int(sum(int((row["supervision"][:, executor_index] >= 0.5).sum()) for row in rows)),
            "feasible_samples": int(
                sum(
                    int(
                        (
                            (row["supervision"][:, executor_index] >= 0.5)
                            & (row["feasibility"][:, executor_index] >= 0.5)
                        ).sum()
                    )
                    for row in rows
                )
            ),
        }
    return {
        "mask_thresholds": mask_thresholds,
        "feasibility_thresholds": feasibility_thresholds,
        "per_executor": per_executor,
        "mask_threshold_grid": mask_grid,
        "feasibility_threshold_grid": feasibility_grid,
    }


def evaluate(
    model: torch.nn.Module,
    loader,
    *,
    device: torch.device,
    config: dict[str, Any],
    mask_thresholds: list[float],
    feasibility_thresholds: list[float],
    feasibility_gate: bool,
    small_part_max_fraction: float,
) -> dict[str, Any]:
    metric = MetricAccumulator(
        small_part_max_fraction=small_part_max_fraction,
        mask_thresholds=mask_thresholds,
        feasibility_thresholds=feasibility_thresholds,
        feasibility_gate=feasibility_gate,
    )
    with torch.no_grad():
        for batch in loader:
            points = batch["points"].to(device)
            task_id = batch["task_id"].to(device)
            tensor_batch = {
                "masks": batch["masks"].to(device),
                "channel_supervision": batch["channel_supervision"].to(device),
                "feasibility": batch["feasibility"].to(device),
            }
            apply_enabled_executors(tensor_batch, config)
            outputs = model(points, task_id)
            metric.update(
                outputs["mask_logits"],
                tensor_batch["masks"],
                outputs["feasibility_logits"],
                tensor_batch["feasibility"],
                tensor_batch["channel_supervision"],
                task_id,
            )
    return metric.compute()


def write_matrix_csv(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["task", "executor", "iou", "dice", "feasible_samples", "supervised_samples"],
        )
        writer.writeheader()
        matrix = metrics.get("task_executor_iou", {})
        if not isinstance(matrix, dict):
            return
        for task, row in matrix.items():
            if not isinstance(row, dict):
                continue
            for executor in EXECUTORS:
                cell = row.get(executor, {})
                if not isinstance(cell, dict):
                    cell = {}
                writer.writerow(
                    {
                        "task": task,
                        "executor": executor,
                        "iou": cell.get("iou", ""),
                        "dice": cell.get("dice", ""),
                        "feasible_samples": cell.get("feasible_samples", ""),
                        "supervised_samples": cell.get("supervised_samples", ""),
                    }
                )


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    dataset_root = Path(args.dataset_root or config["dataset_root"]).resolve()
    if args.val_manifest:
        config["val_manifest"] = args.val_manifest
    if args.test_manifest:
        config["test_manifest"] = args.test_manifest
    device = torch.device(args.device or config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_loaded_model(config, resolve(dataset_root, args.checkpoint), device)

    thresholds_path = resolve(dataset_root, args.thresholds_json) if args.thresholds_json else None
    if args.reuse_thresholds:
        if thresholds_path is None or not thresholds_path.exists():
            raise FileNotFoundError("--reuse-thresholds requires an existing --thresholds-json.")
        thresholds = json.loads(thresholds_path.read_text(encoding="utf-8-sig"))
    else:
        val_loader = make_loader(config, dataset_root, "val_manifest", train=False)
        if val_loader is None:
            raise ValueError("Validation manifest contains no rows; cannot calibrate thresholds.")
        thresholds = calibrate_thresholds(
            collect_logits(model, val_loader, device=device, config=config),
            mask_grid=parse_grid(args.mask_threshold_grid),
            feasibility_grid=parse_grid(args.feasibility_threshold_grid),
        )
        if thresholds_path is not None:
            save_json(thresholds_path, thresholds)

    test_loader = make_loader(config, dataset_root, "test_manifest", train=False)
    if test_loader is None:
        raise ValueError("Test manifest contains no rows.")
    metrics = evaluate(
        model,
        test_loader,
        device=device,
        config=config,
        mask_thresholds=[float(value) for value in thresholds["mask_thresholds"]],
        feasibility_thresholds=[float(value) for value in thresholds["feasibility_thresholds"]],
        feasibility_gate=bool(args.feasibility_gate),
        small_part_max_fraction=float(args.small_part_max_fraction),
    )
    payload = {
        "config": str(Path(args.config)),
        "checkpoint": str(Path(args.checkpoint)),
        "dataset_root": dataset_root.as_posix(),
        "feasibility_gate": bool(args.feasibility_gate),
        "thresholds": thresholds,
        "metrics": metrics,
    }
    output_json = resolve(dataset_root, args.output_json)
    save_json(output_json, payload)
    matrix_csv = resolve(dataset_root, args.output_matrix_csv) if args.output_matrix_csv else output_json.with_name(output_json.stem + "_task_executor_matrix.csv")
    write_matrix_csv(metrics, matrix_csv)
    print(json.dumps({"output_json": output_json.as_posix(), "matrix_csv": matrix_csv.as_posix()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
