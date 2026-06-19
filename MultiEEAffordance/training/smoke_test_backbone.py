#!/usr/bin/env python3
"""Smoke test for interchangeable point backbones."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .dataset import MultiEEFiveTaskDataset
from .model_factory import build_model
from .train import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight backbone/model forward smoke test.")
    parser.add_argument("--config", default="", help="Optional training config JSON.")
    parser.add_argument("--backbone", default="", help="Override backbone_name, e.g. pointnet_mlp or pointnext.")
    parser.add_argument("--dataset-root", default="", help="Optional dataset root for real-batch smoke test.")
    parser.add_argument("--manifest", default="", help="Optional manifest path relative to dataset root.")
    parser.add_argument("--pointnext-root", default="", help="Optional external PointNeXt root.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--input-channels", type=int, default=3)
    parser.add_argument("--device", default="")
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config) if args.config else {}
    if args.backbone:
        config["backbone_name"] = args.backbone
    if args.pointnext_root:
        config["pointnext_root"] = args.pointnext_root
    config.setdefault("input_channels", args.input_channels)
    config.setdefault("hidden_dim", 128)
    config.setdefault("task_dim", 64)
    config.setdefault("executor_dim", 64)
    config.setdefault("executor_condition_mode", config.get("executor_mode", "learnable_id"))
    config.setdefault("pointnext_width", 16)
    config.setdefault("pointnext_blocks", [1, 1, 1])
    config.setdefault("pointnext_strides", [1, 2, 2])
    config.setdefault("pointnext_decoder_stages", 2)
    config.setdefault("pointnext_decoder_layers", 1)
    return config


def load_batch(args: argparse.Namespace, config: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if args.dataset_root and args.manifest:
        dataset = MultiEEFiveTaskDataset(
            Path(args.dataset_root),
            args.manifest,
            sample_size=int(args.sample_size),
            train=False,
            input_channels=int(config.get("input_channels", args.input_channels)),
        )
        loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=0)
        batch = next(iter(loader))
        return batch["points"].to(device), batch["task_id"].to(device)
    points = torch.randn(
        int(args.batch_size),
        int(args.sample_size),
        int(config.get("input_channels", args.input_channels)),
        device=device,
    )
    task_id = torch.zeros(int(args.batch_size), dtype=torch.long, device=device)
    return points, task_id


def main() -> int:
    args = parse_args()
    config = make_config(args)
    device = torch.device(args.device or config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(config).to(device)
    model.eval()
    points, task_id = load_batch(args, config, device)
    with torch.no_grad():
        outputs = model(points, task_id)
    payload = {
        "status": "ok",
        "device": str(device),
        "backbone_name": config.get("backbone_name", "pointnet_mlp"),
        "executor_condition_mode": config.get("executor_condition_mode", config.get("executor_mode", "learnable_id")),
        "points_shape": list(points.shape),
        "mask_logits_shape": list(outputs["mask_logits"].shape),
        "feasibility_logits_shape": list(outputs["feasibility_logits"].shape),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
