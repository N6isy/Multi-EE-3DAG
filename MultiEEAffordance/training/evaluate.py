#!/usr/bin/env python3
"""Evaluate a trained strict five-task Multi-EE baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .model import TaskExecutorPointNet
from .train import load_config, make_loader, resolve, run_epoch, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the five-task TaskExecutorPointNet baseline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--manifest", default="", help="Optional manifest override.")
    parser.add_argument("--output-json", default="", help="Optional output metrics path.")
    parser.add_argument("--device", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    dataset_root = Path(args.dataset_root or config["dataset_root"]).resolve()
    if args.manifest:
        config["test_manifest"] = args.manifest
    loader = make_loader(config, dataset_root, "test_manifest", train=False)
    if loader is None:
        raise ValueError("Test manifest contains no rows.")
    device = torch.device(args.device or config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = TaskExecutorPointNet(
        input_channels=int(config.get("input_channels", 3)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        task_dim=int(config.get("task_dim", 64)),
        executor_dim=int(config.get("executor_dim", 64)),
        executor_mode=str(config.get("executor_mode", "learnable")),
        executor_token_permutation=config.get("executor_token_permutation"),
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    metrics = run_epoch(model, loader, device=device, config=config, optimizer=None)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output_json:
        save_json(resolve(dataset_root, args.output_json), metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
