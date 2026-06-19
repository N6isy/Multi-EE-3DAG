#!/usr/bin/env python3
"""Train the initial strict five-task Multi-EE baseline."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import MultiEEFiveTaskDataset
from .losses import compute_loss
from .metrics import MetricAccumulator
from .model_factory import build_model
from .constants import EXECUTOR_TO_INDEX


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the five-task TaskExecutorPointNet baseline.")
    parser.add_argument("--config", required=True, help="JSON experiment config.")
    parser.add_argument("--dataset-root", default="", help="Override dataset_root from config.")
    parser.add_argument("--output-dir", default="", help="Override output_dir from config.")
    parser.add_argument("--device", default="", help="Override device, for example cuda:0 or cpu.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging for this run.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging for this run.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError("Training config must be a JSON object.")
    return config


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flatten_for_logging(prefix: str, values: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in values.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, bool):
            flat[name] = float(value)
        elif isinstance(value, (int, float)):
            flat[name] = float(value)
        elif isinstance(value, dict):
            flat.update(flatten_for_logging(name, value))
    return flat


def wandb_enabled(config: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.no_wandb:
        return False
    if args.wandb:
        return True
    wandb_config = config.get("wandb", {})
    if isinstance(wandb_config, dict):
        return bool(wandb_config.get("enabled", False))
    return bool(wandb_config)


def init_wandb(
    config: dict[str, Any],
    args: argparse.Namespace,
    *,
    dataset_root: Path,
    output_dir: Path,
):
    if not wandb_enabled(config, args):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "Weights & Biases logging is enabled, but wandb is not installed. "
            "Install it with: python -m pip install wandb"
        ) from exc

    wandb_config = config.get("wandb", {})
    if not isinstance(wandb_config, dict):
        wandb_config = {}
    run_config = dict(config)
    run_config["dataset_root_resolved"] = str(dataset_root)
    run_config["output_dir_resolved"] = str(output_dir)
    init_kwargs: dict[str, Any] = {
        "project": wandb_config.get("project", "multiee-affordance"),
        "name": wandb_config.get("name") or config.get("experiment_name"),
        "config": run_config,
        "dir": str(output_dir),
        "mode": os.environ.get("WANDB_MODE") or wandb_config.get("mode", "online"),
        "resume": wandb_config.get("resume", "allow"),
    }
    for optional_key in ("entity", "group", "job_type", "tags", "notes", "id"):
        if wandb_config.get(optional_key):
            init_kwargs[optional_key] = wandb_config[optional_key]
    run = wandb.init(**init_kwargs)
    return run


def make_loader(
    config: dict[str, Any],
    dataset_root: Path,
    manifest_key: str,
    *,
    train: bool,
) -> DataLoader | None:
    value = str(config.get(manifest_key) or "").strip()
    if not value:
        return None
    dataset = MultiEEFiveTaskDataset(
        dataset_root,
        value,
        sample_size=int(config.get("sample_size", 2048)),
        train=train,
        input_channels=int(config.get("input_channels", 3)),
    )
    if len(dataset) == 0:
        return None
    return DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 8)),
        shuffle=train,
        num_workers=int(config.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def apply_enabled_executors(batch: dict[str, torch.Tensor], config: dict[str, Any]) -> None:
    raw = config.get("enabled_executors")
    if not raw:
        return
    enabled = [str(item).strip() for item in raw if str(item).strip()] if isinstance(raw, list) else [str(raw).strip()]
    mask = torch.zeros_like(batch["channel_supervision"])
    for executor in enabled:
        if executor not in EXECUTOR_TO_INDEX:
            raise ValueError(f"Unknown enabled executor {executor!r}.")
        mask[:, EXECUTOR_TO_INDEX[executor]] = 1.0
    batch["channel_supervision"] = batch["channel_supervision"] * mask


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, Any]:
    train = optimizer is not None
    model.train(train)
    metric = MetricAccumulator()
    loss_sums: dict[str, float] = {}
    steps = 0
    for batch in loader:
        points = batch["points"].to(device)
        task_id = batch["task_id"].to(device)
        tensor_batch = {
            "masks": batch["masks"].to(device),
            "channel_supervision": batch["channel_supervision"].to(device),
            "feasibility": batch["feasibility"].to(device),
        }
        apply_enabled_executors(tensor_batch, config)
        with torch.set_grad_enabled(train):
            outputs = model(points, task_id)
            losses = compute_loss(
                outputs,
                tensor_batch,
                lambda_dice=float(config.get("lambda_dice", 1.0)),
                lambda_feasibility=float(config.get("lambda_feasibility", 0.5)),
                lambda_relation=float(config.get("lambda_relation", 0.0)),
                lambda_empty_area=float(config.get("lambda_empty_area", 0.25)),
                min_relation_points=float(config.get("min_relation_points", 4.0)),
            )
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("max_grad_norm", 1.0)))
                optimizer.step()
        for name, value in losses.items():
            loss_sums[name] = loss_sums.get(name, 0.0) + float(value.detach().cpu())
        metric.update(
            outputs["mask_logits"].detach(),
            tensor_batch["masks"],
            outputs["feasibility_logits"].detach(),
            tensor_batch["feasibility"],
            tensor_batch["channel_supervision"],
            task_id,
        )
        steps += 1
    result = {f"loss_{name}": value / max(steps, 1) for name, value in loss_sums.items()}
    result.update(metric.compute())
    return result


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    dataset_root = Path(args.dataset_root or config["dataset_root"]).resolve()
    output_dir = resolve(dataset_root, args.output_dir or config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_everything(int(config.get("seed", 2026)))
    wandb_run = init_wandb(config, args, dataset_root=dataset_root, output_dir=output_dir)

    train_loader = make_loader(config, dataset_root, "train_manifest", train=True)
    val_loader = make_loader(config, dataset_root, "val_manifest", train=False)
    if train_loader is None:
        raise ValueError("Training manifest contains no rows.")

    model = build_model(config).to(device)
    wandb_config = config.get("wandb", {})
    if wandb_run is not None and isinstance(wandb_config, dict) and wandb_config.get("watch_model", False):
        wandb_run.watch(model, log=str(wandb_config.get("watch_log", "gradients")), log_freq=int(wandb_config.get("watch_log_freq", 100)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    epochs = int(config.get("epochs", 50))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    history: list[dict[str, Any]] = []
    best_score = -1.0

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, device=device, config=config, optimizer=optimizer)
        val_metrics = (
            run_epoch(model, val_loader, device=device, config=config, optimizer=None) if val_loader is not None else {}
        )
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if wandb_run is not None:
            log_payload: dict[str, float] = {
                "epoch": float(epoch),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            log_payload.update(flatten_for_logging("train", train_metrics))
            log_payload.update(flatten_for_logging("val", val_metrics))
            wandb_run.log(log_payload, step=epoch)
        checkpoint = {
            "epoch": epoch,
            "config": config,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, output_dir / "latest.pt")
        score = float(val_metrics.get("macro_iou", train_metrics.get("macro_iou", 0.0)))
        if score > best_score:
            best_score = score
            torch.save(checkpoint, output_dir / "best.pt")
        save_json(output_dir / "history.json", history)
        scheduler.step()

    save_json(output_dir / "resolved_config.json", config)
    if wandb_run is not None:
        wandb_run.summary["best_macro_iou"] = best_score
        if isinstance(wandb_config, dict) and wandb_config.get("log_checkpoints", False):
            import wandb

            artifact = wandb.Artifact(f"{wandb_run.name}-checkpoint", type="model")
            best_path = output_dir / "best.pt"
            history_path = output_dir / "history.json"
            config_path = output_dir / "resolved_config.json"
            if best_path.exists():
                artifact.add_file(str(best_path), name="best.pt")
            if history_path.exists():
                artifact.add_file(str(history_path), name="history.json")
            if config_path.exists():
                artifact.add_file(str(config_path), name="resolved_config.json")
            wandb_run.log_artifact(artifact)
        wandb_run.finish()
    print(f"Training complete. best_macro_iou={best_score:.6f} output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
