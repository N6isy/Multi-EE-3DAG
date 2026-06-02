#!/usr/bin/env python3
"""Synthetic smoke test for the independent five-task training package."""

from __future__ import annotations

import json
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .constants import EXECUTORS, require_five_task
from .dataset import MultiEEFiveTaskDataset
from .losses import compute_loss
from .model import TaskExecutorPointNet
from .prepare_training_dataset import prepare


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="multi_ee_training_smoke_") as temp:
        root = Path(temp)
        points = np.random.default_rng(2026).normal(size=(64, 3)).astype(np.float32)
        masks = np.zeros((64, len(EXECUTORS)), dtype=np.uint8)
        masks[:8, 0] = 1
        masks[8:16, 1] = 1
        masks[16:24, 2] = 1
        masks[24:32, 3] = 1
        points_path = root / "points.npy"
        np.save(points_path, points)
        rows = []
        for index, executor in enumerate(EXECUTORS):
            channel_mask = root / f"{executor}.npy"
            np.save(channel_mask, masks)
            rows.append(
                {
                    "sample_id": "fixture_lift",
                    "object_id": "fixture",
                    "source_dataset": "synthetic",
                    "object_category": "fixture",
                    "task": "lift",
                    "point_cloud_path": "points.npy",
                    "multi_channel_mask_path": channel_mask.name,
                    "executor": executor,
                    "quality_flag": "verified",
                    "point_review_status": "verified",
                    "reviewer": "smoke",
                }
            )
        reviewed = root / "reviewed.jsonl"
        write_jsonl(reviewed, rows)
        summary = prepare(
            Namespace(
                dataset_root=str(root),
                reviewed_samples=reviewed.name,
                output_root="processed/training/v0_2_5tasks",
                dataset_version="smoke",
                split_seed="smoke",
                train_ratio=0.8,
                val_ratio=0.1,
                min_reviewed_channels=4,
                allowed_quality_flags="checked,verified",
                allowed_review_statuses="checked,verified",
                overwrite=True,
            )
        )
        assert summary["training_rows"] == 1
        manifest = root / "processed/training/v0_2_5tasks/manifests/all.jsonl"
        dataset = MultiEEFiveTaskDataset(root, manifest, sample_size=32, train=True)
        batch = next(iter(DataLoader(dataset, batch_size=1)))
        model = TaskExecutorPointNet()
        outputs = model(batch["points"], batch["task_id"])
        losses = compute_loss(
            outputs,
            batch,
            lambda_dice=1.0,
            lambda_feasibility=0.5,
            lambda_relation=0.1,
        )
        losses["total"].backward()
        try:
            require_five_task("open_pull")
        except ValueError:
            pass
        else:
            raise AssertionError("Legacy task open_pull must be rejected by training.")
        print(
            json.dumps(
                {
                    "status": "ok",
                    "training_rows": summary["training_rows"],
                    "points": list(batch["points"].shape),
                    "masks": list(batch["masks"].shape),
                    "mask_logits": list(outputs["mask_logits"].shape),
                    "feasibility_logits": list(outputs["feasibility_logits"].shape),
                    "legacy_task_rejected": True,
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

