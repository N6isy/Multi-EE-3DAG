#!/usr/bin/env python3
"""Pure NumPy smoke test for five-task training dataset preparation."""

from __future__ import annotations

import json
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np

from .constants import EXECUTORS, require_five_task
from .prepare_training_dataset import prepare


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="multi_ee_prepare_smoke_") as temp:
        root = Path(temp)
        points = np.random.default_rng(2026).normal(size=(32, 3)).astype(np.float32)
        mask = np.zeros((32, len(EXECUTORS)), dtype=np.uint8)
        rows = []
        np.save(root / "points.npy", points)
        for index, executor in enumerate(EXECUTORS):
            mask[index * 4 : (index + 1) * 4, index] = 1
            path = root / f"{executor}.npy"
            np.save(path, mask)
            rows.append(
                {
                    "sample_id": "fixture_lift",
                    "object_id": "fixture",
                    "source_dataset": "synthetic",
                    "object_category": "fixture",
                    "task": "lift",
                    "point_cloud_path": "points.npy",
                    "multi_channel_mask_path": path.name,
                    "executor": executor,
                    "quality_flag": "verified",
                    "point_review_status": "verified",
                    "reviewer": "smoke",
                }
            )
        reviewed = root / "reviewed.jsonl"
        reviewed.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
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
        canonical = np.load(root / "processed/training/v0_2_5tasks/masks/fixture_lift.npy", allow_pickle=False)
        try:
            require_five_task("open_pull")
        except ValueError:
            legacy_rejected = True
        else:
            legacy_rejected = False
        assert summary["training_rows"] == 1
        assert canonical.shape == (32, len(EXECUTORS))
        assert canonical.sum(axis=0).tolist() == [4, 4, 4, 4]
        assert legacy_rejected
        print(
            json.dumps(
                {
                    "status": "ok",
                    "training_rows": summary["training_rows"],
                    "canonical_mask_shape": list(canonical.shape),
                    "positive_points": canonical.sum(axis=0).astype(int).tolist(),
                    "legacy_task_rejected": legacy_rejected,
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
