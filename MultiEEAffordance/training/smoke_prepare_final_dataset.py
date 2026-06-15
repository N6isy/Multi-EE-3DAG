#!/usr/bin/env python3
"""Smoke test for final five-task row-level JSONL preparation."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from .constants import EXECUTORS, TASKS
from .prepare_final_5task_training_dataset import prepare


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny final-JSONL preparation smoke test.")
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            file.write("\n")


def build_fixture(root: Path) -> Path:
    object_id = "3danet_full_smoke_final_asset"
    points_dir = root / "processed" / "points" / "smoke_final"
    masks_dir = root / "processed" / "masks" / "smoke_final"
    points_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    points = np.zeros((32, 3), dtype=np.float32)
    points[:, 0] = np.linspace(-1, 1, 32)
    points[:, 1] = np.sin(np.linspace(0, 3.14, 32))
    points_path = points_dir / f"{object_id}.npy"
    np.save(points_path, points)

    rows: list[dict] = []
    for task_index, task in enumerate(TASKS):
        full_mask = np.zeros((32, len(EXECUTORS)), dtype=np.uint8)
        for executor_index, _executor in enumerate(EXECUTORS):
            if (task_index + executor_index) % 2 == 0:
                start = executor_index * 3
                full_mask[start : start + 3, executor_index] = 1
        for executor_index, executor in enumerate(EXECUTORS):
            row_mask = np.zeros_like(full_mask)
            row_mask[:, executor_index] = full_mask[:, executor_index]
            mask_path = masks_dir / f"{object_id}_{task}_{executor}.npy"
            np.save(mask_path, row_mask)
            rows.append(
                {
                    "object_id": object_id,
                    "source_dataset": "3d_affordancenet",
                    "object_category": "SmokeObject",
                    "task": task,
                    "executor": executor,
                    "target_executor": executor,
                    "executor_order": list(EXECUTORS),
                    "point_cloud_path": points_path.relative_to(root).as_posix(),
                    "multi_channel_mask_path": mask_path.relative_to(root).as_posix(),
                    "point_count": int(points.shape[0]),
                    "positive_points_after": int(row_mask[:, executor_index].sum()),
                    "is_synthetic_empty_combo": bool(row_mask[:, executor_index].sum() == 0),
                }
            )
    final_jsonl = root / "processed" / "annotation_batches" / "final_5tasks" / "smoke_final.jsonl"
    write_jsonl(final_jsonl, rows)
    return final_jsonl


def main() -> int:
    args = parse_args()
    temp = tempfile.TemporaryDirectory()
    try:
        root = Path(temp.name).resolve()
        final_jsonl = build_fixture(root)
        ns = argparse.Namespace(
            dataset_root=str(root),
            final_samples=final_jsonl.relative_to(root).as_posix(),
            output_root="processed/training/smoke_final_5tasks",
            dataset_version="smoke_final_5tasks",
            split_seed="smoke",
            split_unit="source_asset",
            train_ratio=0.8,
            val_ratio=0.1,
            copy_masks=False,
            allow_errors=False,
            overwrite=True,
        )
        summary = prepare(ns)
        manifest = root / "processed" / "training" / "smoke_final_5tasks" / "manifests" / "all.jsonl"
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        splits = {row["split"] for row in rows}
        assert summary["input_rows"] == 20, summary
        assert summary["training_object_task_rows"] == 5, summary
        assert len(rows) == 5, rows
        assert len(splits) == 1, splits
        assert all(row["asset_uid"] == "3d_affordancenet:3danet_full_smoke_final_asset" for row in rows)
        assert all(row["channel_supervision"] == [1, 1, 1, 1] for row in rows)
        print(json.dumps({"status": "ok", "training_rows": len(rows), "splits": sorted(splits)}, ensure_ascii=False))
        return 0
    finally:
        if args.keep_temp:
            print(json.dumps({"kept_temp": temp.name}, ensure_ascii=False))
        else:
            temp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
