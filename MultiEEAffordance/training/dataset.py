"""PyTorch dataset for strict five-task Multi-EE training manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import EXECUTORS, TASK_TO_INDEX, require_five_task


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}.")
            require_five_task(str(row.get("task") or ""))
            rows.append(row)
    return rows


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def normalize_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).copy()
    xyz = points[:, :3]
    xyz -= xyz.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(xyz, axis=1).max()
    if scale > 1e-8:
        xyz /= scale
    points[:, :3] = xyz
    return points


def choose_indices(point_count: int, sample_size: int, *, train: bool) -> np.ndarray:
    if point_count <= 0:
        raise ValueError("Point cloud is empty.")
    if sample_size <= 0:
        return np.arange(point_count, dtype=np.int64)
    if train:
        return np.random.choice(point_count, size=sample_size, replace=point_count < sample_size).astype(np.int64)
    if point_count >= sample_size:
        return np.linspace(0, point_count - 1, sample_size, dtype=np.int64)
    repeats = int(np.ceil(sample_size / point_count))
    return np.tile(np.arange(point_count, dtype=np.int64), repeats)[:sample_size]


class MultiEEFiveTaskDataset(Dataset):
    """Loads canonical object-task rows prepared from human review outputs."""

    def __init__(
        self,
        dataset_root: str | Path,
        manifest: str | Path,
        *,
        sample_size: int = 2048,
        train: bool = False,
        input_channels: int = 3,
    ) -> None:
        self.root = Path(dataset_root).resolve()
        self.manifest_path = resolve(self.root, manifest)
        self.rows = read_jsonl(self.manifest_path)
        self.sample_size = int(sample_size)
        self.train = bool(train)
        self.input_channels = int(input_channels)
        if self.input_channels not in (3, 6):
            raise ValueError("input_channels must be 3 or 6.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        points_path = resolve(self.root, str(row["point_cloud_path"]))
        masks_path = resolve(self.root, str(row["multi_channel_mask_path"]))
        points = np.load(points_path, allow_pickle=False)
        masks = np.load(masks_path, allow_pickle=False)
        if points.ndim != 2 or points.shape[1] not in (3, 6):
            raise ValueError(f"Bad point shape {points.shape}: {points_path}")
        if masks.ndim != 2 or masks.shape != (points.shape[0], len(EXECUTORS)):
            raise ValueError(f"Bad mask shape {masks.shape}; expected {(points.shape[0], len(EXECUTORS))}: {masks_path}")
        if self.input_channels == 6 and points.shape[1] < 6:
            points = np.concatenate([points[:, :3], np.zeros_like(points[:, :3])], axis=1)
        points = normalize_points(points[:, : self.input_channels])
        indices = choose_indices(points.shape[0], self.sample_size, train=self.train)
        supervision = np.asarray(row.get("channel_supervision", [1] * len(EXECUTORS)), dtype=np.float32)
        feasibility = np.asarray(row.get("feasibility", (masks.sum(axis=0) > 0).astype(int)), dtype=np.float32)
        return {
            "points": torch.from_numpy(points[indices].astype(np.float32)),
            "masks": torch.from_numpy((masks[indices] > 0).astype(np.float32)),
            "task_id": torch.tensor(TASK_TO_INDEX[require_five_task(str(row["task"]))], dtype=torch.long),
            "channel_supervision": torch.from_numpy(supervision),
            "feasibility": torch.from_numpy(feasibility),
            "training_id": str(row.get("training_id") or ""),
            "object_id": str(row.get("object_id") or ""),
            "source_asset_id": str(row.get("source_asset_id") or ""),
            "asset_uid": str(row.get("asset_uid") or ""),
            "split_key": str(row.get("split_key") or ""),
            "task": str(row["task"]),
        }
