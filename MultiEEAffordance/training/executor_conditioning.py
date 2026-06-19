"""Executor identity and attribute conditioning modules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .constants import EXECUTORS


LEGACY_MODE_MAP = {
    "learnable": "learnable_id",
    "one_hot": "one_hot_id",
    "no_token": "no_token",
    "shared": "shared_token",
    "random_frozen": "random_frozen_id",
    "swap": "learnable_id",
}


VALID_MODES = {
    "no_token",
    "shared_token",
    "one_hot_id",
    "learnable_id",
    "random_frozen_id",
    "attr_only",
    "id_attr",
    "id_attr_film",
    "id_attr_crossattn",
}


def normalize_condition_mode(mode: str | None) -> str:
    raw = str(mode or "learnable_id").strip()
    normalized = LEGACY_MODE_MAP.get(raw, raw)
    if normalized not in VALID_MODES:
        raise ValueError(f"Unknown executor condition mode {raw!r}; expected one of {sorted(VALID_MODES)}.")
    return normalized


def default_executor_spec_path() -> Path:
    return Path(__file__).resolve().parent / "configs" / "executor_specs_5tasks.json"


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def load_executor_attribute_matrix(spec_path: str | Path | None = None) -> tuple[torch.Tensor, list[str]]:
    path = Path(spec_path).expanduser() if spec_path else default_executor_spec_path()
    if not path.exists():
        raise FileNotFoundError(f"Executor spec file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    specs = payload.get("executors", payload)
    if not isinstance(specs, dict):
        raise ValueError(f"Executor specs must be a JSON object keyed by executor name: {path}")
    missing = [executor for executor in EXECUTORS if executor not in specs]
    if missing:
        raise ValueError(f"Executor spec file misses executors {missing}: {path}")

    categorical_keys = ["contact_mode", "contact_geometry", "force_pattern", "compliance"]
    categories: dict[str, list[str]] = {}
    for key in categorical_keys:
        categories[key] = sorted({str(specs[executor].get(key, "unknown")) for executor in EXECUTORS})
    numeric_keys = [
        "contact_area",
        "requires_enclosure",
        "requires_flatness",
        "requires_inner_boundary",
        "requires_bilateral_contact",
        "requires_fingertip_precision",
        "can_press",
        "can_pull",
        "can_lift",
    ]
    feature_names: list[str] = list(numeric_keys)
    for key in categorical_keys:
        feature_names.extend(f"{key}={category}" for category in categories[key])
    rows: list[list[float]] = []
    for executor in EXECUTORS:
        spec = specs[executor]
        row: list[float] = []
        for key in numeric_keys:
            row.append(_as_float(spec.get(key, 0.0)))
        for key in categorical_keys:
            value = str(spec.get(key, "unknown"))
            for category in categories[key]:
                row.append(1.0 if value == category else 0.0)
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32), feature_names


class ExecutorConditionEncoder(nn.Module):
    """Produces executor-wise condition vectors and optional FiLM parameters."""

    def __init__(
        self,
        *,
        mode: str = "learnable_id",
        executor_dim: int = 64,
        hidden_dim: int = 128,
        executor_spec_path: str | Path | None = None,
        executor_id_dropout: float = 0.0,
        executor_token_permutation: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.mode = normalize_condition_mode(mode)
        self.executor_count = len(EXECUTORS)
        self.hidden_dim = int(hidden_dim)
        self.executor_id_dropout = float(executor_id_dropout)
        if self.executor_id_dropout < 0.0 or self.executor_id_dropout >= 1.0:
            raise ValueError("executor_id_dropout must be in [0, 1).")

        self.id_embedding: nn.Embedding | None
        self.id_projection: nn.Module | None
        if self.mode in {"learnable_id", "random_frozen_id", "id_attr", "id_attr_film", "id_attr_crossattn"}:
            self.id_embedding = nn.Embedding(self.executor_count, int(executor_dim))
            if self.mode == "random_frozen_id":
                self.id_embedding.weight.requires_grad_(False)
            self.id_projection = nn.Linear(int(executor_dim), self.hidden_dim)
        elif self.mode == "shared_token":
            self.id_embedding = nn.Embedding(1, int(executor_dim))
            self.id_projection = nn.Linear(int(executor_dim), self.hidden_dim)
        elif self.mode == "one_hot_id":
            self.id_embedding = None
            self.id_projection = nn.Linear(self.executor_count, self.hidden_dim)
        else:
            self.id_embedding = None
            self.id_projection = None

        self.attr_projection: nn.Module | None = None
        self.attr_feature_names: list[str] = []
        if self.mode in {"attr_only", "id_attr", "id_attr_film", "id_attr_crossattn"}:
            attr_matrix, feature_names = load_executor_attribute_matrix(executor_spec_path)
            self.register_buffer("attr_matrix", attr_matrix)
            self.attr_feature_names = feature_names
            self.attr_projection = nn.Sequential(
                nn.Linear(attr_matrix.shape[1], int(executor_dim)),
                nn.ReLU(inplace=True),
                nn.Linear(int(executor_dim), self.hidden_dim),
            )
        else:
            self.register_buffer("attr_matrix", torch.empty(self.executor_count, 0))

        if self.mode in {"id_attr", "id_attr_film", "id_attr_crossattn"}:
            self.condition_projection = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        else:
            self.condition_projection = None

        self.film_projection: nn.Module | None = None
        if self.mode in {"id_attr_film", "id_attr_crossattn"}:
            self.film_projection = nn.Linear(self.hidden_dim, self.hidden_dim * 2)

        if executor_token_permutation is None:
            executor_token_permutation = list(range(self.executor_count))
        if sorted(int(item) for item in executor_token_permutation) != list(range(self.executor_count)):
            raise ValueError("executor_token_permutation must be a permutation of executor indices 0..3.")
        self.register_buffer(
            "executor_token_permutation",
            torch.tensor([int(item) for item in executor_token_permutation], dtype=torch.long),
        )

    @property
    def use_cross_attention(self) -> bool:
        return self.mode == "id_attr_crossattn"

    def _id_features(self, device: torch.device) -> torch.Tensor:
        if self.mode == "no_token":
            return torch.zeros(self.executor_count, self.hidden_dim, device=device)
        if self.mode == "one_hot_id":
            eye = torch.eye(self.executor_count, device=device)
            assert self.id_projection is not None
            return self.id_projection(eye)
        if self.mode == "shared_token":
            ids = torch.zeros(self.executor_count, dtype=torch.long, device=device)
        else:
            ids = self.executor_token_permutation.to(device)
        assert self.id_embedding is not None and self.id_projection is not None
        features = self.id_projection(self.id_embedding(ids))
        if self.training and self.executor_id_dropout > 0.0 and self.mode in {"id_attr", "id_attr_film", "id_attr_crossattn"}:
            keep = torch.rand(features.shape[0], 1, device=device) >= self.executor_id_dropout
            features = features * keep.to(features.dtype)
        return features

    def _attr_features(self, device: torch.device) -> torch.Tensor:
        if self.attr_projection is None:
            return torch.zeros(self.executor_count, self.hidden_dim, device=device)
        return self.attr_projection(self.attr_matrix.to(device))

    def forward(self, device: torch.device) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        if self.mode == "attr_only":
            condition = self._attr_features(device)
        elif self.mode in {"id_attr", "id_attr_film", "id_attr_crossattn"}:
            id_features = self._id_features(device)
            attr_features = self._attr_features(device)
            assert self.condition_projection is not None
            condition = self.condition_projection(torch.cat([id_features, attr_features], dim=-1))
        else:
            condition = self._id_features(device)
        film: tuple[torch.Tensor, torch.Tensor] | None = None
        if self.film_projection is not None:
            gamma, beta = self.film_projection(condition).chunk(2, dim=-1)
            film = (gamma, beta)
        return condition, film
