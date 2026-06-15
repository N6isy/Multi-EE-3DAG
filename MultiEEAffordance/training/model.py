"""Initial task-conditioned multi-executor point model."""

from __future__ import annotations

import torch
from torch import nn

from .constants import EXECUTORS, TASKS


class TaskExecutorPointNet(nn.Module):
    """Small baseline with explicit task and executor queries.

    This is intentionally lightweight. It is suitable for validating the data
    contract before replacing the point encoder with PointNeXt or a Point
    Transformer backbone.
    """

    def __init__(
        self,
        *,
        input_channels: int = 3,
        hidden_dim: int = 128,
        task_dim: int = 64,
        executor_dim: int = 64,
        executor_mode: str = "learnable",
        executor_token_permutation: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.hidden_dim = int(hidden_dim)
        self.executor_mode = str(executor_mode or "learnable")
        if self.executor_mode not in {"learnable", "no_token", "shared", "one_hot", "random_frozen", "swap"}:
            raise ValueError(
                "executor_mode must be one of learnable/no_token/shared/one_hot/random_frozen/swap; "
                f"got {self.executor_mode!r}."
            )
        self.point_encoder = nn.Sequential(
            nn.Linear(self.input_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.task_embedding = nn.Embedding(len(TASKS), task_dim)
        if self.executor_mode == "one_hot":
            self.executor_embedding = None
            effective_executor_dim = len(EXECUTORS)
        else:
            executor_embedding_count = 1 if self.executor_mode == "shared" else len(EXECUTORS)
            self.executor_embedding = nn.Embedding(executor_embedding_count, executor_dim)
            if self.executor_mode == "random_frozen":
                self.executor_embedding.weight.requires_grad_(False)
            effective_executor_dim = executor_dim
        self.task_query = nn.Linear(task_dim, self.hidden_dim)
        self.executor_query = nn.Linear(effective_executor_dim, self.hidden_dim)
        if self.executor_mode == "swap":
            if executor_token_permutation is None:
                executor_token_permutation = list(reversed(range(len(EXECUTORS))))
            if sorted(int(item) for item in executor_token_permutation) != list(range(len(EXECUTORS))):
                raise ValueError("executor_token_permutation must be a permutation of executor indices 0..3.")
            self.register_buffer(
                "executor_token_permutation",
                torch.tensor([int(item) for item in executor_token_permutation], dtype=torch.long),
            )
        else:
            self.executor_token_permutation = None
        self.point_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.global_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.mask_bias = nn.Parameter(torch.zeros(len(EXECUTORS)))
        self.feasibility_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, points: torch.Tensor, task_id: torch.Tensor) -> dict[str, torch.Tensor]:
        point_features = self.point_encoder(points)
        global_features = point_features.max(dim=1).values
        task_features = self.task_query(self.task_embedding(task_id))
        if self.executor_mode == "no_token":
            executor_features = torch.zeros(len(EXECUTORS), self.hidden_dim, device=points.device)
        elif self.executor_mode == "one_hot":
            executor_one_hot = torch.eye(len(EXECUTORS), device=points.device)
            executor_features = self.executor_query(executor_one_hot)
        elif self.executor_mode == "shared":
            shared_id = torch.zeros(len(EXECUTORS), dtype=torch.long, device=points.device)
            executor_features = self.executor_query(self.executor_embedding(shared_id))
        else:
            executor_ids = torch.arange(len(EXECUTORS), device=points.device)
            if self.executor_mode == "swap":
                executor_ids = self.executor_token_permutation.to(points.device)
            executor_features = self.executor_query(self.executor_embedding(executor_ids))
        queries = torch.tanh(task_features[:, None, :] + executor_features[None, :, :])
        mask_features = torch.tanh(
            self.point_projection(point_features) + self.global_projection(global_features)[:, None, :]
        )
        mask_logits = torch.einsum("bnh,beh->bne", mask_features, queries) / (self.hidden_dim**0.5)
        mask_logits = mask_logits + self.mask_bias
        feasibility_features = torch.cat(
            [
                global_features[:, None, :].expand(-1, len(EXECUTORS), -1),
                queries,
            ],
            dim=-1,
        )
        feasibility_logits = self.feasibility_head(feasibility_features).squeeze(-1)
        return {
            "mask_logits": mask_logits,
            "feasibility_logits": feasibility_logits,
        }
