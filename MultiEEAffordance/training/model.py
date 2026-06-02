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
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.hidden_dim = int(hidden_dim)
        self.point_encoder = nn.Sequential(
            nn.Linear(self.input_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.task_embedding = nn.Embedding(len(TASKS), task_dim)
        self.executor_embedding = nn.Embedding(len(EXECUTORS), executor_dim)
        self.task_query = nn.Linear(task_dim, self.hidden_dim)
        self.executor_query = nn.Linear(executor_dim, self.hidden_dim)
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
        executor_ids = torch.arange(len(EXECUTORS), device=points.device)
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

