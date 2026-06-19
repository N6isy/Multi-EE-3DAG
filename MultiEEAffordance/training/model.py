"""Task-conditioned multi-executor affordance models."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .backbones import PointMLPBackbone, PointNeXtBackbone
from .constants import EXECUTORS, TASKS
from .executor_conditioning import ExecutorConditionEncoder


def build_backbone(
    backbone_name: str,
    *,
    input_channels: int,
    hidden_dim: int,
    config: dict[str, Any] | None = None,
) -> nn.Module:
    """Builds a point backbone that returns `[B, N, C]` features."""

    config = config or {}
    name = str(backbone_name or "pointnet_mlp").lower()
    if name in {"pointnet", "pointnet_mlp", "mlp"}:
        return PointMLPBackbone(input_channels=input_channels, hidden_dim=hidden_dim)
    if name in {"pointnext", "pointnext_s"}:
        return PointNeXtBackbone(
            input_channels=input_channels,
            pointnext_root=config.get("pointnext_root") or config.get("backbone_external_root"),
            width=int(config.get("pointnext_width", 32)),
            blocks=list(config.get("pointnext_blocks", [1, 1, 1, 1, 1])),
            strides=list(config.get("pointnext_strides", [1, 2, 2, 2, 2])),
            nsample=config.get("pointnext_nsample", 32),
            radius=config.get("pointnext_radius", 0.1),
            decoder_layers=int(config.get("pointnext_decoder_layers", 2)),
            decoder_stages=int(config.get("pointnext_decoder_stages", 4)),
            sa_layers=int(config.get("pointnext_sa_layers", 1)),
            sa_use_res=bool(config.get("pointnext_sa_use_res", False)),
        )
    raise ValueError(f"Unknown backbone_name {backbone_name!r}.")


class TaskExecutorAffordanceModel(nn.Module):
    """Shared four-executor task-conditioned mask + feasibility predictor."""

    def __init__(
        self,
        *,
        input_channels: int = 3,
        hidden_dim: int = 128,
        task_dim: int = 64,
        executor_dim: int = 64,
        backbone_name: str = "pointnet_mlp",
        executor_condition_mode: str = "learnable_id",
        executor_spec_path: str | None = None,
        executor_id_dropout: float = 0.0,
        executor_token_permutation: list[int] | None = None,
        pointnext_root: str | None = None,
        backbone_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.hidden_dim = int(hidden_dim)
        merged_backbone_config = dict(backbone_config or {})
        if pointnext_root:
            merged_backbone_config["pointnext_root"] = pointnext_root
        self.backbone_name = str(backbone_name or "pointnet_mlp")
        self.point_encoder = build_backbone(
            self.backbone_name,
            input_channels=self.input_channels,
            hidden_dim=self.hidden_dim,
            config=merged_backbone_config,
        )
        backbone_out = int(getattr(self.point_encoder, "out_channels", self.hidden_dim))
        self.backbone_projection = (
            nn.Identity() if backbone_out == self.hidden_dim else nn.Linear(backbone_out, self.hidden_dim)
        )
        self.task_embedding = nn.Embedding(len(TASKS), int(task_dim))
        self.task_query = nn.Linear(int(task_dim), self.hidden_dim)
        self.executor_condition = ExecutorConditionEncoder(
            mode=executor_condition_mode,
            executor_dim=int(executor_dim),
            hidden_dim=self.hidden_dim,
            executor_spec_path=executor_spec_path,
            executor_id_dropout=float(executor_id_dropout),
            executor_token_permutation=executor_token_permutation,
        )
        self.point_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.global_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.mask_bias = nn.Parameter(torch.zeros(len(EXECUTORS)))
        self.feasibility_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 1),
        )
        self.cross_attention: nn.MultiheadAttention | None = None
        if self.executor_condition.use_cross_attention:
            heads = max(1, min(4, self.hidden_dim // 32))
            self.cross_attention = nn.MultiheadAttention(self.hidden_dim, heads, batch_first=True)

    def _conditioned_mask_logits(
        self,
        mask_features: torch.Tensor,
        queries: torch.Tensor,
        film: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        if film is None:
            return torch.einsum("bnh,beh->bne", mask_features, queries) / (self.hidden_dim**0.5)
        gamma, beta = film
        conditioned = mask_features[:, None, :, :] * (1.0 + gamma[None, :, None, :]) + beta[None, :, None, :]
        return torch.einsum("benh,beh->bne", conditioned, queries) / (self.hidden_dim**0.5)

    def forward(self, points: torch.Tensor, task_id: torch.Tensor) -> dict[str, torch.Tensor]:
        point_features = self.backbone_projection(self.point_encoder(points))
        global_features = point_features.max(dim=1).values
        task_features = self.task_query(self.task_embedding(task_id))
        executor_features, film = self.executor_condition(points.device)
        queries = torch.tanh(task_features[:, None, :] + executor_features[None, :, :])
        mask_features = torch.tanh(
            self.point_projection(point_features) + self.global_projection(global_features)[:, None, :]
        )
        if self.cross_attention is not None:
            attended, _ = self.cross_attention(queries, mask_features, mask_features, need_weights=False)
            queries = torch.tanh(queries + attended)
        mask_logits = self._conditioned_mask_logits(mask_features, queries, film) + self.mask_bias
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


class TaskExecutorPointNet(TaskExecutorAffordanceModel):
    """Backward-compatible name for the original point-wise MLP baseline."""

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
        super().__init__(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            task_dim=task_dim,
            executor_dim=executor_dim,
            backbone_name="pointnet_mlp",
            executor_condition_mode=executor_mode,
            executor_token_permutation=executor_token_permutation,
        )
