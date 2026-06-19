"""Model factory for training/evaluation entry points."""

from __future__ import annotations

from typing import Any

from torch import nn

from .executor_conditioning import normalize_condition_mode
from .model import TaskExecutorAffordanceModel


def infer_backbone_name(config: dict[str, Any]) -> str:
    explicit = str(config.get("backbone_name") or "").strip()
    if explicit:
        return explicit
    experiment = str(config.get("experiment_name") or config.get("model_name") or "").lower()
    if experiment.startswith("pointnext") or "pointnext" in experiment:
        return "pointnext"
    return "pointnet_mlp"


def infer_executor_condition_mode(config: dict[str, Any]) -> str:
    raw = config.get("executor_condition_mode", config.get("executor_mode", "learnable_id"))
    return normalize_condition_mode(str(raw))


def build_model(config: dict[str, Any]) -> nn.Module:
    """Build a five-task Multi-EE model from JSON config."""

    backbone_config = dict(config)
    return TaskExecutorAffordanceModel(
        input_channels=int(config.get("input_channels", 3)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        task_dim=int(config.get("task_dim", 64)),
        executor_dim=int(config.get("executor_dim", 64)),
        backbone_name=infer_backbone_name(config),
        executor_condition_mode=infer_executor_condition_mode(config),
        executor_spec_path=config.get("executor_spec_path"),
        executor_id_dropout=float(config.get("executor_id_dropout", 0.0)),
        executor_token_permutation=config.get("executor_token_permutation"),
        pointnext_root=config.get("pointnext_root") or config.get("backbone_external_root"),
        backbone_config=backbone_config,
    )


def remap_legacy_pointnet_state(state: dict[str, Any]) -> dict[str, Any]:
    """Maps pre-factory TaskExecutorPointNet checkpoint keys to current names."""

    mapped: dict[str, Any] = {}
    point_layers = {"0", "2", "4"}
    for key, value in state.items():
        new_key = key
        if key.startswith("point_encoder."):
            pieces = key.split(".", 2)
            if len(pieces) == 3 and pieces[1] in point_layers:
                new_key = f"point_encoder.net.{pieces[1]}.{pieces[2]}"
        elif key.startswith("executor_embedding."):
            new_key = "executor_condition.id_embedding." + key.split(".", 1)[1]
        elif key.startswith("executor_query."):
            new_key = "executor_condition.id_projection." + key.split(".", 1)[1]
        elif key == "executor_token_permutation":
            new_key = "executor_condition.executor_token_permutation"
        mapped[new_key] = value
    return mapped


def load_model_state(model: nn.Module, state: dict[str, Any]) -> None:
    """Loads current or legacy checkpoint state into a model."""

    missing, unexpected = model.load_state_dict(state, strict=False)
    if not unexpected and not any(not item.startswith("executor_condition.") for item in missing):
        return
    remapped = remap_legacy_pointnet_state(state)
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    harmful_missing = [
        item
        for item in missing
        if not item.startswith("executor_condition.executor_token_permutation")
        and not item.startswith("attr_matrix")
        and "film_projection" not in item
        and "condition_projection" not in item
    ]
    benign_unexpected = (
        "executor_condition.id_embedding",
        "executor_condition.id_projection",
        "executor_condition.executor_token_permutation",
    )
    harmful_unexpected = [item for item in unexpected if not item.startswith(benign_unexpected)]
    if harmful_unexpected or harmful_missing:
        raise RuntimeError(f"Checkpoint state mismatch. missing={harmful_missing}, unexpected={harmful_unexpected}")
