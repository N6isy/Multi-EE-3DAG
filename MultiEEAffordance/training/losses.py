"""Losses for multi-executor affordance segmentation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mean(values: torch.Tensor, channel_supervision: torch.Tensor) -> torch.Tensor:
    while channel_supervision.ndim < values.ndim:
        channel_supervision = channel_supervision.unsqueeze(1)
    weighted = values * channel_supervision
    denominator = channel_supervision.expand_as(values).sum().clamp_min(1.0)
    return weighted.sum() / denominator


def channel_mean(values: torch.Tensor, channel_supervision: torch.Tensor) -> torch.Tensor:
    weighted = values * channel_supervision
    denominator = channel_supervision.sum().clamp_min(1.0)
    return weighted.sum() / denominator


def focal_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    channel_supervision: torch.Tensor,
    *,
    gamma: float = 2.0,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probabilities = torch.sigmoid(logits)
    pt = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    return masked_mean(((1.0 - pt) ** gamma) * bce, channel_supervision)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, channel_supervision: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=1)
    denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
    values = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    return masked_mean(values, channel_supervision)


def feasibility_loss(
    logits: torch.Tensor, targets: torch.Tensor, channel_supervision: torch.Tensor
) -> torch.Tensor:
    values = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return masked_mean(values, channel_supervision)


def relation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    channel_supervision: torch.Tensor,
    feasibility: torch.Tensor,
    *,
    min_positive_points: float = 4.0,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    executor_count = probabilities.shape[-1]
    losses: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    target_sum_all = targets.sum(dim=1)
    for left in range(executor_count):
        for right in range(left + 1, executor_count):
            prediction_intersection = (probabilities[:, :, left] * probabilities[:, :, right]).sum(dim=1)
            prediction_union = (
                probabilities[:, :, left] + probabilities[:, :, right]
                - probabilities[:, :, left] * probabilities[:, :, right]
            ).sum(dim=1)
            target_intersection = (targets[:, :, left] * targets[:, :, right]).sum(dim=1)
            target_union = (
                targets[:, :, left] + targets[:, :, right] - targets[:, :, left] * targets[:, :, right]
            ).sum(dim=1)
            prediction_iou = prediction_intersection / prediction_union.clamp_min(1e-6)
            target_iou = target_intersection / target_union.clamp_min(1.0)
            losses.append((prediction_iou - target_iou).abs())
            weights.append(
                channel_supervision[:, left]
                * channel_supervision[:, right]
                * (feasibility[:, left] >= 0.5).to(targets.dtype)
                * (feasibility[:, right] >= 0.5).to(targets.dtype)
                * (target_sum_all[:, left] >= min_positive_points).to(targets.dtype)
                * (target_sum_all[:, right] >= min_positive_points).to(targets.dtype)
            )
    values = torch.stack(losses, dim=1)
    pair_supervision = torch.stack(weights, dim=1)
    denominator = pair_supervision.sum().clamp_min(1.0)
    return (values * pair_supervision).sum() / denominator


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    lambda_dice: float,
    lambda_feasibility: float,
    lambda_relation: float,
    lambda_empty_area: float = 0.25,
    min_relation_points: float = 4.0,
) -> dict[str, torch.Tensor]:
    supervision = batch["channel_supervision"]
    feasibility_targets = batch["feasibility"]
    feasible_supervision = supervision * (feasibility_targets >= 0.5).to(supervision.dtype)
    empty_supervision = supervision * (feasibility_targets < 0.5).to(supervision.dtype)

    mask_focal = focal_bce(outputs["mask_logits"], batch["masks"], feasible_supervision)
    mask_dice = dice_loss(outputs["mask_logits"], batch["masks"], feasible_supervision)
    probabilities = torch.sigmoid(outputs["mask_logits"])
    mask_empty_area = channel_mean(probabilities.mean(dim=1), empty_supervision)
    feasibility = feasibility_loss(outputs["feasibility_logits"], feasibility_targets, supervision)
    if lambda_relation > 0:
        relation = relation_loss(
            outputs["mask_logits"],
            batch["masks"],
            supervision,
            feasibility_targets,
            min_positive_points=min_relation_points,
        )
    else:
        relation = outputs["mask_logits"].sum() * 0.0
    total = (
        mask_focal
        + lambda_dice * mask_dice
        + lambda_empty_area * mask_empty_area
        + lambda_feasibility * feasibility
        + lambda_relation * relation
    )
    return {
        "total": total,
        "mask_focal": mask_focal,
        "mask_dice": mask_dice,
        "mask_empty_area": mask_empty_area,
        "feasibility": feasibility,
        "relation": relation,
    }
