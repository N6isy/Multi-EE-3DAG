"""Streaming metrics for five-task multi-executor evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .constants import EXECUTORS


@dataclass
class MetricAccumulator:
    intersection: torch.Tensor = field(default_factory=lambda: torch.zeros(len(EXECUTORS), dtype=torch.float64))
    union: torch.Tensor = field(default_factory=lambda: torch.zeros(len(EXECUTORS), dtype=torch.float64))
    prediction_sum: torch.Tensor = field(default_factory=lambda: torch.zeros(len(EXECUTORS), dtype=torch.float64))
    target_sum: torch.Tensor = field(default_factory=lambda: torch.zeros(len(EXECUTORS), dtype=torch.float64))
    feasibility_correct: torch.Tensor = field(default_factory=lambda: torch.zeros(len(EXECUTORS), dtype=torch.float64))
    supervised_channels: torch.Tensor = field(default_factory=lambda: torch.zeros(len(EXECUTORS), dtype=torch.float64))

    def update(
        self,
        mask_logits: torch.Tensor,
        masks: torch.Tensor,
        feasibility_logits: torch.Tensor,
        feasibility: torch.Tensor,
        channel_supervision: torch.Tensor,
    ) -> None:
        predictions = torch.sigmoid(mask_logits) >= 0.5
        targets = masks >= 0.5
        supervised = channel_supervision >= 0.5
        intersection = (predictions & targets).sum(dim=1).to(torch.float64)
        union = (predictions | targets).sum(dim=1).to(torch.float64)
        prediction_sum = predictions.sum(dim=1).to(torch.float64)
        target_sum = targets.sum(dim=1).to(torch.float64)
        self.intersection += (intersection * supervised).sum(dim=0).cpu()
        self.union += (union * supervised).sum(dim=0).cpu()
        self.prediction_sum += (prediction_sum * supervised).sum(dim=0).cpu()
        self.target_sum += (target_sum * supervised).sum(dim=0).cpu()
        feasibility_prediction = torch.sigmoid(feasibility_logits) >= 0.5
        feasibility_target = feasibility >= 0.5
        self.feasibility_correct += ((feasibility_prediction == feasibility_target) * supervised).sum(dim=0).cpu()
        self.supervised_channels += supervised.sum(dim=0).cpu()

    def compute(self) -> dict[str, object]:
        iou = self.intersection / self.union.clamp_min(1.0)
        dice = 2.0 * self.intersection / (self.prediction_sum + self.target_sum).clamp_min(1.0)
        feasibility_accuracy = self.feasibility_correct / self.supervised_channels.clamp_min(1.0)
        return {
            "macro_iou": float(iou.mean()),
            "macro_dice": float(dice.mean()),
            "macro_feasibility_accuracy": float(feasibility_accuracy.mean()),
            "per_executor": {
                executor: {
                    "iou": float(iou[index]),
                    "dice": float(dice[index]),
                    "feasibility_accuracy": float(feasibility_accuracy[index]),
                    "supervised_samples": int(self.supervised_channels[index]),
                }
                for index, executor in enumerate(EXECUTORS)
            },
        }

