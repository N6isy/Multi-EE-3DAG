"""Streaming metrics for five-task multi-executor evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .constants import EXECUTORS, TASKS


def _zeros(*shape: int) -> torch.Tensor:
    return torch.zeros(*shape, dtype=torch.float64)


def _safe_auc(scores: list[float], labels: list[int]) -> float | None:
    positives = sum(1 for label in labels if label == 1)
    negatives = sum(1 for label in labels if label == 0)
    if positives == 0 or negatives == 0:
        return None
    order = sorted(range(len(scores)), key=lambda idx: scores[idx])
    rank_sum = 0.0
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and scores[order[end]] == scores[order[index]]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for pos in range(index, end):
            if labels[order[pos]] == 1:
                rank_sum += average_rank
        index = end
    return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


@dataclass
class MetricAccumulator:
    """Accumulates mask, feasibility, empty-mask, and overlap metrics."""

    small_part_max_fraction: float = 0.02
    intersection: torch.Tensor = field(default_factory=lambda: _zeros(len(TASKS), len(EXECUTORS)))
    union: torch.Tensor = field(default_factory=lambda: _zeros(len(TASKS), len(EXECUTORS)))
    prediction_sum: torch.Tensor = field(default_factory=lambda: _zeros(len(TASKS), len(EXECUTORS)))
    target_sum: torch.Tensor = field(default_factory=lambda: _zeros(len(TASKS), len(EXECUTORS)))
    feasible_channels: torch.Tensor = field(default_factory=lambda: _zeros(len(TASKS), len(EXECUTORS)))
    supervised_channels: torch.Tensor = field(default_factory=lambda: _zeros(len(TASKS), len(EXECUTORS)))
    feasibility_tp: torch.Tensor = field(default_factory=lambda: _zeros(len(EXECUTORS)))
    feasibility_fp: torch.Tensor = field(default_factory=lambda: _zeros(len(EXECUTORS)))
    feasibility_fn: torch.Tensor = field(default_factory=lambda: _zeros(len(EXECUTORS)))
    feasibility_tn: torch.Tensor = field(default_factory=lambda: _zeros(len(EXECUTORS)))
    empty_correct: torch.Tensor = field(default_factory=lambda: _zeros(len(EXECUTORS)))
    empty_total: torch.Tensor = field(default_factory=lambda: _zeros(len(EXECUTORS)))
    small_intersection: torch.Tensor = field(default_factory=lambda: _zeros(len(EXECUTORS)))
    small_target: torch.Tensor = field(default_factory=lambda: _zeros(len(EXECUTORS)))
    overlap_error_sum: float = 0.0
    overlap_error_count: int = 0
    feasibility_scores: list[float] = field(default_factory=list)
    feasibility_labels: list[int] = field(default_factory=list)

    def update(
        self,
        mask_logits: torch.Tensor,
        masks: torch.Tensor,
        feasibility_logits: torch.Tensor,
        feasibility: torch.Tensor,
        channel_supervision: torch.Tensor,
        task_id: torch.Tensor,
    ) -> None:
        probabilities = torch.sigmoid(mask_logits)
        predictions = probabilities >= 0.5
        targets = masks >= 0.5
        supervised = channel_supervision >= 0.5
        feasibility_prediction = torch.sigmoid(feasibility_logits) >= 0.5
        feasibility_target = feasibility >= 0.5
        feasible_supervised = supervised & feasibility_target

        intersection = (predictions & targets).sum(dim=1).to(torch.float64)
        union = (predictions | targets).sum(dim=1).to(torch.float64)
        prediction_sum = predictions.sum(dim=1).to(torch.float64)
        target_sum = targets.sum(dim=1).to(torch.float64)

        for batch_index in range(mask_logits.shape[0]):
            task_index = int(task_id[batch_index].detach().cpu())
            feasible_mask = feasible_supervised[batch_index].to(torch.float64).cpu()
            supervised_mask = supervised[batch_index].to(torch.float64).cpu()
            self.intersection[task_index] += intersection[batch_index].detach().cpu() * feasible_mask
            self.union[task_index] += union[batch_index].detach().cpu() * feasible_mask
            self.prediction_sum[task_index] += prediction_sum[batch_index].detach().cpu() * feasible_mask
            self.target_sum[task_index] += target_sum[batch_index].detach().cpu() * feasible_mask
            self.feasible_channels[task_index] += feasible_mask
            self.supervised_channels[task_index] += supervised_mask

            for left in range(len(EXECUTORS)):
                for right in range(left + 1, len(EXECUTORS)):
                    pair_enabled = feasible_supervised[batch_index, left] & feasible_supervised[batch_index, right]
                    if not bool(pair_enabled.detach().cpu()):
                        continue
                    pred_left = predictions[batch_index, :, left]
                    pred_right = predictions[batch_index, :, right]
                    gt_left = targets[batch_index, :, left]
                    gt_right = targets[batch_index, :, right]
                    pred_intersection = (pred_left & pred_right).sum().to(torch.float64)
                    pred_union = (pred_left | pred_right).sum().to(torch.float64)
                    gt_intersection = (gt_left & gt_right).sum().to(torch.float64)
                    gt_union = (gt_left | gt_right).sum().to(torch.float64)
                    pred_iou = pred_intersection / pred_union.clamp_min(1.0)
                    gt_iou = gt_intersection / gt_union.clamp_min(1.0)
                    self.overlap_error_sum += float((pred_iou - gt_iou).abs().detach().cpu())
                    self.overlap_error_count += 1

        tp = (feasibility_prediction & feasibility_target & supervised).sum(dim=0).to(torch.float64).cpu()
        fp = (feasibility_prediction & ~feasibility_target & supervised).sum(dim=0).to(torch.float64).cpu()
        fn = (~feasibility_prediction & feasibility_target & supervised).sum(dim=0).to(torch.float64).cpu()
        tn = (~feasibility_prediction & ~feasibility_target & supervised).sum(dim=0).to(torch.float64).cpu()
        self.feasibility_tp += tp
        self.feasibility_fp += fp
        self.feasibility_fn += fn
        self.feasibility_tn += tn

        empty_mask = supervised & ~feasibility_target
        predicted_empty = (~feasibility_prediction) & (prediction_sum.to(mask_logits.device) == 0)
        self.empty_correct += (predicted_empty & empty_mask).sum(dim=0).to(torch.float64).cpu()
        self.empty_total += empty_mask.sum(dim=0).to(torch.float64).cpu()

        point_count = masks.shape[1]
        small_mask = feasible_supervised & (target_sum.to(mask_logits.device) <= max(1.0, point_count * self.small_part_max_fraction))
        self.small_intersection += (intersection.to(mask_logits.device) * small_mask).sum(dim=0).to(torch.float64).cpu()
        self.small_target += (target_sum.to(mask_logits.device) * small_mask).sum(dim=0).to(torch.float64).cpu()

        supervised_scores = torch.sigmoid(feasibility_logits)[supervised].detach().cpu().tolist()
        supervised_labels = feasibility_target[supervised].to(torch.int64).detach().cpu().tolist()
        self.feasibility_scores.extend(float(value) for value in supervised_scores)
        self.feasibility_labels.extend(int(value) for value in supervised_labels)

    def compute(self) -> dict[str, object]:
        iou = self.intersection / self.union.clamp_min(1.0)
        dice = 2.0 * self.intersection / (self.prediction_sum + self.target_sum).clamp_min(1.0)
        matrix_weights = self.feasible_channels > 0
        f1 = 2.0 * self.feasibility_tp / (2.0 * self.feasibility_tp + self.feasibility_fp + self.feasibility_fn).clamp_min(1.0)
        precision = self.feasibility_tp / (self.feasibility_tp + self.feasibility_fp).clamp_min(1.0)
        recall = self.feasibility_tp / (self.feasibility_tp + self.feasibility_fn).clamp_min(1.0)
        empty_accuracy = self.empty_correct / self.empty_total.clamp_min(1.0)
        small_recall = self.small_intersection / self.small_target.clamp_min(1.0)
        macro_iou = iou[matrix_weights].mean() if bool(matrix_weights.any()) else torch.tensor(0.0, dtype=torch.float64)
        macro_dice = dice[matrix_weights].mean() if bool(matrix_weights.any()) else torch.tensor(0.0, dtype=torch.float64)
        feasibility_auc = _safe_auc(self.feasibility_scores, self.feasibility_labels)
        overlap_error = self.overlap_error_sum / max(self.overlap_error_count, 1)
        return {
            "macro_iou": float(macro_iou),
            "macro_dice": float(macro_dice),
            "macro_feasibility_f1": float(f1.mean()),
            "macro_feasibility_precision": float(precision.mean()),
            "macro_feasibility_recall": float(recall.mean()),
            "feasibility_auroc": feasibility_auc,
            "empty_mask_accuracy": float(empty_accuracy.mean()),
            "small_part_recall": float(small_recall.mean()),
            "executor_overlap_matrix_error": float(overlap_error),
            "per_task": {
                task: {
                    "iou": float(iou[index][self.feasible_channels[index] > 0].mean())
                    if bool((self.feasible_channels[index] > 0).any())
                    else 0.0,
                    "dice": float(dice[index][self.feasible_channels[index] > 0].mean())
                    if bool((self.feasible_channels[index] > 0).any())
                    else 0.0,
                    "feasible_channels": int(self.feasible_channels[index].sum()),
                }
                for index, task in enumerate(TASKS)
            },
            "per_executor": {
                executor: {
                    "iou": float(iou[:, index][self.feasible_channels[:, index] > 0].mean())
                    if bool((self.feasible_channels[:, index] > 0).any())
                    else 0.0,
                    "dice": float(dice[:, index][self.feasible_channels[:, index] > 0].mean())
                    if bool((self.feasible_channels[:, index] > 0).any())
                    else 0.0,
                    "feasibility_f1": float(f1[index]),
                    "feasibility_precision": float(precision[index]),
                    "feasibility_recall": float(recall[index]),
                    "empty_mask_accuracy": float(empty_accuracy[index]),
                    "small_part_recall": float(small_recall[index]),
                    "supervised_samples": int(self.supervised_channels[:, index].sum()),
                    "feasible_samples": int(self.feasible_channels[:, index].sum()),
                }
                for index, executor in enumerate(EXECUTORS)
            },
            "task_executor_iou": {
                task: {
                    executor: {
                        "iou": float(iou[task_index, executor_index]),
                        "dice": float(dice[task_index, executor_index]),
                        "feasible_samples": int(self.feasible_channels[task_index, executor_index]),
                        "supervised_samples": int(self.supervised_channels[task_index, executor_index]),
                    }
                    for executor_index, executor in enumerate(EXECUTORS)
                }
                for task_index, task in enumerate(TASKS)
            },
        }
