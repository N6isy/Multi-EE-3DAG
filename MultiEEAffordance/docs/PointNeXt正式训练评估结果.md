# PointNeXt 正式训练与评估结果

> 记录对象：`pointnext_shared4_5tasks`
>
> 训练状态：正式 50 epoch 训练完成，test / threshold calibration / feasibility-gated evaluation 均已完成。

更新时间：2026-06-17

## 1. Run 信息

训练配置：

```text
training/configs/pointnext_shared4_5tasks.json
```

输出目录：

```text
processed/training_runs/pointnext_shared4_5tasks/
```

关键文件：

```text
best.pt
latest.pt
history.json
resolved_config.json
test_metrics.json
thresholds.json
calibrated_metrics.json
calibrated_metrics_task_executor_matrix.csv
calibrated_gated_metrics.json
calibrated_gated_metrics_task_executor_matrix.csv
```

W&B run：

```text
https://wandb.ai/nao15506066880-beijing-institute-of-technology/multiee-affordance/runs/vww9l4pb
```

配置要点：

```text
backbone: PointNeXt
pointnext_cfg: cfgs/shapenetpart/pointnext-s.yaml
sample_size: 2048
batch_size: 16
epochs: 50
learning_rate: 0.001
executor_condition_mode: learnable_id
lambda_dice: 1.0
lambda_feasibility: 0.5
lambda_empty_area: 0.25
lambda_relation: 0.0
```

## 2. Validation 最优点

`best.pt` 对应 validation macro IoU 最优点：

```text
best epoch: 48
val macro_iou: 0.6015
val macro_dice: 0.7414
val feasibility_auroc: 0.9569
val small_part_recall: 0.4949
val executor_overlap_matrix_error: 0.1996
```

epoch 50 指标：

```text
val macro_iou: 0.5962
val macro_dice: 0.7367
val feasibility_auroc: 0.9548
```

判断：模型后期基本收敛，最后两轮仅轻微回落，没有明显训练崩溃。

## 3. Test 主指标

### 3.1 三种评估设置对比

| metric | test, default threshold | calibrated | calibrated + feasibility gate |
| --- | ---: | ---: | ---: |
| macro_iou | 0.5794 | **0.5887** | 0.5326 |
| macro_dice | 0.7147 | **0.7214** | 0.6630 |
| macro_feasibility_f1 | 0.7984 | **0.8146** | **0.8146** |
| feasibility_auroc | 0.9715 | 0.9715 | 0.9715 |
| empty_mask_accuracy | 0.2067 | 0.1788 | **0.9250** |
| small_part_recall | 0.5842 | **0.6074** | 0.5807 |
| executor_overlap_matrix_error | **0.1715** | 0.1786 | 0.2067 |

结论：

```text
1. 如果主表以 mask mIoU / Dice 为核心，优先报告 calibrated 版本。
2. calibrated 相比 default threshold 小幅提升 mIoU、Dice、feasibility F1 和 small-part recall。
3. feasibility-gated 版本显著提升 empty_mask_accuracy，但明显牺牲 mIoU。
4. gated 版本适合作为补充分析，说明模型可以强抑制 infeasible 通道，但不应作为最高 mIoU 主结果。
```

### 3.2 校准阈值

来自 validation split：

| executor | mask threshold | feasibility threshold | val mask calibration IoU | val feasibility calibration F1 |
| --- | ---: | ---: | ---: | ---: |
| gripper | 0.45 | 0.35 | 0.7081 | 0.8122 |
| suction | 0.40 | 0.20 | 0.6973 | 0.8438 |
| hook | 0.50 | 0.20 | 0.6850 | 0.7647 |
| dexterous_hand | 0.45 | 0.10 | 0.7516 | 0.8502 |

观察：

```text
1. feasibility threshold 普遍低于 0.5，说明默认 0.5 阈值偏保守。
2. dexterous_hand 的 feasibility threshold 只有 0.1，后续需要检查 feasibility logit calibration。
3. hook 的 mask threshold 保持 0.5，但 feasibility threshold 降到 0.2。
```

## 4. Test 分任务结果

### 4.1 Default threshold

| task | IoU | Dice |
| --- | ---: | ---: |
| lift | 0.6881 | 0.8102 |
| open | 0.5627 | 0.7115 |
| pull | 0.5643 | 0.7140 |
| press | 0.6322 | 0.7704 |
| push | 0.4628 | 0.5813 |

### 4.2 Calibrated

| task | IoU | Dice |
| --- | ---: | ---: |
| lift | 0.7080 | 0.8263 |
| open | 0.5665 | 0.7148 |
| pull | 0.5758 | 0.7227 |
| press | 0.6474 | 0.7811 |
| push | 0.4605 | 0.5771 |

结论：

```text
1. calibrated 对 lift / open / pull / press 有提升。
2. push 仍是最弱任务，calibration 后略降。
3. press 在 test 上不弱，但 validation 上较弱，说明 press 的 split 间波动需要关注。
```

## 5. Test 分执行器结果

### 5.1 Default threshold

| executor | IoU | Dice | feasibility F1 | small-part recall |
| --- | ---: | ---: | ---: | ---: |
| gripper | 0.5522 | 0.6987 | 0.8162 | 0.7756 |
| suction | 0.6666 | 0.7974 | 0.8167 | 0.2258 |
| hook | 0.3934 | 0.5293 | 0.7111 | 0.6535 |
| dexterous_hand | 0.6682 | 0.7965 | 0.8497 | 0.6819 |

### 5.2 Calibrated

| executor | IoU | Dice | feasibility F1 | small-part recall |
| --- | ---: | ---: | ---: | ---: |
| gripper | 0.5550 | 0.6993 | 0.8062 | 0.7915 |
| suction | 0.6897 | 0.8153 | 0.8352 | 0.2823 |
| hook | 0.3934 | 0.5293 | 0.7564 | 0.6535 |
| dexterous_hand | 0.6777 | 0.8033 | 0.8605 | 0.7024 |

结论：

```text
1. suction 和 dexterous_hand 是当前最强执行器。
2. hook 是当前最弱执行器，主要受样本少和可行样本少影响。
3. suction 的 small-part recall 仍低，可能因为 suction 标注区域更偏大平面，或小部件正样本极少。
4. gripper 的 mIoU 中等，但 small-part recall 较好。
```

## 6. 关键问题

当前 baseline 已经可作为 PointNeXt shared-4 主基线，但有几个明显问题：

```text
1. empty_mask_accuracy 在不 gated 时很低，说明 infeasible 通道仍有较多正预测。
2. feasibility gate 虽然把 empty_mask_accuracy 提到 0.9250，但 mIoU 从 0.5887 降到 0.5326。
3. push 是当前最弱任务。
4. hook 是当前最弱执行器。
5. executor overlap error 在 calibrated / gated 后变差，说明阈值策略改变了 executor 间 overlap 结构。
```

## 7. 当前推荐汇报方式

主表建议报告：

```text
pointnext_shared4_5tasks calibrated:
macro_iou = 0.5887
macro_dice = 0.7214
macro_feasibility_f1 = 0.8146
feasibility_auroc = 0.9715
small_part_recall = 0.6074
```

补充表或消融表报告：

```text
default threshold:
macro_iou = 0.5794

calibrated + feasibility gate:
macro_iou = 0.5326
empty_mask_accuracy = 0.9250
```

建议不要只报告 gated 版本，因为它的 mIoU 明显低于 calibrated 版本。gated 版本更适合说明 feasibility head 对不可行通道的过滤价值。

## 8. 下一步实验优先级

第一优先级：

```text
1. pointnext_one_hot_executor_token_5tasks
   目的：判断 learnable executor id 是否比 one-hot 更稳定。

2. pointnext_with_relation_loss_5tasks
   目的：改善 executor_overlap_matrix_error 和通道间结构一致性。

3. pointnext_executor_id_attr_5tasks
   目的：验证显式执行器属性是否改善 hook / push / feasibility calibration。
```

第二优先级：

```text
4. pointnext_executor_id_attr_film_5tasks
   目的：让执行器属性直接调制点特征，可能改善不同接触模式。

5. pointnext_executor_id_attr_crossattn_5tasks
   目的：验证 executor query 到点特征的 cross-attention 是否优于简单点积。
```

建议每个新实验都生成同样三套评估：

```bash
python -m MultiEEAffordance.training.evaluate ...
python -m MultiEEAffordance.training.evaluate_calibrated ...
python -m MultiEEAffordance.training.evaluate_calibrated --reuse-thresholds --feasibility-gate ...
```

并统一比较：

```text
macro_iou
macro_dice
macro_feasibility_f1
feasibility_auroc
empty_mask_accuracy
small_part_recall
executor_overlap_matrix_error
per_task_iou
per_executor_iou
5 x 4 task-executor matrix
```

