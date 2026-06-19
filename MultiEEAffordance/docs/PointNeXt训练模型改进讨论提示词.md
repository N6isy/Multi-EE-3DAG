# PointNeXt 训练模型改进讨论材料

> 用途：把本文档发给 ChatGPT 或其他模型，让对方快速理解当前 Multi-EE-3DAG / MultiEEAffordance 训练模块的基础情况，并围绕模型结构、损失函数、数据采样、评估和实验设计提出改进建议。
>
> 当前状态：PointNeXt + OpenPoints CUDA extension 已经跑通 debug 训练；debug 指标只用于证明链路可用，不代表正式模型效果。

更新时间：2026-06-17

## 1. 项目目标

当前项目是 Multi-EE Affordance 训练模块，目标是在 3D 点云上预测不同任务、不同末端执行器对应的 affordance 区域。

固定任务集合：

```text
lift / open / pull / press / push
```

固定执行器集合及通道顺序：

```text
gripper / suction / hook / dexterous_hand
```

每条训练样本对应一个：

```text
object_id + task
```

监督信号是一个多通道 mask：

```text
mask shape = [N, 4]
```

其中 4 个通道分别对应四类执行器。模型需要同时输出：

```text
1. 每个点、每个执行器的 affordance mask logits: [B, N, 4]
2. 每个任务、每个执行器是否 feasible 的 logits: [B, 4]
```

## 2. 当前数据和训练入口

数据根目录：

```text
/home/lzq/data/MultiEEAffordance
```

最终五任务标注源文件：

```text
processed/annotation_batches/final_5tasks/all_sources_5tasks_4exec_complete_aligned_posfixed.jsonl
```

训练 manifest：

```text
processed/training/v0_4_final_5tasks/manifests/train.jsonl
processed/training/v0_4_final_5tasks/manifests/val.jsonl
processed/training/v0_4_final_5tasks/manifests/test.jsonl
```

训练入口：

```bash
cd /home/lzq/data
python -m MultiEEAffordance.training.train \
  --config MultiEEAffordance/training/configs/<config_name>.json
```

评估入口：

```bash
cd /home/lzq/data
python -m MultiEEAffordance.training.evaluate \
  --config MultiEEAffordance/training/configs/<config_name>.json \
  --checkpoint /path/to/best.pt \
  --output-json /path/to/test_metrics.json
```

## 3. 当前已跑通的 PointNeXt Debug 训练

已执行命令：

```bash
cd /home/lzq/data
CUDA_VISIBLE_DEVICES=0 python -m MultiEEAffordance.training.train \
  --config MultiEEAffordance/training/configs/pointnext_shared4_5tasks_debug.json \
  --no-wandb
```

训练完成，输出目录：

```text
/home/lzq/data/MultiEEAffordance/processed/training_runs/pointnext_shared4_5tasks_debug
```

已生成：

```text
best.pt
latest.pt
history.json
resolved_config.json
```

debug 配置要点：

```json
{
  "experiment_name": "pointnext_shared4_5tasks_debug",
  "model_type": "pointnext_shared4",
  "backbone": "pointnext",
  "pointnext_cfg": "cfgs/shapenetpart/pointnext-s.yaml",
  "sample_size": 512,
  "batch_size": 2,
  "epochs": 1,
  "executor_condition_mode": "learnable_id",
  "lambda_dice": 1.0,
  "lambda_feasibility": 0.5,
  "lambda_empty_area": 0.25,
  "lambda_relation": 0.0
}
```

重要说明：这是 1 epoch debug run，目的只是验证训练链路、CUDA extension、PointNeXt adapter、loss 和 checkpoint 保存是否打通。不要把它当成正式结果。

### 3.1 Debug Train 指标

```text
train loss_total:          0.8264
train macro_iou:           0.1517
train macro_dice:          0.2253
train feasibility_auroc:   0.7305
train empty_mask_accuracy: 0.6799
train small_part_recall:   0.2088
```

train 分任务 IoU：

```text
lift:  0.3907
open:  0.1146
pull:  0.0907
press: 0.0362
push:  0.1261
```

train 分执行器 IoU：

```text
gripper:        0.1253
suction:        0.2114
hook:           0.0117
dexterous_hand: 0.2583
```

### 3.2 Debug Val 指标

```text
val loss_total:            0.8949
val macro_iou:             0.1650
val macro_dice:            0.2347
val feasibility_auroc:     0.7301
val empty_mask_accuracy:   0.6491
val small_part_recall:     0.0597
best_macro_iou:            0.165019
```

val 分任务 IoU：

```text
lift:  0.5132
open:  0.0077
pull:  0.0578
press: 0.1042
push:  0.1270
```

val 分执行器 IoU：

```text
gripper:        0.1364
suction:        0.1809
hook:           0.0968
dexterous_hand: 0.2323
```

val 中比较明显的问题：

```text
1. open / pull 的 IoU 很低。
2. suction 和 hook 的 feasibility F1 为 0，说明默认阈值下可行性分类可能严重偏置。
3. small_part_recall 很低，说明小区域 affordance 召回差。
4. lift 显著高于其他任务，模型可能先学到了最容易的几何模式。
```

## 4. 当前模型结构

相关代码：

```text
training/backbones/pointnext_adapter.py
training/model_factory.py
training/executor_conditioning.py
training/losses.py
training/dataset.py
```

### 4.1 Backbone

当前 PointNeXt adapter 使用外部 OpenPoints 实现：

```text
external/backbones/PointNeXt-master
```

它加载：

```text
PointNextEncoder
PointNextDecoder
cfgs/shapenetpart/pointnext-s.yaml
```

当前 adapter 不使用 OpenPoints 原始 segmentation head，而是只取 per-point features：

```python
features = backbone(points)  # [B, N, D]
```

输入点云：

```text
[B, N, C]
C = 3 或 6
当前 debug 使用 C = 3
```

dataset 会对 xyz 做中心化和尺度归一化。

### 4.2 Task / Executor Condition Head

当前模型整体是：

```text
PointNeXt backbone -> per-point features -> shared task/executor condition head
```

head 的核心思想：

```text
1. 点特征投影到 hidden_dim。
2. 点云全局 max pooling 得到 global feature。
3. task_id 经过 task embedding。
4. executor 经过 executor condition encoder。
5. task feature + executor feature 形成 4 个 executor query。
6. mask logits 由 point feature 与 executor query 做点积得到。
7. feasibility logits 用 global feature + executor query 预测。
```

简化公式：

```text
mask_features = tanh(point_projection(point_features) + global_projection(global_max))
queries = tanh(task_query(task_embedding) + executor_condition)
mask_logits = einsum(mask_features, queries) / sqrt(hidden_dim) + mask_bias
```

### 4.3 已实现的执行器条件模式

当前支持：

```text
learnable_id
one_hot_id
attr_only
id_attr
id_attr_film
id_attr_crossattn
```

执行器属性文件：

```text
training/configs/executor_specs_v0_1.json
```

属性包括：

```text
contact_mode
contact_geometry
requires_flatness
requires_edge_or_thin_part
can_hook_inside
can_press
can_pull
can_lift
tip_width_norm
contact_patch_norm
dof_norm
```

已有 PointNeXt 属性消融配置：

```text
pointnext_executor_attr_only_5tasks.json
pointnext_executor_id_attr_5tasks.json
pointnext_executor_id_attr_film_5tasks.json
pointnext_executor_id_attr_crossattn_5tasks.json
```

## 5. 当前 Loss 设计

相关代码：

```text
training/losses.py
```

总 loss：

```text
total =
  mask_focal
  + lambda_dice * mask_dice
  + lambda_empty_area * mask_empty_area
  + lambda_feasibility * feasibility
  + lambda_relation * relation
```

当前 debug：

```text
lambda_dice = 1.0
lambda_empty_area = 0.25
lambda_feasibility = 0.5
lambda_relation = 0.0
```

mask focal 和 dice 只在 feasible executor 通道上计算。

empty-area loss 用于约束 infeasible executor 通道不要预测出大面积正区域。

relation loss 已实现但 debug 中没有开启。它约束不同 executor 之间预测 overlap 与真实 overlap 的差异。

## 6. 当前最需要讨论的问题

请重点讨论下面这些问题，不要只给泛泛建议。

### 6.1 模型结构是否过于简单

当前 head 本质是：

```text
共享 point features + task/executor query 点积
```

需要讨论：

```text
1. 是否应该把 task 和 executor query 做得更强，例如用 Transformer decoder query。
2. 是否应该让每个 task-executor pair 有独立 query，而不是 task + executor 简单相加。
3. 是否应该让 executor 属性直接调制每层 point feature，而不只在 head 末端使用。
4. FiLM / cross-attention / hypernetwork 哪个更适合表达不同末端执行器的接触模式。
5. 是否需要 multi-scale decoder feature fusion，而不仅用最后 decoder 输出的 per-point feature。
```

### 6.2 类别不平衡和小区域召回

debug 显示：

```text
small_part_recall 很低
hook / press / open / pull 相对弱
部分 executor feasibility F1 为 0
```

需要讨论：

```text
1. 是否需要 task-executor balanced sampler。
2. 是否需要按 positive point ratio 设计 reweighting。
3. 是否需要 focal alpha 或 Tversky / focal Tversky loss。
4. 是否需要针对 small positive masks 的 hard mining。
5. 是否需要对 empty mask 和 feasible mask 分别设不同阈值。
```

### 6.3 Feasibility 与 Mask 是否应该解耦或互相约束

当前模型同时预测：

```text
mask_logits [B,N,4]
feasibility_logits [B,4]
```

但 mask 和 feasibility 只通过共享特征间接关联。

需要讨论：

```text
1. 是否应该让 feasibility 从 mask pooling 中预测，而不是只看 global feature + query。
2. 是否应该用 feasibility-gated mask loss 或 consistency loss。
3. 是否应该先判断 feasible，再对 feasible 通道预测 mask。
4. 是否应该对 infeasible 通道使用更强的 empty-area penalty。
```

### 6.4 PointNeXt 是否需要预训练或更强输入特征

当前 debug 使用：

```text
input_channels = 3
sample_size = 512
```

需要讨论：

```text
1. 正式训练是否应该使用 sample_size=2048 或更高。
2. 如果点云有 RGB / normals / curvature，是否应该接入 input_channels=6 或更多几何特征。
3. 是否需要 PointNeXt 预训练权重。
4. 是否先 freeze backbone 再 unfreeze，还是端到端训练。
5. 是否应该比较 PointNeXt-S / PointNeXt-B / PointNeXt-L。
```

### 6.5 评估策略

正式结果不能只看 overall mIoU。

需要讨论：

```text
1. 每个 task 的 IoU / Dice。
2. 每个 executor 的 IoU / Dice。
3. 5 x 4 task-executor matrix。
4. feasibility AUROC / F1 / precision / recall。
5. threshold calibration 后的 mask 指标。
6. feasibility-gated mask 指标。
7. small_part_recall。
8. executor overlap matrix error。
```

## 7. 建议让 ChatGPT 输出什么

希望 ChatGPT 给出：

```text
1. 对当前模型结构的诊断。
2. 最值得优先尝试的 3 个模型改进。
3. 最值得优先尝试的 3 个 loss / sampler 改进。
4. 一套 1-2 周内可以完成的实验计划。
5. 每个实验应该对比什么 baseline，预期改善哪个指标。
6. 哪些改动可能造成无效实验或污染论文结果。
7. 对 AAAI 论文主表和消融表的建议。
```

## 8. 可直接复制给 ChatGPT 的提示词

下面这段可以直接复制到 ChatGPT：

```text
我正在做一个 3D multi-end-effector affordance segmentation 项目，请你基于下面信息，帮我讨论当前训练模型如何改进。请不要给泛泛建议，要结合我的数据形态、模型结构、loss、当前 debug 指标，提出可执行的模型改进和实验计划。

项目任务：
- 输入：物体 3D 点云，shape [N,3] 或 [N,6]。
- 固定任务：lift / open / pull / press / push。
- 固定执行器：gripper / suction / hook / dexterous_hand。
- 每条训练样本是 object_id + task。
- 监督 mask shape = [N,4]，4 个通道对应四类执行器。
- 模型需要输出每点每执行器的 mask logits [B,N,4]，以及每执行器 feasibility logits [B,4]。

当前数据：
- 数据根目录：/home/lzq/data/MultiEEAffordance。
- train/val/test manifest 位于 processed/training/v0_4_final_5tasks/manifests/。
- split 按 CAD asset 做，避免同 asset 泄漏。
- mask 是人工清洗后的五任务四执行器标注。

当前模型：
- Backbone 是 PointNeXt，使用 OpenPoints 的 PointNextEncoder + PointNextDecoder。
- 当前没有使用 OpenPoints 原始 segmentation head，只把 PointNeXt 当作 per-point feature extractor。
- backbone 输出 features = [B,N,D]。
- 后接共享的 TaskExecutorConditionHead。
- head 做法：
  1. point features 投影到 hidden_dim。
  2. global max pooling 得到 global feature。
  3. task_id 经过 task embedding。
  4. executor 经过 executor condition encoder。
  5. task feature + executor feature 形成 executor query。
  6. mask logits = point feature 与 executor query 点积 / sqrt(hidden_dim) + mask_bias。
  7. feasibility logits 由 global feature + executor query 预测。

当前 executor condition 已实现这些模式：
- learnable_id
- one_hot_id
- attr_only
- id_attr
- id_attr_film
- id_attr_crossattn

执行器属性包括：
- contact_mode
- contact_geometry
- requires_flatness
- requires_edge_or_thin_part
- can_hook_inside
- can_press
- can_pull
- can_lift
- tip_width_norm
- contact_patch_norm
- dof_norm

当前 loss：
- total = mask_focal + lambda_dice * mask_dice + lambda_empty_area * mask_empty_area + lambda_feasibility * feasibility + lambda_relation * relation。
- mask focal 和 dice 只在 feasible executor 通道上算。
- empty-area loss 约束 infeasible 通道不要预测大面积正区域。
- relation loss 已实现，用于约束 executor 之间 overlap，但 debug 中 lambda_relation=0。

当前 debug run：
- config = pointnext_shared4_5tasks_debug.json。
- sample_size=512, batch_size=2, epochs=1。
- 这只是链路验证，不是正式结果。
- train macro_iou=0.1517, macro_dice=0.2253, feasibility_auroc=0.7305。
- val macro_iou=0.1650, macro_dice=0.2347, feasibility_auroc=0.7301。
- val per task IoU: lift=0.5132, open=0.0077, pull=0.0578, press=0.1042, push=0.1270。
- val per executor IoU: gripper=0.1364, suction=0.1809, hook=0.0968, dexterous_hand=0.2323。
- val small_part_recall=0.0597。
- suction 和 hook 的 feasibility F1 在 debug val 中为 0，说明默认阈值或分类头可能有严重偏置。

我想请你重点分析：
1. 当前 PointNeXt + task/executor query dot-product head 是否表达能力不足？如何改？
2. task 和 executor 的条件建模应该用 learnable id、属性、FiLM、cross-attention、hypernetwork，还是 task-executor pair query？
3. 如何提升 open / pull / press / hook / suction 这些弱项？
4. 如何处理 small positive region 和类别不平衡？
5. feasibility head 应该如何与 mask head 耦合？是否需要 mask pooling consistency 或 feasibility-gated loss？
6. relation loss 是否应该开启？还可以设计哪些 executor overlap / compatibility loss？
7. 正式训练时 sample_size、batch_size、学习率、warmup、backbone freeze/unfreeze、预训练权重应该怎么设？
8. 请设计一套 1-2 周内可执行的实验计划，包含 baseline、消融、指标、预期收益和失败风险。

请输出：
- 对当前模型的诊断。
- 优先级最高的 3 个模型结构改进。
- 优先级最高的 3 个 loss / sampler / training 改进。
- 详细实验表格：实验名、改动、对比 baseline、看哪些指标、预期效果、风险。
- 哪些结果可以放论文主表，哪些只适合做消融或附录。
```

## 9. 下一步本地操作建议

在和 ChatGPT 讨论之前，建议先补一个 test 评估，确认 debug checkpoint 可以完整评估：

```bash
cd /home/lzq/data
conda activate multiee-train

CUDA_VISIBLE_DEVICES=0 python -m MultiEEAffordance.training.evaluate \
  --config MultiEEAffordance/training/configs/pointnext_shared4_5tasks_debug.json \
  --checkpoint /home/lzq/data/MultiEEAffordance/processed/training_runs/pointnext_shared4_5tasks_debug/best.pt \
  --output-json processed/training_runs/pointnext_shared4_5tasks_debug/test_metrics.json
```

如果通过，再开始正式 PointNeXt shared-4 训练：

```bash
cd /home/lzq/data
conda activate multiee-train

CUDA_VISIBLE_DEVICES=0 python -m MultiEEAffordance.training.train \
  --config MultiEEAffordance/training/configs/pointnext_shared4_5tasks.json
```

如果暂时不想记录 wandb：

```bash
CUDA_VISIBLE_DEVICES=0 python -m MultiEEAffordance.training.train \
  --config MultiEEAffordance/training/configs/pointnext_shared4_5tasks.json \
  --no-wandb
```

建议第一轮正式实验顺序：

```text
1. pointnext_shared4_5tasks.json
2. pointnext_one_hot_executor_token_5tasks.json
3. pointnext_with_relation_loss_5tasks.json
4. pointnext_executor_id_attr_5tasks.json
5. pointnext_executor_id_attr_film_5tasks.json
6. pointnext_executor_id_attr_crossattn_5tasks.json
```

每个实验至少保存：

```text
best.pt
history.json
test_metrics.json
calibrated_metrics.json
task_executor_matrix.json / csv
```

