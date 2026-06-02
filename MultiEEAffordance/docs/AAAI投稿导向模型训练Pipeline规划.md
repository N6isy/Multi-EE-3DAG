# AAAI 投稿导向模型训练 Pipeline 规划

更新时间：2026-06-01

本文档定义研究思路和实施计划。当前已经新增独立初版训练目录 `MultiEEAffordance/training/`，用于并行验证数据 schema、object-disjoint split 和最小训练闭环。正式论文实验仍应等待一批 verified 标注冻结。

训练侧只接受五个独立任务：`lift/open/pull/press/push`。旧复合任务只存在于历史候选生成和人工审查输入准备阶段，不进入训练模型、训练 loss 或训练评估。

## 1. 论文问题定义

输入：

```text
物体点云 P + 五任务指令 q
```

任务集合：

```text
lift / open / pull / press / push
```

输出：

```text
[M_gripper, M_suction, M_hook, M_dexterous_hand]
```

其中每个 `M` 是长度为 `N` 的点级 mask。模型还应输出四类执行器的 object-task feasibility：

```text
F_gripper / F_suction / F_hook / F_dexterous_hand
```

核心研究问题不是普通的单 mask affordance segmentation，而是：

> 在同一个物体、同一个任务下，不同末端执行器具有不同接触机制。模型能否同时预测四类执行器的可操作区域，并显式区分它们的物理约束差异？

## 2. 论文主线

建议论文主线写成：

```text
Multi-EE Affordance Grounding
  = task-conditioned
  + heterogeneous end-effector-aware
  + multi-label point-level prediction
  + feasibility-aware empty-mask modeling
```

数据集贡献和方法贡献需要并行成立：

1. 数据贡献：构建面向四类异构执行器、五任务、点级 `[N,4]` mask 的 object-level 数据集。
2. 方法贡献：提出显式建模 task、executor mechanism 和局部几何关系的多执行器 affordance grounding 网络。
3. 评估贡献：除总体 mIoU 外，增加执行器差异、空 mask、跨来源泛化和小部件召回评估。

## 3. 推荐模型：HeteroAffordanceFormer

暂定模型名：

```text
HeteroAffordanceFormer
```

模型由五个模块组成。

### 3.1 3D 点云编码器

输入：

```text
P in R^[N,3] 或 R^[N,6]
```

第一版使用成熟 point backbone：

```text
PointNeXt 或 Point Transformer 系列
```

输出多尺度点特征：

```text
F_point in R^[N,D]
```

建议先以 PointNeXt 作为稳定 baseline，再增加 Point Transformer 作为更强 backbone 对照。论文创新点不要建立在重新发明 backbone 上。

### 3.2 五任务语义编码器

输入五任务 instruction：

```text
Lift the object from the supporting surface.
Open an articulated or openable component.
Pull a handle, ring, lip, panel, or movable component along the pull direction.
Press a button, key, switch, or local pressable part.
Push a panel, surface, button, or movable component along the push direction.
```

建议保留两种实现用于实验：

1. `task_id embedding`：五个可学习向量，作为低成本 baseline。
2. `text embedding + MLP`：使用冻结文本编码器生成 instruction embedding，用于验证语义表达能力。

五个任务在训练模型中作为五个独立类别处理：

```text
lift
open
pull
press
push
```

第一版使用五个独立的 `task_id embedding`。不增加旧复合任务 ancestor，不使用旧复合任务作为辅助标签，不让模型重新学习混合语义。

### 3.3 执行器机制 Token

为四类执行器建立可学习 token：

```text
E_gripper
E_suction
E_hook
E_dexterous_hand
```

每个 token 不只是类别 ID，还应编码机制先验：

| 执行器 | 机制重点 |
| --- | --- |
| `gripper` | 对夹、边缘、柄部、双侧接触 |
| `suction` | 局部平整、法向稳定、足够面积 |
| `hook` | 孔、环、内边界、挂接和提拉约束 |
| `dexterous_hand` | 包覆抓握、按钮、开关、精细操作 |

实现上不需要把规则硬写成 GT。建议把局部几何描述符作为额外 feature 输入，让 executor token 通过 cross-attention 自主学习如何使用这些 feature。

### 3.4 Task-Executor Cross Attention

组合 task token 和 executor token：

```text
Q_e = Fuse(E_executor, E_task)
```

四个 query 分别对点云特征做 cross-attention：

```text
Q_gripper         -> F_point -> M_gripper
Q_suction         -> F_point -> M_suction
Q_hook            -> F_point -> M_hook
Q_dexterous_hand  -> F_point -> M_dexterous_hand
```

共享 point backbone，但不共享最后一个 executor query。这样既能学习公共物体几何，又能避免四个 mask 退化成几乎相同的输出。

### 3.5 Feasibility 与空 Mask 分支

数据集中必须保留无合适操作区域的 object-task-executor 组合。模型因此需要显式预测：

```text
F_e in [0,1]
```

当 `F_e` 较低时，对应 mask 应为空或整体低置信度。这个分支非常重要，因为它体现了数据集的区别性：不是每个物体、每个任务、每个执行器都一定存在正例区域。

## 4. 局部几何输入

为了让模型更好地区分执行器机制，建议在 xyz 或 xyz+normal 之外加入轻量局部几何描述：

```text
normal consistency
local curvature proxy
local density
boundary score
thin-structure score
local component scale
```

这些描述符只作为输入 feature，不直接决定标签。

后续可以增加 hole/ring proxy，但不要一开始就把复杂拓扑检测强绑定到主模型。先用可解释、稳定、可批处理的几何量建立 baseline。

## 5. 训练损失

建议总损失：

```text
L =
  L_mask
  + lambda_feas * L_feasibility
  + lambda_rel * L_executor_relation
```

### 5.1 点级 Mask 损失

每个执行器独立计算：

```text
BCE 或 Focal BCE + Dice Loss
```

小部件容易被大面积背景淹没，建议优先验证：

```text
Focal BCE + Dice
```

### 5.2 Feasibility 损失

使用四通道 object-level BCE：

```text
L_feasibility = BCE(F_pred, F_gt)
```

空 mask 样本必须参与该分支训练。

### 5.3 Executor Relation 损失

不能简单强迫四个通道互斥，因为同一区域可能同时适合 gripper 和 dexterous_hand。建议监督 predicted pairwise overlap matrix 接近 GT overlap matrix：

```text
R_pred[e_i,e_j] = soft IoU(M_i, M_j)
R_gt[e_i,e_j]   = IoU(GT_i, GT_j)
L_executor_relation = distance(R_pred, R_gt)
```

目标是学习“该重叠时重叠、该分开时分开”，而不是机械去重。

## 6. 数据划分

所有 split 必须按 object 划分，不按 sample row 随机划分。

原因：

```text
同一物体可能有多个 task 和 executor 行
```

如果同一物体进入 train 和 test，会产生严重泄漏。

建议准备四套评估：

1. `standard split`：object-disjoint，类别分布近似平衡。
2. `cross-source split`：3D AffordanceNet 训练，PartNet-Mobility 测试，验证跨来源泛化。
3. `small-part subset`：button、knob、switch、handle joint、thin structure。
4. `empty-mask subset`：验证模型能否正确判断不可行组合。

## 7. 指标

基础指标：

```text
mIoU
mDice
per-task mIoU
per-executor mIoU
macro average
```

必须增加：

```text
feasibility F1 / AUROC
empty-mask accuracy
small-part recall
cross-source mIoU
executor overlap matrix error
```

其中 macro average 必须作为主指标，避免大类别、大平面和高频执行器掩盖 small part、hook、press 等困难样本。

## 8. Baseline 与消融

### 8.1 Baseline

至少准备：

1. `PointNet++ + shared 4-channel head`
2. `PointNeXt + shared 4-channel head`
3. `PointNeXt + four independent heads`
4. `Point Transformer + shared 4-channel head`
5. `HeteroAffordanceFormer`

### 8.2 消融实验

至少准备：

1. 去掉 executor token。
2. 去掉 task text embedding，只用 task ID。
3. 去掉 feasibility branch。
4. 去掉 executor relation loss。
5. 去掉 local geometry feature。
6. 只使用 3D AffordanceNet。
7. 增加 PartNet-Mobility 后的跨来源对比。

## 9. 数据质量分层

训练时区分：

```text
weak
checked
verified
```

建议阶段：

1. 只用 `verified` 做最可信测试集。
2. `checked + verified` 用作第一版监督训练集。
3. `weak` 只用于预训练或半监督实验，不能与人工 GT 混为同一权重。

人工审查阶段要额外标记：

```text
source_dataset
reviewer
task_taxonomy_version
quality_flag
empty-mask decision
```

## 10. 与人工审查并行推进的节奏

### 阶段 A：立即开始

不等待 1.2w 全部完成，先冻结：

```text
200-300 条 verified 样本
```

用途：

- 检查 dataloader schema。
- 检查 object-disjoint split。
- 跑通最小 baseline。
- 统计各任务、各执行器、空 mask 和 small part 分布。

### 阶段 B：第一版可训练集

达到：

```text
2000-3000 条 checked/verified 样本
```

用途：

- 训练 PointNet++ / PointNeXt baseline。
- 验证 feasibility branch。
- 检查是否存在类别和执行器严重失衡。

### 阶段 C：完整数据集

达到：

```text
约 1.2w 五任务样本行
```

用途：

- 训练完整 HeteroAffordanceFormer。
- 完成 cross-source、small-part、empty-mask 和消融实验。
- 冻结论文主表。

## 11. AAAI 投稿前必须回答的问题

1. 多执行器同时预测相比单执行器独立训练是否更好？
2. executor token 是否真的学到了不同物理机制？
3. feasibility branch 是否显著改善空 mask 判断？
4. 五个独立任务之间是否呈现可解释的语义和区域差异？
5. PartNet-Mobility 是否提升 articulated object 和 small part 泛化？
6. 模型在 hook、button、handle、flat panel 等关键结构上是否有可解释可视化？
7. 同一物体上四类执行器预测的差异是否符合物理直觉？

## 12. 当前建议

现在不要直接开始写完整训练框架。先完成：

```text
数据 schema 冻结
object-disjoint split 规则冻结
200-300 条 verified 样本冻结
类别 / task / executor / empty-mask 分布统计
```

当前已经实现轻量 `TaskExecutorPointNet` 作为数据闭环 baseline。完成 verified 数据冻结后，再将点云 backbone 替换为 PointNeXt，并逐步加入更强的 executor token、feasibility branch、局部几何特征和 relation loss。这样每个模块都能通过消融实验独立说明价值。
