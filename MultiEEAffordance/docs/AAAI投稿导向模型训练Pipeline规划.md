# AAAI 投稿导向模型训练 Pipeline 规划

更新时间：2026-06-15

本文档定义五任务 Multi-EE Affordance Grounding 的实验路线。当前数据质量只有一种：人工标注/人工审查后的标注结果。训练 pipeline 不再区分 `weak/checked/verified` 层级，也不做质量加权。

## 1. 问题定义

输入：

```text
object-level 3D point cloud P
task instruction q in {lift, open, pull, press, push}
```

输出：

```text
M_gripper
M_suction
M_hook
M_dexterous_hand

F_gripper
F_suction
F_hook
F_dexterous_hand
```

其中 `M_*` 是点级 affordance mask，最终 shape 固定为 `[N,4]`；`F_*` 是 object-task-executor feasibility。

论文表述建议使用：

```text
task-conditioned heterogeneous end-effector affordance grounding
```

不要过度宣称 open-vocabulary language grounding。当前任务集合只有五个固定任务。

## 2. 数据划分与防泄漏

仅 object-disjoint split 不够。正式实验使用 CAD asset-level split。

规则：

- `asset_uid = source_dataset + ":" + source_asset_id`
- 3D AffordanceNet 如果只有 `object_id=3danet_full_xxx`，则 `source_asset_id=object_id`
- PartNet-Mobility 后续使用其原始 model id 作为 `source_asset_id`
- 同一 `asset_uid` 的所有 task、executor、empty mask、重复审查样本必须进入同一 split
- 禁止按 sample row 随机划分

当前最终数据先由 `prepare_final_5task_training_dataset.py` 生成训练 manifest，再审计 split。已实现审计脚本：

```bash
python -m MultiEEAffordance.training.audit_splits \
  --dataset-root /home/lzq/data/MultiEEAffordance \
  --manifest processed/training/v0_4_final_5tasks/manifests/all.jsonl \
  --output-json processed/training/v0_4_final_5tasks/split_audit.json \
  --fail-on-leakage
```

论文需要报告：

- train/val/test asset 数量
- object/category/task/executor 分布
- empty-mask ratio
- 是否存在 asset/object leakage
- cross-source split：3D AffordanceNet 与 PartNet-Mobility 的跨来源泛化

## 3. 人工标注一致性

因为训练数据只有人工标注一种质量层级，必须证明标注可信。

最低成本方案：

- 从最终数据中抽取至少 50 个 CAD asset 作为 calibration subset
- 两位 reviewer 都标注这批样本
- 统计 point-level mask IoU、feasibility agreement、empty-mask agreement

如果保留了两位 reviewer 的原始 refined samples，可继续使用一致性审计脚本：

```bash
python -m MultiEEAffordance.training.audit_annotation_consistency \
  --dataset-root /home/lzq/data/MultiEEAffordance \
  --reviewed-samples processed/annotation_batches/v0_2_5tasks/reviewer_a_refined_samples.jsonl,processed/annotation_batches/v0_2_5tasks/reviewer_b_refined_samples.jsonl \
  --output-json processed/training/v0_4_final_5tasks/annotation_consistency.json \
  --output-csv processed/training/v0_4_final_5tasks/annotation_disagreements.csv
```

建议写入论文 appendix：

- mean / median point IoU between reviewers
- feasibility agreement
- empty-mask agreement
- top disagreement cases by task and executor

## 4. Feasibility 与 empty-mask 建模

当前定义：

```text
F_gt = 1  <=>  该 object-task-executor 可行，且人工 mask 非空
F_gt = 0  <=>  该组合被人工确认不可行，mask 为空
```

训练 loss 采用条件式 mask loss：

- feasible 通道：计算 Focal BCE + Dice
- empty 通道：不计算 Dice，只计算 empty-area penalty 和 feasibility BCE
- 所有 supervised 通道都计算 feasibility BCE
- relation loss 默认关闭，只作为消融

核心原因：

- 空 mask 上 Dice 没有稳定意义
- 全零预测不能靠 empty-mask accuracy 获得虚高结论
- feasibility 是数据集区分异构执行器能力的关键分支

## 5. Baseline 和消融

当前仓库已能运行轻量 `TaskExecutorPointNet` 闭环，配置包括：

```text
pointnet_shared4_5tasks
pointnet_no_executor_token_5tasks
pointnet_shared_executor_token_5tasks
pointnet_one_hot_executor_token_5tasks
pointnet_random_frozen_executor_token_5tasks
pointnet_executor_token_swap_5tasks
pointnet_with_relation_loss_5tasks
single_ee_gripper_5tasks
single_ee_suction_5tasks
single_ee_hook_5tasks
single_ee_dexterous_hand_5tasks
```

AAAI 主实验还必须补强 baseline：

```text
PointNet++ + shared 4-channel head
PointNeXt + shared 4-channel head
PointNeXt + four independent heads
four independent single-EE models
Point Transformer + shared 4-channel head
geometry-only heuristic baseline
HeteroAffordanceFormer
```

注意：当前代码尚未实现 PointNeXt、Point Transformer 和 HeteroAffordanceFormer 主模型。不能用当前 MLP PointNet 结果冒充这些强 baseline。
配置目录中已用 `MultiEEAffordance/training/configs/README.md` 明确区分“可直接运行配置”和“待实现强 baseline”。

four independent single-EE models 是必须项，因为它直接回答：joint multi-executor learning 是否真的优于为每个执行器单独训练模型。比较时需要报告参数量和 FLOPs，避免 joint model 只是参数更多。

## 6. Executor token 论证

executor token 消融至少包含：

- no executor token
- shared executor token
- random frozen executor token
- learnable executor token
- executor token swap
- single-EE models

要证明 executor token 不只是类别 ID，需要：

- 同一 object-task 下四执行器预测可视化
- token swap 后性能下降或预测机制错位
- mechanism-specific subset 指标

机制子集建议：

```text
gripper: handle / thin edge / bilateral graspable region
suction: flat panel / smooth local surface
hook: hole / ring / inner boundary / handle loop
dexterous_hand: button / switch / complex local manipulation
```

## 7. Relation loss 风险

relation loss 不作为第一版主模型核心，只作为 ablation。

启用条件：

- 双方 executor 通道均 supervised
- 双方 `F_gt=1`
- 双方 GT mask 点数大于最小阈值
- `lambda_relation` 建议不超过 `0.05`

必须报告 with / without relation loss，若 relation loss 提升 overlap error 但损害单通道 mIoU，应优先保留单通道 mIoU 更好的模型。

## 8. Local geometry feature 风险

几何特征必须满足：

- label-free
- task-agnostic
- executor-agnostic
- 不直接编码人工规则

消融顺序：

```text
xyz only
xyz + normal
xyz + normal + curvature/density
xyz + normal + curvature/density + boundary/thin-structure
```

第一版不要加入强 hole/ring proxy，避免被审稿人质疑为手工规则 shortcut。

## 9. 主结果表设计

主表至少包含：

```text
Method
Backbone
Params
FLOPs
Macro mIoU
Macro Dice
Feasibility F1
Feasibility AUROC
Empty-mask Acc
Small-part Recall
Overlap Matrix Error
```

另设两张细表：

1. `5 task x 4 executor` mIoU matrix
2. cross-source generalization：3D AffordanceNet / PartNet-Mobility 互测

不要只报告 overall mIoU。Macro 指标必须作为主指标，避免大类别、大平面和高频执行器掩盖 hook、小部件、press 等困难样本。

## 10. 标注完成后实验顺序

当前最终输入不是旧 `reviewer_a_refined_samples.jsonl`，而是最终清洗后的 row-level JSONL：

```text
/home/lzq/data/MultiEEAffordance/processed/annotation_batches/final_5tasks/all_sources_5tasks_4exec_complete_aligned_posfixed.jsonl
```

实验顺序：

1. `validate_final_5task_rows.py`
2. `prepare_final_5task_training_dataset.py --split-unit source_asset`
3. `audit_splits.py --fail-on-leakage`
4. 可选：如果保留双人重叠原始记录，跑 `audit_annotation_consistency.py`
5. 跑 `pointnet_shared4_5tasks`
6. 跑 four single-EE baselines
7. 跑 token ablations
8. 汇总 `collect_experiment_table.py`
9. 再接入 PointNeXt / Point Transformer / HeteroAffordanceFormer

完整命令见：

```text
MultiEEAffordance/training/README.md
MultiEEAffordance/docs/最终五任务训练数据接入README.md
```
