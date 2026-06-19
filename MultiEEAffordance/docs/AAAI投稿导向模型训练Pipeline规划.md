# AAAI 投稿导向模型训练 Pipeline 规划

> 文档状态：当前主线
> 当前主线：五任务人工标注完成，训练进入 PointNeXt + 执行器属性条件建模阶段。
> 最新入口：docs/README.md、training/README.md、五任务训练初版结果进度汇报.md。

更新时间：2026-06-16

## 主实验顺序

第一阶段：确认数据和评估闭环。

```text
PointNet shared-4
PointNet single-EE ensemble
PointNet executor-token 消融
PointNet relation-loss 消融
```

第二阶段：接入更强 3D backbone。

```text
PointNeXt shared-4
PointNeXt one-hot executor token
PointNeXt relation-loss
```

第三阶段：执行器属性条件建模。

```text
PointNeXt attr_only
PointNeXt id_attr
PointNeXt id_attr_film
PointNeXt id_attr_crossattn
```

第四阶段：论文主模型或更强结构。

```text
PointTransformer
Heterogeneous executor-aware model
new-executor generalization analysis
```

## 主表建议

主表不要只报告 overall mIoU。建议至少包含：

```text
macro_iou
macro_dice
macro_feasibility_f1
empty_mask_accuracy
small_part_recall
executor_overlap_matrix_error
```

附表展示：

```text
5 x 4 task-executor IoU matrix
per-executor Dice
per-task IoU
calibrated/gated evaluation
```

## 论文叙事

推荐叙事路线：

1. Multi-EE affordance 不只是多类别分割，而是同一任务下多个异构执行器的可行性和接触区域联合预测。
2. PointNet baseline 证明数据闭环可训练，但几何表达能力有限。
3. PointNeXt 提供更强局部几何建模。
4. 执行器属性条件建模把机械机制显式引入模型，提升可解释性，并为未来新执行器扩展留出口。

## 风险控制

不能把 PointNet 结果命名成 PointNeXt。

不能在 OpenPoints CUDA extension 未编译时报告 PointNeXt 训练结果。

不能只用一次 debug run 作为方法有效性结论。

不能修改任务集合、executor 顺序或 mask shape 来追求指标。
