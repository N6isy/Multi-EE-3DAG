# Training Configs

当前目录分为两类配置：

## 可直接运行

这些配置使用当前已实现的 `TaskExecutorPointNet`：

```text
pointnet_shared4_5tasks.json
pointnet_no_executor_token_5tasks.json
pointnet_shared_executor_token_5tasks.json
pointnet_one_hot_executor_token_5tasks.json
pointnet_random_frozen_executor_token_5tasks.json
pointnet_executor_token_swap_5tasks.json
pointnet_with_relation_loss_5tasks.json
single_ee_gripper_5tasks.json
single_ee_suction_5tasks.json
single_ee_hook_5tasks.json
single_ee_dexterous_hand_5tasks.json
```

## 不能用当前模型冒充的强 baseline

以下 AAAI 主实验需要等对应 backbone/model class 实现后再新增正式 JSON：

```text
pointnetpp_shared4_5tasks
pointnext_shared4_5tasks
pointnext_four_heads_5tasks
pointtransformer_shared4_5tasks
heteroaffordanceformer_5tasks
geometry_only_heuristic_5tasks
```

不要把 `TaskExecutorPointNet` 的结果命名成 PointNeXt、Point Transformer 或 HeteroAffordanceFormer。这样会污染实验记录，也无法经得住论文审查。
