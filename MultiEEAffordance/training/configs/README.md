# Training Configs

当前目录分为两类配置。所有可运行配置默认读取最终五任务训练 manifest：

```text
/home/lzq/data/MultiEEAffordance/processed/training/v0_4_final_5tasks/manifests/
```

该 manifest 由最终清洗 JSONL 生成：

```text
processed/annotation_batches/final_5tasks/all_sources_5tasks_4exec_complete_aligned_posfixed.jsonl
```

生成和审计命令见：

```text
MultiEEAffordance/training/README.md
MultiEEAffordance/docs/最终五任务训练数据接入README.md
```

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

所有可运行训练配置都默认启用 wandb。每个配置的 `wandb.name` 与 `experiment_name` 对齐，避免主实验、single-EE baseline 和 executor-token 消融混在同一个 run 里。统一配置结构如下：

```json
"wandb": {
  "enabled": true,
  "project": "multiee-affordance",
  "name": "<experiment_name>",
  "group": "v0_4_final_5tasks",
  "job_type": "train",
  "tags": ["baseline", "5tasks"],
  "mode": "online",
  "watch_model": false,
  "log_checkpoints": false
}
```

命令行可用 `--wandb` 强制开启，也可用 `--no-wandb` 临时关闭。正式跑实验时建议不要关闭 wandb；如果服务器不能联网，用 `WANDB_MODE=offline` 保留离线日志。

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
