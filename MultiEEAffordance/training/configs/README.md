# Training Configs

更新时间：2026-06-16

本目录存放五任务训练实验配置。训练任务固定为：

```text
lift / open / pull / press / push
```

执行器顺序固定为：

```text
gripper / suction / hook / dexterous_hand
```

mask 始终是 `[N,4]`。所有配置默认读取最终人工标注训练 manifest：

```text
/home/lzq/data/MultiEEAffordance/processed/training/v0_4_final_5tasks/manifests/
```

该 manifest 来自最终清洗样本：

```text
processed/annotation_batches/final_5tasks/all_sources_5tasks_4exec_complete_aligned_posfixed.jsonl
```

## 1. PointNet 基线

这些配置使用当前轻量 PointNet MLP backbone，适合建立第一批 baseline 和 executor-token 消融：

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

运行示例：

```bash
CUDA_VISIBLE_DEVICES=0 python -m MultiEEAffordance.training.train \
  --config MultiEEAffordance/training/configs/pointnet_shared4_5tasks.json
```

输出目录由配置里的 `output_dir` 决定，例如：

```text
/home/lzq/data/MultiEEAffordance/processed/training_runs/pointnet_shared4_5tasks/
```

## 2. PointNeXt 基线

当前已经新增 PointNeXt adapter：

```text
MultiEEAffordance/training/backbones/pointnext_adapter.py
```

可运行配置：

```text
pointnext_shared4_5tasks_debug.json
pointnext_shared4_5tasks.json
```

先跑 debug 配置，确认 OpenPoints 环境、CUDA extension、显存和输出 shape：

```bash
CUDA_VISIBLE_DEVICES=0 python -m MultiEEAffordance.training.train \
  --config MultiEEAffordance/training/configs/pointnext_shared4_5tasks_debug.json
```

debug 输出：

```text
/home/lzq/data/MultiEEAffordance/processed/training_runs/pointnext_shared4_5tasks_debug/
```

debug 通过后再跑正式配置：

```bash
CUDA_VISIBLE_DEVICES=0 python -m MultiEEAffordance.training.train \
  --config MultiEEAffordance/training/configs/pointnext_shared4_5tasks.json
```

正式输出：

```text
/home/lzq/data/MultiEEAffordance/processed/training_runs/pointnext_shared4_5tasks/
```

## 3. 执行器属性条件建模配置

执行器属性文件：

```text
MultiEEAffordance/training/configs/executor_specs_5tasks.json
```

当前属性只描述执行器本身的接触机制和几何偏好，不包含任务标签，也不读取 ground truth mask，避免形成标签泄漏。

可运行配置：

```text
pointnet_attr_only_5tasks.json
pointnet_id_attr_5tasks.json
pointnet_id_attr_film_5tasks.json
pointnext_id_attr_film_5tasks.json
pointnext_id_attr_crossattn_5tasks.json
```

建议先跑：

```bash
CUDA_VISIBLE_DEVICES=0 python -m MultiEEAffordance.training.train \
  --config MultiEEAffordance/training/configs/pointnet_id_attr_film_5tasks.json
```

再跑 PointNeXt + 属性条件：

```bash
CUDA_VISIBLE_DEVICES=0 python -m MultiEEAffordance.training.train \
  --config MultiEEAffordance/training/configs/pointnext_id_attr_film_5tasks.json
```

## 4. 推荐实验顺序

1. `pointnet_shared4_5tasks`
2. `single_ee_*_5tasks`
3. `pointnet_one_hot_executor_token_5tasks`
4. `pointnet_id_attr_film_5tasks`
5. `pointnext_shared4_5tasks_debug`
6. `pointnext_shared4_5tasks`
7. `pointnext_id_attr_film_5tasks`
8. `pointnext_id_attr_crossattn_5tasks`

不要先跑 cross-attention 版本。先确认 PointNeXt shared-4 和 FiLM 版本稳定，再把 cross-attention 作为增强消融。

## 5. 评估命令

普通评估：

```bash
python -m MultiEEAffordance.training.evaluate \
  --config MultiEEAffordance/training/configs/pointnext_shared4_5tasks.json \
  --checkpoint /home/lzq/data/MultiEEAffordance/processed/training_runs/pointnext_shared4_5tasks/best.pt \
  --dataset-root /home/lzq/data/MultiEEAffordance \
  --output-json /home/lzq/data/MultiEEAffordance/processed/training_runs/pointnext_shared4_5tasks/test_metrics.json
```

阈值校准 + feasibility gate：

```bash
python -m MultiEEAffordance.training.evaluate_calibrated \
  --config MultiEEAffordance/training/configs/pointnext_shared4_5tasks.json \
  --checkpoint /home/lzq/data/MultiEEAffordance/processed/training_runs/pointnext_shared4_5tasks/best.pt \
  --dataset-root /home/lzq/data/MultiEEAffordance \
  --output-json /home/lzq/data/MultiEEAffordance/processed/training_runs/pointnext_shared4_5tasks/test_metrics_calibrated_gated.json \
  --thresholds-json /home/lzq/data/MultiEEAffordance/processed/training_runs/pointnext_shared4_5tasks/thresholds_val.json \
  --feasibility-gate
```

输出包括：

```text
test_metrics.json
test_metrics_calibrated_gated.json
test_metrics_calibrated_gated_task_executor_matrix.csv
thresholds_val.json
```

## 6. 公平性注意事项

对比 PointNet、PointNeXt 和属性条件版本时，必须保持：

- 同一份 train/val/test manifest；
- 同一批最终人工标注数据；
- 同一任务和执行器定义；
- 同一 mask shape `[N,4]`；
- 同一评估脚本和阈值校准策略；
- 清楚记录 batch size、epoch、学习率和参数量。

如果 PointNeXt 结果明显好于 PointNet，需要进一步说明提升来自 backbone 表达能力，而不是数据划分、阈值策略或训练步数不一致。
