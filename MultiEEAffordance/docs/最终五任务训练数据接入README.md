# 最终五任务训练数据接入 README

更新时间：2026-06-15

本文档面向维护者，说明数据标注完成后，如何把最终清洗 JSONL 接入第一批 baseline 训练。

## 1. 当前最终输入

数据根目录：

```text
/home/lzq/data/MultiEEAffordance
```

最终训练样本 JSONL：

```text
/home/lzq/data/MultiEEAffordance/processed/annotation_batches/final_5tasks/all_sources_5tasks_4exec_complete_aligned_posfixed.jsonl
```

这个文件已经是最终清洗后的人工标注结果。每一行是一个：

```text
object_id + task + executor
```

不是一个完整 object，也不是一个完整 object-task。每个物体应有 20 行：

```text
5 tasks * 4 executors
```

固定任务：

```text
lift / open / pull / press / push
```

固定执行器：

```text
gripper / suction / hook / dexterous_hand
```

mask 四通道顺序：

```text
mask[:, 0] = gripper
mask[:, 1] = suction
mask[:, 2] = hook
mask[:, 3] = dexterous_hand
```

## 2. 为什么还需要 prepare

训练 dataloader 读取的是：

```text
point cloud + task -> [N,4] mask
```

也就是一条训练样本包含同一物体、同一任务下四个执行器的 mask。

但最终 JSONL 是：

```text
point cloud + task + executor -> one executor channel supervision
```

因此需要把每个 `object_id + task` 下的四条 executor 行合并成一条训练行。这个操作不会改动原始点云和 mask，只会生成训练用 manifest。

## 3. 服务器执行位置

推荐在数据服务器执行：

```text
server: 10.24.1.11
python package parent: /home/lzq/data
dataset root:          /home/lzq/data/MultiEEAffordance
training entry:        /home/lzq/data/MultiEEAffordance/training/train.py
```

开始前：

```bash
cd /home/lzq/data
conda activate multiee-train
export PYTHONPATH=/home/lzq/data:$PYTHONPATH
python -c "import MultiEEAffordance.training.train as t; print(t.__file__)"
```

期望打印：

```text
/home/lzq/data/MultiEEAffordance/training/train.py
```

如果后续训练放在其他 GPU 服务器，也应先在 `10.24.1.11` 完成数据校验和 manifest 生成，再同步 `/home/lzq/data/MultiEEAffordance`。

如果要连接 wandb，先登录：

```bash
wandb login
```

不能联网时先用离线模式：

```bash
export WANDB_MODE=offline
```

## 4. 第一步：校验最终 JSONL

当前在做什么：逐行读取最终 JSONL，检查每行的 task、executor、路径、点云 shape、mask shape、正点数量是否正确。

命令：

```bash
python -m MultiEEAffordance.training.validate_final_5task_rows \
  --dataset-root /home/lzq/data/MultiEEAffordance \
  --final-samples processed/annotation_batches/final_5tasks/all_sources_5tasks_4exec_complete_aligned_posfixed.jsonl \
  --output-json processed/training/v0_4_final_5tasks/final_rows_validation.json \
  --fail-on-error \
  --overwrite
```

输入：

```text
processed/annotation_batches/final_5tasks/all_sources_5tasks_4exec_complete_aligned_posfixed.jsonl
```

输出：

```text
processed/training/v0_4_final_5tasks/final_rows_validation.json
```

必须确认：

```text
status = ok
error_count = 0
coverage.incomplete_objects = 0
coverage.duplicate_combination_count = 0
```

如果失败，先修数据，不要继续训练。

## 5. 第二步：生成训练 manifest

当前在做什么：把最终 JSONL 的 `object-task-executor` 行合并成训练用 `object-task` 行，并按 CAD asset 做 train/val/test split。

命令：

```bash
python -m MultiEEAffordance.training.prepare_final_5task_training_dataset \
  --dataset-root /home/lzq/data/MultiEEAffordance \
  --final-samples processed/annotation_batches/final_5tasks/all_sources_5tasks_4exec_complete_aligned_posfixed.jsonl \
  --output-root processed/training/v0_4_final_5tasks \
  --dataset-version v0_4_final_5tasks \
  --split-unit source_asset \
  --overwrite
```

输出目录：

```text
processed/training/v0_4_final_5tasks/
  manifests/
    all.jsonl
    train.jsonl
    val.jsonl
    test.jsonl
  summary.json
  validation_errors.json
  object_task_coverage.json
  object_task_conflicts.json
  split_assignments.json
```

脚本会写出合并后的 canonical `[N,4]` mask：

```text
processed/training/v0_4_final_5tasks/masks/<object_id>_<task>.npy
```

原因是最终 JSONL 中同一个 `object_id + task` 的四个 executor 行可能分别引用四个不同的 refined mask 文件。训练 manifest 需要一个统一的 `[N,4]` mask，因此 prepare 阶段会分别读取每行对应 executor 通道，再合并成一个训练 mask。`--copy-masks` 仅作为兼容参数保留；当前真实数据形态下即使不传这个参数，也会自动写出 canonical mask。

## 6. 第三步：审计 split

当前在做什么：检查同一个 CAD asset 是否被拆到了多个 split。3D AffordanceNet 中，如果只有 `object_id=3danet_full_xxx`，就把这个 `object_id` 当成 `source_asset_id`。

命令：

```bash
python -m MultiEEAffordance.training.audit_splits \
  --dataset-root /home/lzq/data/MultiEEAffordance \
  --manifest processed/training/v0_4_final_5tasks/manifests/all.jsonl \
  --output-json processed/training/v0_4_final_5tasks/split_audit.json \
  --fail-on-leakage
```

输出：

```text
processed/training/v0_4_final_5tasks/split_audit.json
```

必须确认：

```text
asset_leakage = {}
object_leakage = {}
missing_fields = []
```

## 7. 第四步：跑第一批 baseline

当前先跑轻量 PointNet baseline，目的是验证数据和训练闭环。

命令：

```bash
CUDA_VISIBLE_DEVICES=0 python -m MultiEEAffordance.training.train \
  --config MultiEEAffordance/training/configs/pointnet_shared4_5tasks.json
```

所有可运行训练配置都默认打开 wandb，项目名为 `multiee-affordance`，run 名与各自的 `experiment_name` 对齐。比如 `pointnet_shared4_5tasks.json` 的 run 名为 `pointnet_shared4_5tasks`，single-EE 和消融配置也会各自生成独立 run。如果临时关闭：

```bash
CUDA_VISIBLE_DEVICES=0 python -m MultiEEAffordance.training.train \
  --config MultiEEAffordance/training/configs/pointnet_shared4_5tasks.json \
  --no-wandb
```

输出：

```text
processed/training_runs/pointnet_shared4_5tasks/
  latest.pt
  best.pt
  history.json
  resolved_config.json
```

建议第一批 baseline 顺序：

```text
1. pointnet_shared4_5tasks
2. single_ee_gripper_5tasks
3. single_ee_suction_5tasks
4. single_ee_hook_5tasks
5. single_ee_dexterous_hand_5tasks
6. executor token ablations
```

这样可以先建立论文中最基本的公平对照：联合四执行器预测 vs 四个独立 single-EE 模型。

## 8. 第五步：评估和汇总表格

评估：

```bash
python -m MultiEEAffordance.training.evaluate \
  --config MultiEEAffordance/training/configs/pointnet_shared4_5tasks.json \
  --checkpoint /home/lzq/data/MultiEEAffordance/processed/training_runs/pointnet_shared4_5tasks/best.pt \
  --output-json processed/training_runs/pointnet_shared4_5tasks/test_metrics.json
```

汇总：

```bash
python -m MultiEEAffordance.training.collect_experiment_table \
  --runs-root /home/lzq/data/MultiEEAffordance/processed/training_runs \
  --output-csv /home/lzq/data/MultiEEAffordance/processed/training/v0_4_final_5tasks/aaai_main_table.csv \
  --output-json /home/lzq/data/MultiEEAffordance/processed/training/v0_4_final_5tasks/aaai_main_table.json
```

## 9. 本地代码检查

本地修改后先跑：

```bash
python -m compileall -q MultiEEAffordance/training
python -m MultiEEAffordance.training.smoke_prepare_final_dataset
```

如果本地没有 torch，`smoke_test` 可以留到服务器训练环境跑：

```bash
python -m MultiEEAffordance.training.smoke_test
```

## 10. 关键风险

1. 不能按 JSONL 行随机 split。
   同一个物体有 20 行，如果随机按行切分，会造成严重数据泄漏。

2. 不能把旧任务候选当成训练 GT。
   当前最终 JSONL 才是训练输入。

3. 不能改变 executor channel 顺序。
   顺序固定为 `gripper/suction/hook/dexterous_hand`。

4. 不能只看 overall mIoU。
   论文实验必须报告 per-task、per-executor、task-executor matrix、feasibility 和 empty-mask 指标。
