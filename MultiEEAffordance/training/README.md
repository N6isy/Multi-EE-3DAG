# 五任务训练 Pipeline README

更新时间：2026-06-15

本文档说明如何把最终清洗后的五任务人工标注 JSONL 接入第一批 baseline 训练。当前训练侧只接受五个独立任务：

```text
lift / open / pull / press / push
```

四类执行器顺序固定为：

```text
gripper / suction / hook / dexterous_hand
```

mask shape 固定为 `[N, 4]`，通道顺序不能改。

## 1. 当前最终数据

最终训练样本文件位于数据服务器：

```text
dataset_root = /home/lzq/data/MultiEEAffordance
final_jsonl  = /home/lzq/data/MultiEEAffordance/processed/annotation_batches/final_5tasks/all_sources_5tasks_4exec_complete_aligned_posfixed.jsonl
```

这个 JSONL 的每一行是一个 `object_id + task + executor` 组合，不是一个完整物体，也不是一个完整 object-task。每个物体应该有：

```text
5 tasks * 4 executors = 20 rows
```

训练 dataloader 当前使用的是 object-task 格式，即一行对应：

```text
object_id + task -> [N,4] mask
```

因此训练前需要先执行 `prepare_final_5task_training_dataset.py`，把最终 JSONL 的 20 行/物体压缩成 5 行/物体。

## 2. 每一步做什么

训练前准备分为四步：

| 步骤 | 脚本 | 目的 | 主要输出 |
| --- | --- | --- | --- |
| 1 | `validate_final_5task_rows.py` | 检查最终 JSONL 每一行是否合法 | `final_rows_validation.json` |
| 2 | `prepare_final_5task_training_dataset.py` | 合并成训练 manifest，并按 CAD asset split | `manifests/train.jsonl` 等 |
| 3 | `audit_splits.py` | 检查 train/val/test 是否有 object/asset 泄漏 | `split_audit.json` |
| 4 | `train.py` / `evaluate.py` | 训练和评估第一批 baseline | `best.pt`、`test_metrics.json` |

旧脚本 `prepare_training_dataset.py` 仍然保留，用于早期 `reviewer_a_refined_samples.jsonl`、`reviewer_b_refined_samples.jsonl` 格式。当前最终数据不要再走这个旧入口。

## 3. 环境准备

在服务器上执行：

```bash
cd /home/lzq/data
conda activate multiee-train
```

当前 `10.24.1.11` 采用代码和数据同目录的布局：

```text
Python 包父目录：/home/lzq/data
数据根目录：/home/lzq/data/MultiEEAffordance
训练入口：/home/lzq/data/MultiEEAffordance/training/train.py
```

也就是说，执行 `python -m MultiEEAffordance.training.train` 时，应该从 `/home/lzq/data` 这个父目录启动。`MultiEEAffordance` 目录本身既是 Python package，也是 dataset root。

如果还没有训练环境：

```bash
conda create -n multiee-train python=3.11 -y
conda activate multiee-train

python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r MultiEEAffordance/training/requirements-training.txt
```

如果 `10.24.1.11` 没有 GPU，可以在该服务器上完成数据准备，再把 `/home/lzq/data/MultiEEAffordance` 挂载或同步到 GPU 服务器训练。配置文件里的 `dataset_root` 默认就是 `/home/lzq/data/MultiEEAffordance`。

如果需要把训练过程记录到 Weights & Biases，先在服务器登录一次：

```bash
wandb login
```

如果服务器不能直接联网，可以先用离线模式跑：

```bash
export WANDB_MODE=offline
```

训练结束后再按 wandb 提示执行 `wandb sync` 上传离线日志。

## 4. Step 1：校验最终 JSONL

当前在做什么：检查每一行是否满足训练前提，包括五任务合法性、执行器合法性、点云和 mask 是否存在、`mask.shape == [N,4]`、`positive_points_after` 是否等于对应 executor 通道正点数。

输入：

```text
processed/annotation_batches/final_5tasks/all_sources_5tasks_4exec_complete_aligned_posfixed.jsonl
```

命令：

```bash
python -m MultiEEAffordance.training.validate_final_5task_rows \
  --dataset-root /home/lzq/data/MultiEEAffordance \
  --final-samples processed/annotation_batches/final_5tasks/all_sources_5tasks_4exec_complete_aligned_posfixed.jsonl \
  --output-json processed/training/v0_4_final_5tasks/final_rows_validation.json \
  --fail-on-error \
  --overwrite
```

输出：

```text
/home/lzq/data/MultiEEAffordance/processed/training/v0_4_final_5tasks/final_rows_validation.json
```

必须确认：

```text
status = ok
error_count = 0
coverage.incomplete_objects = 0
coverage.duplicate_combination_count = 0
```

如果失败，不要训练。先看 `errors` 和 `coverage.missing_combinations`。

## 5. Step 2：生成训练 manifest

当前在做什么：把最终 JSONL 从 `object-task-executor` 行压缩成 `object-task` 行。每条训练行保留完整 `[N,4]` mask，并写入 `feasibility`、`positive_points`、`source_asset_id`、`asset_uid`、`split_key`。

输入：

```text
processed/annotation_batches/final_5tasks/all_sources_5tasks_4exec_complete_aligned_posfixed.jsonl
```

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

脚本会为每个 `object_id + task` 写出一个 canonical `[N,4]` mask：

```text
processed/training/v0_4_final_5tasks/masks/<object_id>_<task>.npy
```

原因是最终 JSONL 可能每个 executor 行都有自己的 refined mask 路径。训练需要四个 executor 通道在同一个 `[N,4]` 文件里，所以 prepare 阶段会逐行读取对应 executor 通道并合并。`--copy-masks` 仅作为兼容参数保留；当前真实数据形态下即使不传这个参数，也会自动写出合并后的 canonical mask。

输出：

```text
/home/lzq/data/MultiEEAffordance/processed/training/v0_4_final_5tasks/
  masks/
    <object_id>_<task>.npy
  manifests/
    all.jsonl
    train.jsonl
    val.jsonl
    test.jsonl
  summary.json
  final_rows_validation.json
  validation_errors.json
  object_task_coverage.json
  object_task_conflicts.json
  split_assignments.json
```

每条 manifest 行对应一个 `object_id + task`，关键字段包括：

```text
point_cloud_path
multi_channel_mask_path
executor_order
channel_supervision = [1,1,1,1]
feasibility
positive_points
source_asset_id
asset_uid
split_key
split
```

划分规则：

```text
优先按 split_key / asset_uid / source_asset_id
如果 3D AffordanceNet 只有 object_id=3danet_full_xxx，则 source_asset_id=object_id
同一个 CAD asset 派生的 20 行必须进入同一个 split
```

## 6. Step 3：审计 split

当前在做什么：确认 train/val/test 之间没有同一物体或同一 CAD asset 泄漏，并检查 task、executor、category、empty mask 分布。

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
/home/lzq/data/MultiEEAffordance/processed/training/v0_4_final_5tasks/split_audit.json
```

必须确认：

```text
asset_leakage = {}
object_leakage = {}
missing_fields = []
```

如果有 leakage，不能开始训练。

## 7. Step 4：训练第一批 baseline

当前在做什么：先训练轻量 PointNet 风格 baseline，验证数据、loss、metric 和训练闭环是否稳定。

配置文件：

```text
MultiEEAffordance/training/configs/pointnet_shared4_5tasks.json
```

该配置已经指向：

```text
processed/training/v0_4_final_5tasks/manifests/train.jsonl
processed/training/v0_4_final_5tasks/manifests/val.jsonl
processed/training/v0_4_final_5tasks/manifests/test.jsonl
```

命令：

```bash
CUDA_VISIBLE_DEVICES=0 python -m MultiEEAffordance.training.train \
  --config MultiEEAffordance/training/configs/pointnet_shared4_5tasks.json
```

所有可运行训练配置都默认打开 wandb，包括 shared-4 baseline、single-EE baseline 和 executor-token/relation-loss 消融。每个配置会用自己的 `experiment_name` 作为 wandb run name，记录每个 epoch 的 `train/*`、`val/*`、`learning_rate` 和 `best_macro_iou`。如果这次训练临时不想连接 wandb，可以不改配置，直接加 `--no-wandb`：

```bash
CUDA_VISIBLE_DEVICES=0 python -m MultiEEAffordance.training.train \
  --config MultiEEAffordance/training/configs/pointnet_shared4_5tasks.json \
  --no-wandb
```

输出：

```text
/home/lzq/data/MultiEEAffordance/processed/training_runs/pointnet_shared4_5tasks/
  latest.pt
  best.pt
  history.json
  resolved_config.json
```

第一批建议先跑：

```text
pointnet_shared4_5tasks.json
single_ee_gripper_5tasks.json
single_ee_suction_5tasks.json
single_ee_hook_5tasks.json
single_ee_dexterous_hand_5tasks.json
```

这样可以先回答一个关键审稿问题：联合四执行器预测是否优于四个独立 single-EE 模型。

## 8. Step 5：评估和汇总

评估单个模型：

```bash
python -m MultiEEAffordance.training.evaluate \
  --config MultiEEAffordance/training/configs/pointnet_shared4_5tasks.json \
  --checkpoint /home/lzq/data/MultiEEAffordance/processed/training_runs/pointnet_shared4_5tasks/best.pt \
  --output-json processed/training_runs/pointnet_shared4_5tasks/test_metrics.json
```

汇总实验表：

```bash
python -m MultiEEAffordance.training.collect_experiment_table \
  --runs-root /home/lzq/data/MultiEEAffordance/processed/training_runs \
  --output-csv /home/lzq/data/MultiEEAffordance/processed/training/v0_4_final_5tasks/aaai_main_table.csv \
  --output-json /home/lzq/data/MultiEEAffordance/processed/training/v0_4_final_5tasks/aaai_main_table.json
```

论文主表不能只报告 overall mIoU。至少需要关注：

```text
macro_iou
macro_dice
per_task_iou
per_executor_iou
task_executor_iou_5x4
macro_feasibility_f1
feasibility_auroc
empty_mask_accuracy
non_empty_recall
small_part_recall
executor_overlap_matrix_error
```

## 9. 可直接运行的配置

当前已经实现的模型是轻量 `TaskExecutorPointNet`，用于建立第一批可运行 baseline 和消融：

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

PointNeXt、Point Transformer、PointNet++、真正 four-head model 和 HeteroAffordanceFormer 主模型还没有在当前代码中实现。不要把当前 PointNet 结果命名成这些强 baseline。

## 10. 本地 smoke test

不依赖 PyTorch 的最终 JSONL 准备测试：

```bash
python -m MultiEEAffordance.training.smoke_prepare_final_dataset
```

旧 refined-samples 入口测试：

```bash
python -m MultiEEAffordance.training.smoke_prepare_dataset
```

依赖 PyTorch 的训练闭环测试：

```bash
python -m MultiEEAffordance.training.smoke_test
```

如果本地没有安装 torch，第三个测试会失败，这是环境问题；服务器训练环境需要通过。

## 11. Git 和大文件约束

Git 只提交代码、配置、文档和小型元数据。不要提交：

```text
.npy
.npz
.pt
大规模 processed 数据目录
```

项目要求 Git 提交说明使用中文。

## 12. 常见训练启动错误

### JSONDecodeError: Unexpected UTF-8 BOM

如果训练启动时报错：

```text
json.decoder.JSONDecodeError: Unexpected UTF-8 BOM
```

含义是 JSON 配置文件开头带有 UTF-8 BOM。当前训练代码已经使用 `utf-8-sig` 读取配置，正常同步最新代码后可以直接重新运行训练命令。

如果同步代码后仍然报同样错误，优先检查 Python 实际导入的是不是仓库代码，而不是数据目录里的旧副本：

```bash
cd /home/lzq/data
export PYTHONPATH=/home/lzq/data:$PYTHONPATH
python -c "import MultiEEAffordance.training.train as t; print(t.__file__)"
```

期望输出类似：

```text
/home/lzq/data/MultiEEAffordance/training/train.py
```

如果输出不是这个路径，例如仍然指向其他旧仓库目录：

```text
/home/lzq/Multi-EE-3DAG/MultiEEAffordance/training/train.py
```

说明当前运行到了另一份代码。需要把最新代码同步到 `/home/lzq/data/MultiEEAffordance`，或者明确切换到你希望使用的那份代码后再运行。
