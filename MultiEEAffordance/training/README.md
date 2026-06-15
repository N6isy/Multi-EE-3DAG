# 五任务训练实验 Pipeline README

更新时间：2026-06-09

本文档说明标注完成后如何把人工审查结果整理成 AAAI 实验可用的训练集，并启动第一批 baseline。训练侧只接受五个独立任务：

```text
lift / open / pull / press / push
```

旧任务 `pick_up/open_pull/press_push/lift_carry` 只能出现在候选生成和人工审查准备阶段。旧候选不是五任务真值；必须经过五任务人工审查后，才能进入本训练目录。

## 1. 数据服务器和执行位置

当前数据存储服务器：

```text
server: 10.24.1.11
code:   /home/lzq/Multi-EE-3DAG
data:   /home/lzq/data/MultiEEAffordance
```

建议在 `10.24.1.11` 上执行训练数据准备、split 审计、人工一致性审计和实验表格汇总，因为这些步骤会读写大量点云和 mask 文件。

GPU 训练也可以放在 `10.24.1.11`，但需要先确认：

```bash
ssh lzq@10.24.1.11
nvidia-smi

python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY
```

如果该服务器没有可用 GPU，就在 `10.24.1.11` 生成训练 manifest，然后把 `/home/lzq/data/MultiEEAffordance` 挂载或同步到 GPU 服务器训练。

## 2. 训练数据契约

训练输入不是 VLM 输出，也不是旧候选目录，而是网页人工审查后保存的 refined samples：

```text
/home/lzq/data/MultiEEAffordance/processed/annotation_batches/v0_2_5tasks/
  reviewer_a_refined_samples.jsonl
  reviewer_b_refined_samples.jsonl
  manual_refined_masks_reviewer_a/
  manual_refined_masks_reviewer_b/
```

每条 refined sample 至少需要：

```text
object_id
object_category
source_dataset
task
executor
point_cloud_path
multi_channel_mask_path
reviewer
```

训练准备脚本会补齐：

```text
source_asset_id
asset_uid
split_key
split_unit
annotation_source=human_review
task_taxonomy_version=v0_2_5tasks
executor_order=[gripper,suction,hook,dexterous_hand]
```

3D AffordanceNet 的 CAD asset 规则：

```text
如果只有 object_id = 3danet_full_xxx
则 source_asset_id = object_id
asset_uid = source_dataset + ":" + source_asset_id
```

同一个 `asset_uid` 派生出的所有 task、executor、empty mask、重复审查样本必须进入同一个 split。

## 3. 安装训练环境

```bash
cd /home/lzq/Multi-EE-3DAG
conda create -n multiee-train python=3.11 -y
conda activate multiee-train

python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r MultiEEAffordance/training/requirements-training.txt
```

如果服务器已有可用 PyTorch 环境，可以复用，但要保证能从 `/home/lzq/data/MultiEEAffordance` 读取数据。

## 4. 标注完成后的完整操作

以下命令默认在 `10.24.1.11` 上执行。

### Step 1：确认人工标注输出存在

当前在做什么：确认两位审查者的 refined samples 已经汇总到数据目录。

```bash
cd /home/lzq/Multi-EE-3DAG
conda activate multiee-train

ls /home/lzq/data/MultiEEAffordance/processed/annotation_batches/v0_2_5tasks/*_refined_samples.jsonl
```

预期至少看到：

```text
reviewer_a_refined_samples.jsonl
reviewer_b_refined_samples.jsonl
```

### Step 2：检查 refined samples

当前在做什么：检查五任务字段是否合法、reviewer 是否为空、点云和 mask 文件是否存在、mask shape 是否能和点云对齐。

```bash
python -m MultiEEAffordance.training.validate_reviewed_samples \
  --dataset-root /home/lzq/data/MultiEEAffordance \
  --reviewed-samples processed/annotation_batches/v0_2_5tasks/reviewer_a_refined_samples.jsonl,processed/annotation_batches/v0_2_5tasks/reviewer_b_refined_samples.jsonl \
  --output-json processed/training/v0_3_human_5tasks/reviewed_samples_validation.json
```

如果输出 `status=failed`，先修正 `errors` 中列出的样本，不要继续生成训练集。

### Step 3：生成 canonical 训练集

当前在做什么：把网页保存的 `object + task + executor` 单通道审查结果，合并成模型训练需要的 `object + task -> [N,4]` mask。

```bash
python -m MultiEEAffordance.training.prepare_training_dataset \
  --dataset-root /home/lzq/data/MultiEEAffordance \
  --reviewed-samples processed/annotation_batches/v0_2_5tasks/reviewer_a_refined_samples.jsonl,processed/annotation_batches/v0_2_5tasks/reviewer_b_refined_samples.jsonl \
  --output-root processed/training/v0_3_human_5tasks \
  --dataset-version v0_3_human_5tasks \
  --split-unit source_asset \
  --min-reviewed-channels 4 \
  --overwrite
```

输出目录：

```text
/home/lzq/data/MultiEEAffordance/processed/training/v0_3_human_5tasks/
  masks/
    <object_id>_<task>.npy
  manifests/
    all.jsonl
    train.jsonl
    val.jsonl
    test.jsonl
    rejected_rows.json
    incomplete_rows.json
    conflict_rows.json
  summary.json
```

其中 `rejected_rows.json`、`incomplete_rows.json`、`conflict_rows.json` 必须人工看一遍。正式实验建议 `--min-reviewed-channels 4`，确保每个 object-task 都有四个执行器通道的人工结论。

### Step 4：审计 split

当前在做什么：检查同一个 CAD asset 是否泄漏到多个 split，并检查 task/executor/category/empty mask 分布。

```bash
python -m MultiEEAffordance.training.audit_splits \
  --dataset-root /home/lzq/data/MultiEEAffordance \
  --manifest processed/training/v0_3_human_5tasks/manifests/all.jsonl \
  --output-json processed/training/v0_3_human_5tasks/split_audit.json \
  --fail-on-leakage
```

必须确认：

```text
asset_leakage = {}
object_leakage = {}
missing_fields = []
```

如果有 leakage，说明 split 规则或 source asset 字段有问题，不能开始训练。

### Step 5：审计双人一致性

当前在做什么：对两位审查者都标过的 calibration subset 计算 point-level IoU、feasibility agreement 和 empty-mask agreement。

```bash
python -m MultiEEAffordance.training.audit_annotation_consistency \
  --dataset-root /home/lzq/data/MultiEEAffordance \
  --reviewed-samples processed/annotation_batches/v0_2_5tasks/reviewer_a_refined_samples.jsonl,processed/annotation_batches/v0_2_5tasks/reviewer_b_refined_samples.jsonl \
  --output-json processed/training/v0_3_human_5tasks/annotation_consistency.json \
  --output-csv processed/training/v0_3_human_5tasks/annotation_disagreements.csv
```

如果 `reviewer_pairs=0`，说明没有双人重叠样本。建议补一个最小 consistency audit subset：至少 50 个 CAD asset，由两位审查者都标一遍，用于论文 appendix 的人工一致性统计。

### Step 6：训练第一版 baseline

当前在做什么：先训练可运行的 PointNet 风格轻量 baseline，用来验证数据闭环、loss、metrics 和 split 是否稳定。

```bash
CUDA_VISIBLE_DEVICES=0 python -m MultiEEAffordance.training.train \
  --config MultiEEAffordance/training/configs/pointnet_shared4_5tasks.json
```

输出：

```text
/home/lzq/data/MultiEEAffordance/processed/training_runs/pointnet_shared4_5tasks/
  latest.pt
  best.pt
  history.json
  resolved_config.json
```

### Step 7：评估并汇总论文表格

```bash
python -m MultiEEAffordance.training.evaluate \
  --config MultiEEAffordance/training/configs/pointnet_shared4_5tasks.json \
  --checkpoint /home/lzq/data/MultiEEAffordance/processed/training_runs/pointnet_shared4_5tasks/best.pt \
  --output-json processed/training_runs/pointnet_shared4_5tasks/test_metrics.json

python -m MultiEEAffordance.training.collect_experiment_table \
  --runs-root /home/lzq/data/MultiEEAffordance/processed/training_runs \
  --output-csv /home/lzq/data/MultiEEAffordance/processed/training/v0_3_human_5tasks/aaai_main_table.csv \
  --output-json /home/lzq/data/MultiEEAffordance/processed/training/v0_3_human_5tasks/aaai_main_table.json
```

## 5. 当前可运行配置

当前代码实现的是轻量 `TaskExecutorPointNet`，用于验证训练闭环。已经提供：

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

注意：PointNeXt、Point Transformer、真正 four-head model 和 HeteroAffordanceFormer 主模型还没有在当前代码中实现。不要用当前 PointNet 配置冒充这些强 baseline。后续实现新 backbone 后，再补对应配置和主表。

配置目录的可运行/待实现边界见：

```text
MultiEEAffordance/training/configs/README.md
```

## 6. Loss 和指标定义

训练 loss：

```text
L = L_mask_positive + lambda_empty * L_empty_area + lambda_feas * L_feasibility + lambda_relation * L_relation
```

规则：

- `F_gt=1` 的通道计算 Focal BCE + Dice。
- `F_gt=0` 的通道不计算 Dice，只计算 empty area penalty 和 feasibility BCE。
- relation loss 默认关闭；只作为 ablation，且只在双方 feasible 且 mask 足够大时计算。

评估指标包括：

```text
macro_iou
macro_dice
task_executor_iou 5x4
per_task
per_executor
macro_feasibility_f1
feasibility_auroc
empty_mask_accuracy
small_part_recall
executor_overlap_matrix_error
```

主表不要只报告 overall mIoU。论文主结果至少报告 macro mIoU、macro Dice、5x4 task-executor matrix、feasibility F1/AUROC、empty-mask 指标和 small-part recall。

## 7. 本地 smoke test

不依赖 PyTorch 的数据准备测试：

```bash
python -m MultiEEAffordance.training.smoke_prepare_dataset
```

依赖 PyTorch 的训练闭环测试：

```bash
python -m MultiEEAffordance.training.smoke_test
```

正常输出应包含：

```text
"status": "ok"
"legacy_task_rejected": true
```

## 8. Git 说明

本项目要求 Git 提交说明使用中文。大规模 `.npy/.npz/.pt` 不进入普通 Git 仓库，只保存在 `/home/lzq/data/MultiEEAffordance` 或实验服务器的数据目录。
