# 五任务训练 Pipeline README

更新时间：2026-06-01

## 1. 这个目录做什么

`MultiEEAffordance/training/` 是独立的模型训练目录。它和候选生成、VLM 辅助标注、网页审查工具分开维护。

训练侧只接受五个独立任务：

```text
lift
open
pull
press
push
```

训练代码不会读取、展开或解释历史复合任务。以下任务如果进入训练数据，会直接报错：

```text
pick_up
lift_carry
open_pull
press_push
```

历史候选只能先经过五任务展开和人工审查，再作为训练数据使用。

## 2. 能否直接在数据存储服务器训练

可以，但需要区分两类操作。

### 2.1 数据准备

数据准备应直接在数据存储服务器执行：

```text
服务器：10.24.1.11
代码目录：/home/lzq/Multi-EE-3DAG
数据根目录：/home/lzq/data/MultiEEAffordance
```

原因是点云、refined mask、训练专用 canonical mask 和 checkpoint 都比较大。把它们放在 `/home/lzq/data/MultiEEAffordance` 可以避免在服务器之间反复复制。

### 2.2 GPU 训练

GPU 训练也可以直接在 `10.24.1.11` 执行，但必须先确认该服务器具备 GPU 和 PyTorch 环境：

```bash
ssh lzq@10.24.1.11
nvidia-smi

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
PY
```

如果 `nvidia-smi` 不存在，或者 `torch.cuda.is_available()` 为 `False`，仍然可以在 `10.24.1.11` 生成训练 manifest，但正式训练需要改到 GPU 服务器，并挂载或同步 `/home/lzq/data/MultiEEAffordance`。

## 3. 初版目录结构

```text
MultiEEAffordance/training/
  README.md
  requirements-training.txt
  constants.py
  prepare_training_dataset.py
  dataset.py
  model.py
  losses.py
  metrics.py
  train.py
  evaluate.py
  smoke_test.py
  configs/
    baseline_task_executor_pointnet_5tasks.json
```

其中：

| 文件 | 用途 |
| --- | --- |
| `constants.py` | 训练专用任务和执行器定义。只包含五任务。 |
| `prepare_training_dataset.py` | 将网页人工审查结果合并成训练专用 `[N,4]` mask。 |
| `dataset.py` | 加载点云、mask、任务 ID 和通道监督信息。 |
| `model.py` | 初版轻量模型 `TaskExecutorPointNet`。 |
| `losses.py` | Focal BCE、Dice、feasibility 和 executor relation loss。 |
| `metrics.py` | 统计 mIoU、Dice 和 feasibility accuracy。 |
| `train.py` | 训练入口。 |
| `evaluate.py` | 测试集评估入口。 |
| `smoke_test.py` | 使用临时构造数据检查训练闭环。 |

## 4. 训练需要哪些数据

训练输入不是旧候选目录，也不是 VLM 输出。训练输入必须是网页人工审查后保存的五任务 refined samples。

正式输入示例：

```text
/home/lzq/data/MultiEEAffordance/
  processed/
    annotation_batches/
      v0_2_5tasks/
        reviewer_a_refined_samples.jsonl
        reviewer_b_refined_samples.jsonl
        manual_refined_masks_reviewer_a/
        manual_refined_masks_reviewer_b/
    points/
      ...
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
quality_flag
point_review_status
reviewer
```

其中：

- `task` 必须是五任务之一。
- `point_cloud_path` 指向 `[N,3]` 或 `[N,6]` 点云。
- `multi_channel_mask_path` 指向人工保存的 `[N,4]` mask。
- `executor` 表示本条网页审查实际确认的是哪个通道。
- 空 mask 也是有效标签，不能删除。

## 5. 为什么还需要生成训练专用 mask

网页审查通常每次保存一个 `object + task + executor` 通道。模型训练需要一次读取同一个 `object + task` 的四通道结果：

```text
P, q -> [M_gripper, M_suction, M_hook, M_dexterous_hand]
```

因此必须先运行 `prepare_training_dataset.py`：

1. 按 `object_id + task` 聚合审查记录。
2. 将四类执行器通道合并为 canonical `[N,4]` mask。
3. 生成 `channel_supervision=[1,1,1,1]` 或部分监督标记。
4. 对同一通道存在冲突的记录停止合并，写入冲突清单。
5. 对旧任务、非法 shape、缺失文件和未审查记录停止合并，写入诊断清单。
6. 按 `object_id` 做 deterministic split，避免同一物体泄漏到 train 和 test。

## 6. 安装训练环境

在 `10.24.1.11` 上建议创建独立环境：

```bash
conda create -n multiee-train python=3.11 -y
conda activate multiee-train

cd /home/lzq/Multi-EE-3DAG

python -m pip install \
  torch torchvision \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install -r MultiEEAffordance/training/requirements-training.txt
```

CUDA wheel 版本需要根据 `nvidia-smi` 和服务器驱动调整。如果服务器已有可用 PyTorch 环境，可以直接复用。

## 7. 先运行 synthetic smoke test

只检查训练数据准备，不依赖 PyTorch：

```bash
cd /home/lzq/Multi-EE-3DAG
python -m MultiEEAffordance.training.smoke_prepare_dataset
```

安装 PyTorch 后，再检查 dataloader、模型前向、loss 和反向传播：

```bash
cd /home/lzq/Multi-EE-3DAG
python -m MultiEEAffordance.training.smoke_test
```

正常输出应包含：

```text
"status": "ok"
"legacy_task_rejected": true
```

## 8. 生成人工审查训练集

人工标注仍在进行时，可以允许部分通道进入初版试跑：

```bash
python -m MultiEEAffordance.training.prepare_training_dataset \
  --dataset-root /home/lzq/data/MultiEEAffordance \
  --reviewed-samples \
processed/annotation_batches/v0_2_5tasks/reviewer_a_refined_samples.jsonl,processed/annotation_batches/v0_2_5tasks/reviewer_b_refined_samples.jsonl \
  --output-root processed/training/v0_2_5tasks_partial \
  --min-reviewed-channels 1 \
  --overwrite
```

正式训练集建议要求四通道都已经审查：

```bash
python -m MultiEEAffordance.training.prepare_training_dataset \
  --dataset-root /home/lzq/data/MultiEEAffordance \
  --reviewed-samples \
processed/annotation_batches/v0_2_5tasks/reviewer_a_refined_samples.jsonl,processed/annotation_batches/v0_2_5tasks/reviewer_b_refined_samples.jsonl \
  --output-root processed/training/v0_2_5tasks \
  --min-reviewed-channels 4 \
  --overwrite
```

输出：

```text
processed/training/v0_2_5tasks/
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

## 9. 运行初版训练

配置文件默认已经指向数据服务器路径：

```text
MultiEEAffordance/training/configs/baseline_task_executor_pointnet_5tasks.json
```

单卡运行：

```bash
cd /home/lzq/Multi-EE-3DAG
conda activate multiee-train

CUDA_VISIBLE_DEVICES=0 python -m MultiEEAffordance.training.train \
  --config MultiEEAffordance/training/configs/baseline_task_executor_pointnet_5tasks.json
```

checkpoint 和训练记录默认保存在：

```text
/home/lzq/data/MultiEEAffordance/
  processed/
    training_runs/
      baseline_task_executor_pointnet_5tasks_v0_1/
        latest.pt
        best.pt
        history.json
        resolved_config.json
```

## 10. 运行评估

```bash
CUDA_VISIBLE_DEVICES=0 python -m MultiEEAffordance.training.evaluate \
  --config MultiEEAffordance/training/configs/baseline_task_executor_pointnet_5tasks.json \
  --checkpoint /home/lzq/data/MultiEEAffordance/processed/training_runs/baseline_task_executor_pointnet_5tasks_v0_1/best.pt \
  --output-json processed/training_runs/baseline_task_executor_pointnet_5tasks_v0_1/test_metrics.json
```

## 11. 初版模型的定位

初版 `TaskExecutorPointNet` 用于验证：

1. 五任务数据是否能稳定加载。
2. `[N,4]` mask 是否正确。
3. 空 mask 是否进入 feasibility 学习。
4. 四类执行器是否能分别输出。
5. object-disjoint split 是否成立。

它不是最终 AAAI 投稿模型。数据闭环稳定后，再替换点云 backbone，并增加更强的 task-executor cross attention、局部几何特征和 cross-source 泛化实验。
