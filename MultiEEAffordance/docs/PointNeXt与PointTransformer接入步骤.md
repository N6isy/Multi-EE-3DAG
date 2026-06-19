# PointNeXt 与 PointTransformer 接入步骤

> 文档状态：操作手册
> 当前主线：五任务人工标注完成，训练进入 PointNeXt + 执行器属性条件建模阶段。
> 最新入口：docs/README.md、training/README.md、五任务训练初版结果进度汇报.md。

更新时间：2026-06-16

## 当前 PointNeXt 状态

外部仓库目录：

```text
/home/lzq/data/MultiEEAffordance/external/backbones/PointNeXt-master
```

关键目录已经存在：

```text
openpoints/
openpoints/models/backbone/pointnext.py
openpoints/models/segmentation/base_seg.py
cfgs/shapenetpart/pointnext-s.yaml
```

本项目新增 adapter：

```text
training/backbones/pointnext_adapter.py
```

adapter 使用 OpenPoints 的 `PointNextEncoder` 和 `PointNextDecoder`，不使用 OpenPoints 原始 segmentation head。输出点特征 `[B,N,D]` 后，继续接本项目自己的 task/executor head，最终保持：

```text
mask_logits: [B,N,4]
feasibility_logits: [B,4]
```

## 环境依赖

训练环境需要：

```text
torch
PyYAML
easydict
multimethod
scikit-learn
shortuuid
termcolor
```

这些 Python 依赖已经写入：

```text
training/requirements-training.txt
```

当前环境中，顶层 `openpoints` 已经可以 import：

```bash
cd /home/lzq/data
/home/lzq/miniconda3/envs/multiee-train/bin/python -c "import sys; sys.path.insert(0, '/home/lzq/data/MultiEEAffordance/external/backbones/PointNeXt-master'); import openpoints; print('openpoints ok')"
```

## CUDA extension

真实 PointNeXt forward 还需要 OpenPoints CUDA extension：

```text
pointnet2_batch_cuda
```

如果 smoke test 报：

```text
ModuleNotFoundError: No module named 'pointnet2_batch_cuda'
```

说明还没有编译扩展。需要在有 NVIDIA driver 和 CUDA toolkit 的机器上执行：

```bash
cd /home/lzq/data/MultiEEAffordance/external/backbones/PointNeXt-master/openpoints/cpp/pointnet2_batch
/home/lzq/miniconda3/envs/multiee-train/bin/python setup.py install
```

当前这台环境 `nvidia-smi` 无法连到驱动，且没有 `nvcc`，因此不能在这里完成 extension 编译。

## Smoke test

PointNeXt backbone smoke test：

```bash
cd /home/lzq/data
/home/lzq/miniconda3/envs/multiee-train/bin/python -m MultiEEAffordance.training.smoke_test_backbone \
  --backbone pointnext \
  --dataset-root /home/lzq/data/MultiEEAffordance \
  --manifest processed/training/v0_4_final_5tasks/manifests/val.jsonl \
  --pointnext-root /home/lzq/data/MultiEEAffordance/external/backbones/PointNeXt-master
```

在 CUDA extension 正常时，输出应包含：

```text
input_shape: [B,N,C]
output_shape: [B,N,D]
has_nan: false
```

## 训练配置

新增 PointNeXt 配置：

```text
training/configs/pointnext_shared4_5tasks_debug.json
training/configs/pointnext_shared4_5tasks.json
training/configs/pointnext_one_hot_executor_token_5tasks.json
training/configs/pointnext_with_relation_loss_5tasks.json
```

新增执行器属性消融：

```text
training/configs/pointnext_executor_attr_only_5tasks.json
training/configs/pointnext_executor_id_attr_5tasks.json
training/configs/pointnext_executor_id_attr_film_5tasks.json
training/configs/pointnext_executor_id_attr_crossattn_5tasks.json
```

debug 训练命令：

```bash
cd /home/lzq/data
CUDA_VISIBLE_DEVICES=0 /home/lzq/miniconda3/envs/multiee-train/bin/python -m MultiEEAffordance.training.train \
  --config MultiEEAffordance/training/configs/pointnext_shared4_5tasks_debug.json
```

## PointTransformer 后续接入

PointTransformer 也应遵守同一个接口：

```text
features = backbone(points)
输入 points: [B,N,C]
输出 features: [B,N,D]
```

不要直接复用外部 segmentation head。所有模型都必须接本项目统一 task/executor head，保证输出 shape、任务集合和 executor 顺序一致。
