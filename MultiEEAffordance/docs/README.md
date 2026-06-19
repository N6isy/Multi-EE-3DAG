# Multi-EE Affordance 文档入口

> 文档状态：当前主线
> 当前主线：五任务人工标注完成，训练进入 PointNeXt + 执行器属性条件建模阶段。
> 最新入口：docs/README.md、training/README.md、docs/五任务训练初版结果进度汇报.md。

更新时间：2026-06-16

## 当前任务边界

训练任务固定为五个独立任务：

```text
lift / open / pull / press / push
```

四类执行器顺序固定为：

```text
gripper / suction / hook / dexterous_hand
```

点级 mask 输出固定为 `[N,4]`，feasibility 输出固定为 `[4]`。任何模型接入都不能修改任务集合、执行器顺序和输出 shape。

## 当前代码入口

训练入口：

```text
MultiEEAffordance/training/train.py
```

模型工厂：

```text
MultiEEAffordance/training/model_factory.py
```

新增 backbone：

```text
MultiEEAffordance/training/backbones/pointnet_mlp.py
MultiEEAffordance/training/backbones/pointnext_adapter.py
```

执行器属性条件建模：

```text
MultiEEAffordance/training/executor_conditioning.py
MultiEEAffordance/training/configs/executor_specs_v0_1.json
```

## 推荐阅读顺序

1. `training/README.md`：训练数据、训练命令、评估命令。
2. `training/configs/README.md`：每个实验配置怎么用。
3. `docs/PointNeXt与PointTransformer接入步骤.md`：PointNeXt 当前依赖和接入状态。
4. `docs/执行器属性条件建模方法说明.md`：为什么做 executor attribute，以及 FiLM 如何进入模型。
5. `docs/五任务训练初版结果进度汇报.md`：PointNet 初版结果和下一阶段计划。

## 当前环境提醒

本仓库已经补齐 `external/backbones/PointNeXt-master/openpoints` 目录。当前训练环境可执行：

```bash
python -c "import sys; sys.path.insert(0, '/home/lzq/data/MultiEEAffordance/external/backbones/PointNeXt-master'); import openpoints; print('openpoints ok')"
```

但真实 PointNeXt forward 还依赖 OpenPoints 的 CUDA 扩展：

```text
pointnet2_batch_cuda
```

如果该扩展未编译，PointNeXt smoke test 会失败，这是环境依赖未完成，不应把 PointNet 或简化 MLP 冒充 PointNeXt。
