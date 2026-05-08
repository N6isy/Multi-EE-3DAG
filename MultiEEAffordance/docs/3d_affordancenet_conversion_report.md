# 3D AffordanceNet 接入报告

更新时间：2026-05-07

## 当前状态

已在 `raw/3d_affordancenet/` 下识别到三个压缩包：

| 文件 | 内容 | 说明 |
| --- | --- | --- |
| `rotate.zip` | `rotate_val_data.pkl` | 只包含旋转矩阵、类别和 affordance 名称，不包含点云。 |
| `full-shape.zip` | `full_shape_train_data.pkl`、`full_shape_val_data.pkl` | 包含完整物体点云和 18 类原始 affordance 逐点 mask。 |
| `partial.zip` | `partial_train_data.pkl`、`partial_val_data.pkl` | 体积较大，暂未作为 v0.1 首批物体级原型输入。 |

本轮没有全量解压 zip，而是直接从 zip 中读取 `full_shape_val_data.pkl`。这样可以避免把 `partial_train_data.pkl` 这类 10GB 级文件解压到工作区。

## 已生成的数据原型

当前正式 metadata 使用 v3 规则版本：

```text
manifests/3d_affordancenet_full_shape_val_balanced_v3_manifest.jsonl
processed/metadata/samples.jsonl
processed/metadata/3d_affordancenet_full_shape_val_balanced_v3_summary.json
```

对应数据文件位于：

```text
processed/points/3d_affordancenet_full_shape_val_balanced_v3/
processed/candidates/3d_affordancenet_full_shape_val_balanced_v3/
processed/masks/3d_affordancenet_full_shape_val_balanced_v3/
```

## 数据规模

从 `full_shape_val_data.pkl` 中读取到 2285 个物体。本轮为了先做可检查的平衡原型，每个类别最多选 2 个物体，共转换 46 个物体。

跳过四个执行器全不可行的 object-task 后，保留 61 条样本：

| 任务 | 样本数 |
| --- | ---: |
| `pick_up` | 21 |
| `lift_carry` | 21 |
| `open_pull` | 13 |
| `press_push` | 6 |

文件数量：

| 目录 | 文件数 |
| --- | ---: |
| `processed/points/3d_affordancenet_full_shape_val_balanced_v3/` | 46 |
| `processed/candidates/3d_affordancenet_full_shape_val_balanced_v3/` | 46 |
| `processed/masks/3d_affordancenet_full_shape_val_balanced_v3/` | 61 |

## 执行器可行性统计

| 任务 | gripper | suction | hook | dexterous_hand |
| --- | ---: | ---: | ---: | ---: |
| `pick_up` | 15 | 6 | 0 | 15 |
| `lift_carry` | 15 | 6 | 0 | 15 |
| `open_pull` | 11 | 4 | 6 | 11 |
| `press_push` | 0 | 2 | 0 | 4 |

## v3 弱标签规则

### `pick_up` / `lift_carry`

- `gripper`：来自原始 `grasp`、`wrap_grasp`、`lift`。
- `dexterous_hand`：来自原始 `grasp`、`wrap_grasp`、`lift`。
- `suction`：来自原始 `support`、`layable`，作为平面/支撑面弱代理。
- `hook`：不从 3D AffordanceNet full-shape 标签中自动生成。原因是该数据没有明确孔洞、内环、挂接边界标签。

### `open_pull`

- `gripper`：来自较小比例的 `pull`、`openable` 区域。
- `dexterous_hand`：来自较小比例的 `pull`、`openable` 区域。
- `hook`：来自更小比例的 `pull`、`openable` 区域。
- `suction`：来自 `pushable`，以及较大比例的 `pull`、`openable` 面板区域。

### `press_push`

- `suction`：来自原始 `pushable`。
- `dexterous_hand`：来自原始 `press`，以及小比例 `pushable`。
- `gripper` / `hook`：不自动生成。

## 已做的自我纠错

1. 初始版本对所有物体生成所有任务，导致一些不合理 object-task 组合存在。v2 开始跳过四个执行器全不可行的任务。
2. 初始版本在 `pick_up` / `lift_carry` 中用 `pull/openable` 推断 `hook`，这会把门把手或可开区域误当成可挂接结构。v2 已移除。
3. 初始版本在 `open_pull` 中把 `grasp` 也作为拉开候选，导致 `Earphone + open_pull` 这类明显牵强样本出现。v3 已移除。
4. 初始版本在 `press_push` 中把 `support/layable` 作为 suction 推断来源，容易把桌面/床面等支撑区域误当成推动区域。v2 已改为只使用 `pushable`。

## 当前检查结果

已运行：

```bash
python tools/check_dataset.py --dataset-root .
```

结果：

```text
samples_checked: 61
errors: 0
warnings: 0
```

## 当前限制

- 本机 Python 环境暂时没有 `matplotlib` 和 `open3d`，因此本轮没有生成 PNG 或交互式点云可视化。
- 3D AffordanceNet full-shape 没有法向，当前点云保存为 `[N, 3]`。
- 3D AffordanceNet 的原始 affordance 并不是按执行器定义的，因此当前 mask 是弱标签，不是最终人工确认标签。
- `hook` 通道在 full-shape 数据中信息不足，后续应优先用 PartNet-Mobility 的 handle、hole、ring、gap 等结构化部件补强。
- `suction` 通道目前主要依赖 `support/layable/pushable` 代理，后续最好结合真实法向、曲率和平面检测规则修正。

## 下一步建议

1. 安装或启用 `matplotlib` / `open3d` 后，对 v3 样本生成逐通道可视化。
2. 人工检查 `open_pull` 中的 Bag、Bottle、TrashCan 等类别，确认是否应该保留。
3. 接入 PartNet-Mobility，优先补强 drawer、cabinet、door、button、handle、ring、hole 等结构。
4. 对 full-shape train split 使用同一转换脚本生成更大规模 weak train，但建议先完成 v3 可视化抽检。
