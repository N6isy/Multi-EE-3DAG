# 可视化环境与导出报告

更新时间：2026-05-07

## Open3D 专用虚拟环境

已创建项目专用虚拟环境：

```text
MultiEEAffordance/.venv-open3d/
```

该环境使用 Codex 工作区自带 Python：

```text
Python 3.12.13
```

创建过程中发现默认系统临时目录不可写，已将 `TEMP/TMP` 临时指向项目内：

```text
MultiEEAffordance/.tmp/
```

并成功修复 `pip`。

## Conda 可视化环境

用户已创建并启用 Conda 环境：

```text
multieeaffordance
```

当前环境检查结果：

```text
Python 3.11.15
open3d 0.19.0
matplotlib 3.10.9
numpy 2.4.4
```

因此 matplotlib PNG 可视化已经可以运行。

## 自纠偏方案：无依赖 HTML 可视化

为了不让图形检查流程卡在依赖安装上，已新增无外部依赖的 HTML 可视化导出脚本：

```text
tools/export_mask_html.py
```

该脚本只依赖 `numpy`，会生成可在浏览器中打开的独立 HTML 文件。页面支持：

- raw 点云显示；
- `gripper` / `suction` / `hook` / `dexterous_hand` 通道切换；
- 鼠标拖拽旋转；
- 滚轮缩放；
- 每个通道正样本点数显示。

## 已导出的可视化文件

已为当前 v3 的 61 条样本全部导出 HTML：

```text
processed/visualizations/html_v3/
```

索引页：

```text
processed/visualizations/html_v3/index.html
```

文件数量：

```text
61 个样本 HTML + 1 个 index.html = 62 个 HTML 文件
```

## 已导出的 PNG 可视化

已使用 `multieeaffordance` 环境导出 4 个代表样本 PNG：

```text
processed/visualizations/png_v3/door_open_pull.png
processed/visualizations/png_v3/door_press_push.png
processed/visualizations/png_v3/dishwasher_open_pull.png
processed/visualizations/png_v3/earphone_pick_up.png
```

导出过程中发现 matplotlib 在该 Conda 环境中默认尝试使用 Tk 后端，但 Tcl/Tk 不完整。已修正 `tools/visualize_masks.py`：当使用 `--output` 导出图片时，自动切换到无 GUI 的 `Agg` 后端。

## 生成命令

```bash
python tools/export_mask_html.py \
  --dataset-root . \
  --samples processed/metadata/samples.jsonl \
  --output-dir processed/visualizations/html_v3 \
  --limit 100 \
  --max-points 2048 \
  --write-index
```

## Open3D 交互窗口

Open3D 已安装，但从当前自动化环境直接启动 GUI 窗口时，权限审批超时。因此建议在 VSCode 的 PowerShell 终端中手动运行：

```bash
conda activate multieeaffordance

python tools/visualize_masks.py \
  --points processed/points/3d_affordancenet_full_shape_val_balanced_v3/<object_id>.npy \
  --masks processed/masks/3d_affordancenet_full_shape_val_balanced_v3/<sample_id>.npy \
  --channel all \
  --backend open3d
```

或使用 `matplotlib` backend 生成 PNG。
