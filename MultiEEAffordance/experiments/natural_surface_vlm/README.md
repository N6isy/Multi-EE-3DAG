# 自然化点云表面渲染与 2D-to-3D 回投实验

更新时间：2026-05-25

## 1. 实验目标

本实验单独验证一个新的视觉输入路线：

```text
稀疏点云 -> 接近自然图像的连续表面渲染 -> VLM/SAM2 产生 2D 区域
       -> 利用 point_index / confidence map 回投到原始 3D 点云
```

核心目标不是得到真实 mesh，也不是生成新的几何真值，而是让 VLM 更容易看懂物体结构，同时仍然保留可回投到原始点云的索引关系。

## 2. 为什么不直接用纯 mesh

前一轮 mesh 实验说明：对 3D AffordanceNet 这类稀疏点云，直接 Poisson / Ball Pivoting / Alpha Shape 重建容易出现：

| 问题 | 影响 |
| --- | --- |
| 细结构被吃掉 | bag handle、ring、hook hole 这类关键结构可能在 mesh 中消失 |
| 孔洞被补平 | hook 判断依赖“能进入/能挂住”的开口结构，mesh 可能把它补成普通表面 |
| 生成假面 | 稀疏点之间被错误连成面，VLM 可能识别到不存在的结构 |
| 回投不稳定 | mesh 面片不是原始点，若不维护点索引，很难形成可靠 `[N,4]` mask |

因此本实验采用 **surface splatting + confidence-controlled filling**，而不是把 mesh 当作主路径。

## 3. 当前设计

`render_natural_surface_views.py` 会为每个视角生成两类同步输出：

| 输出 | 用途 |
| --- | --- |
| `*_natural.png` | 给 VLM / Grounding / SAM2 使用的自然化渲染图 |
| `*_point_index.npy` | 每个有效像素对应的原始 3D 点索引 |
| `*_exact_point_index.npy` | 小半径稀疏点索引图，用于诊断真实点位置 |
| `*_confidence.npy` | 每个像素回投可信度，直接点最高，填补像素较低 |
| `*_source.npy` | 像素来源：0 背景，1 原始点 splat，2 邻近填补 |
| `*_panel.png` | 人工检查图：自然化渲染、回投置信度、前景放大 |

这样做的关键原则是：

```text
自然图像负责“让 VLM 看懂”；
point_index/confidence/source 负责“让 2D 区域可回投”。
```

## 4. 运行示例

以 `vlm_pilot_005` 为例生成自然化多视角图：

```bash
python MultiEEAffordance/experiments/natural_surface_vlm/render_natural_surface_views.py \
  --dataset-root MultiEEAffordance \
  --pilot-id vlm_pilot_005 \
  --splat-radius 10 \
  --fill-radius 12 \
  --blur-radius 1.6 \
  --edge-mode silhouette \
  --smooth \
  --overwrite
```

输出目录：

```text
MultiEEAffordance/processed/natural_surface_vlm/renders/<sample_id>/
```

建议优先查看：

```text
yaw000_elev20_panel.png
yaw045_elev20_panel.png
yaw180_elev20_panel.png
view_manifest.json
```

## 5. 2D mask 回投示例

假设 VLM/SAM2 已经在同名视角上生成了 2D mask：

```text
processed/natural_surface_vlm/example_2d_masks/
  yaw000_elev20_mask.png
  yaw045_elev20_mask.png
  yaw180_elev20_mask.png
```

将这些 2D mask 回投到原始点云的 `hook` 通道：

```bash
python MultiEEAffordance/experiments/natural_surface_vlm/project_natural_masks_to_3d.py \
  --dataset-root MultiEEAffordance \
  --manifest processed/natural_surface_vlm/renders/<sample_id>/view_manifest.json \
  --mask-root processed/natural_surface_vlm/example_2d_masks \
  --output-dir processed/natural_surface_vlm/projected_masks/vlm_pilot_005 \
  --executor hook \
  --min-confidence 0.35 \
  --min-view-votes 1 \
  --overwrite
```

生成：

| 文件 | 含义 |
| --- | --- |
| `*_point_mask.npy` | 单通道 3D candidate mask，shape `[N]` |
| `*_view_votes.npy` | 每个点被多少个视角命中 |
| `*_pixel_votes.npy` | 每个点累计对应多少个 2D mask 像素 |
| `*_multi_channel_mask.npy` | 写入指定 executor 通道的 `[N,4]` candidate mask |
| `projection_summary.json` | 回投统计和路径记录 |

## 6. 参数如何调

| 参数 | 建议 | 作用 |
| --- | --- | --- |
| `--splat-radius` | 8 到 14 | 越大越连续，但细节可能被糊成一片 |
| `--fill-radius` | 8 到 16 | 填小孔，让图更自然；过大可能扩张到背景 |
| `--blur-radius` | 1.0 到 2.0 | 弱化圆点颗粒感，让 VLM 输入更接近连续表面 |
| `--edge-mode` | `silhouette` | 默认只画外轮廓，避免内部裂纹干扰 VLM |
| `--fill-external-background` | 默认不开 | 开启后会向外填补背景，图更满但更容易产生胖边和假连接 |
| `--smooth` | 建议打开 | 让图更接近自然表面，但不改变索引图 |
| `--min-confidence` | 0.35 起步 | 回投时过滤过度填补的低可信像素 |
| `--direct-only` | 调试时使用 | 只允许原始点 splat 像素回投，不用填补像素 |

如果图像仍然像“颗粒球”：

```bash
--splat-radius 12 --fill-radius 14 --blur-radius 1.8 --edge-mode silhouette --smooth
```

如果细结构被糊掉，例如 handle 变粗或与主体粘连：

```bash
--splat-radius 7 --fill-radius 8 --blur-radius 1.0 --edge-mode silhouette
```

如果需要诊断深度断裂和孔洞位置，而不是给 VLM 看，可以临时使用：

```bash
--edge-mode depth
```

## 7. 判断是否成功

人工检查时重点看三件事：

| 检查项 | 通过标准 |
| --- | --- |
| VLM 可读性 | 物体主体、把手、孔洞、按钮等结构比点云图更容易辨认 |
| 细结构保留 | bag handle / ring / hole 没有被填平或消失 |
| 回投可信 | `confidence.png` 中目标区域不是大面积低置信度填补产生的假区域 |

如果自然化图看起来更像物体，但 `confidence.npy` 显示目标区域主要来自低置信度填补，则不能直接作为高质量候选，需要人工复核或调低 `fill-radius`。

## 8. 当前定位

本实验仍然只是 candidate proposal 生成路线，不改变项目的基本标注原则：

```text
VLM/SAM2 输出不是 ground truth；
自然化渲染不是新几何真值；
最终 mask 必须经过规则检查和人工审查。
```
