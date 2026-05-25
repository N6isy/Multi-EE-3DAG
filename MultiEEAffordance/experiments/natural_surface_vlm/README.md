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
| `<view>/natural_render.png` | 给 VLM / Grounding / SAM2 使用的自然化渲染图 |
| `<view>/point_index.npy` | 每个有效像素对应的原始 3D 点索引 |
| `<view>/exact_point_index.npy` | 小半径稀疏点索引图，用于诊断真实点位置 |
| `<view>/confidence.npy` | 每个像素回投可信度，直接点最高，填补像素较低 |
| `<view>/source.npy` | 像素来源：0 背景，1 原始点 splat，2 邻近填补 |
| `<view>/panel.png` | 人工检查图：自然化渲染、回投置信度、前景放大 |

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
  --densify-midpoints \
  --densify-threshold-multiplier 2.2 \
  --densify-max-neighbors 3 \
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
  view_manifest.json
  yaw000_elev20/
    natural_render.png
    point_index.npy
    exact_point_index.npy
    confidence.npy
    source.npy
    panel.png
```

建议优先查看：

```text
yaw000_elev20/panel.png
yaw045_elev20/panel.png
yaw180_elev20/panel.png
view_manifest.json
```

## 4.1 接入 v3 主 pipeline

从 2026-05-25 起，自然化渲染已从独立实验模块接入正式 v3 候选生成主链路。正式入口是：

```bash
python MultiEEAffordance/tools/render_natural_surface_views.py \
  --dataset-root MultiEEAffordance \
  --pilot-csv processed/metadata/v3_large_scale_review_queue_v0_1.csv \
  --samples processed/metadata/samples_v3_large_batch_v0_1.jsonl \
  --overwrite
```

正式 v3 默认输出目录为：

```text
processed/vlm_candidate_v3/natural_renders/<sample_id>/<view>/
```

`run_v3_pipeline.py` 也已支持新的候选主链路：

```bash
CUDA_VISIBLE_DEVICES=1,2 python MultiEEAffordance/tools/run_v3_pipeline.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml \
  --pilot-csv processed/metadata/v3_large_scale_review_queue_v0_1.csv \
  --samples processed/metadata/samples_v3_large_batch_v0_1.jsonl \
  --candidate-source part3d \
  --view-renderer natural \
  --stages views,plan,part_propose,render,part_select,part_filter,build \
  --allow-empty \
  --overwrite
```

这条新路径中，VLM 不再作为像素级 box/point 生成器。它先在 `plan` 阶段判断 target/reject 语义部件，再在 `part_select` 阶段从候选 ID 中选择；候选点本身由 `part_propose` 通过自然化前景、连通组件、几何结构和 source weak mask 等信息生成，最终仍然只回到原始点云索引，形成 `[N,4]` candidate mask。

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

## 6. 独立测试 Qwen3-VL 是否看懂自然化图

如果只想验证“自然化渲染图能不能让 VLM 更好地识别任务部件”，不要改 v3 pipeline，直接运行独立 probe：

```bash
CUDA_VISIBLE_DEVICES=1,2 python MultiEEAffordance/experiments/natural_surface_vlm/run_natural_vlm_probe.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml \
  --pilot-id vlm_pilot_005 \
  --image-key natural_render_path \
  --probe-mode semantic_and_localize \
  --refine-min-confidence 0.25 \
  --refine-snap-radius 48 \
  --overwrite
```

输出目录：

```text
processed/natural_surface_vlm/vlm_probe/vlm_pilot_005/
```

重点查看：

| 文件 | 含义 |
| --- | --- |
| `combined_probe_summary.json` | 多视角汇总，统计 VLM 是否认为目标部件可见、可定位 |
| `view_probe_results.json` | 每个视角的语义判断和粗定位结果 |
| `*_semantic.json` | 当前视角下 VLM 识别出的目标部件、reject 部件和机制解释 |
| `*_localization.json` | 当前视角下 VLM 给出的粗 box / positive point |
| `*_probe_overlay.png` | 将 VLM 的粗定位画回自然化图，便于人工判断是否靠谱 |
| `index.html` | 浏览器查看的汇总页面 |

默认会对 VLM 粗定位做一层轻量后处理：

```text
VLM box/point -> snap 到可回投前景 -> 若目标词包含 handle/loop/ring/hole/top，则优先裁剪到主体上方结构
```

后处理只用于这个 probe 的诊断输出，不会修改 v3 pipeline。若想看 Qwen3-VL 原始定位，可加：

```bash
--no-refine-localization
```

判断标准：

| 检查项 | 说明 |
| --- | --- |
| `ranked_target_parts` 是否包含目标部件 | 例如 `Bag / lift_carry / hook` 应该出现 `bag handle / handle loop / handle inner rim` |
| `ranked_reject_parts` 是否排除主体 | 例如不应把 `bag body panel` 当成 hook 正例 |
| `localizable_views` 是否足够 | 如果语义正确但定位差，说明自然化图改善了理解，但仍需更强 grounding |
| overlay 是否紧贴目标部件 | box/point 应落在 handle，不应覆盖整块 bag body |

该 probe 不生成 3D mask，也不写入 v2/v3 candidate 目录，只用于判断这个自然化渲染想法是否值得继续接入后续 Grounding/SAM2。

## 7. 参数如何调

| 参数 | 建议 | 作用 |
| --- | --- | --- |
| `--splat-radius` | 8 到 14 | 越大越连续，但细节可能被糊成一片 |
| `--densify-midpoints` | 建议尝试 | 在相邻点之间生成渲染用中点，让 handle / ring 等细结构更连续 |
| `--densify-threshold-multiplier` | 1.6 到 3.0 | 自动距离阈值 = 中位最近邻距离 × 该倍率 |
| `--densify-max-neighbors` | 2 到 4 | 每个原始点最多连接几个邻近点 |
| `--densify-distance` | 默认 0 | 若手动指定，则使用归一化坐标下的绝对距离阈值 |
| `--fill-radius` | 8 到 16 | 填小孔，让图更自然；过大可能扩张到背景 |
| `--blur-radius` | 1.0 到 2.0 | 弱化圆点颗粒感，让 VLM 输入更接近连续表面 |
| `--edge-mode` | `silhouette` | 默认只画外轮廓，避免内部裂纹干扰 VLM |
| `--fill-external-background` | 默认不开 | 开启后会向外填补背景，图更满但更容易产生胖边和假连接 |
| `--smooth` | 建议打开 | 让图更接近自然表面，但不改变索引图 |
| `--min-confidence` | 0.35 起步 | 回投时过滤过度填补的低可信像素 |
| `--direct-only` | 调试时使用 | 只允许原始点 splat 像素回投，不用填补像素 |

如果图像仍然像“颗粒球”：

```bash
--densify-midpoints --densify-threshold-multiplier 2.5 --splat-radius 12 --fill-radius 14 --blur-radius 1.8 --edge-mode silhouette --smooth
```

如果细结构被糊掉，例如 handle 变粗或与主体粘连：

```bash
--densify-midpoints --densify-threshold-multiplier 1.6 --densify-max-neighbors 2 --splat-radius 7 --fill-radius 8 --blur-radius 1.0 --edge-mode silhouette
```

如果需要诊断深度断裂和孔洞位置，而不是给 VLM 看，可以临时使用：

```bash
--edge-mode depth
```

## 8. 判断是否成功

人工检查时重点看三件事：

| 检查项 | 通过标准 |
| --- | --- |
| VLM 可读性 | 物体主体、把手、孔洞、按钮等结构比点云图更容易辨认 |
| 细结构保留 | bag handle / ring / hole 没有被填平或消失 |
| 回投可信 | `confidence.png` 中目标区域不是大面积低置信度填补产生的假区域 |

如果自然化图看起来更像物体，但 `confidence.npy` 显示目标区域主要来自低置信度填补，则不能直接作为高质量候选，需要人工复核或调低 `fill-radius`。

## 9. 当前定位

本实验仍然只是 candidate proposal 生成路线，不改变项目的基本标注原则：

```text
VLM/SAM2 输出不是 ground truth；
自然化渲染不是新几何真值；
最终 mask 必须经过规则检查和人工审查。
```
