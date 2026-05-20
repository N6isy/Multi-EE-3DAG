# 点云转 Mesh 增强 VLM 识别实验

更新时间：2026-05-19

## 1. 实验目的

当前 v2 pipeline 已经可以完成：

```text
3D 候选生成 -> 多视角候选 overlay -> VLM 候选选择 -> 规则过滤 -> 人工审查
```

但真实问题是：3D AffordanceNet 的点云通常较稀疏，细结构例如 handle、loop、ring 只由少量点表示。VLM 看到的是点状图，而不是连续物体表面，因此容易出现：

- 看不清物体结构；
- 把普通边缘误认为功能部件；
- 对 handle / hole / ring 这类细结构判断不稳定；
- SAM2 / grounding 模型在点云渲染图上容易分割整块主体。

本实验单独验证一个新想法：

```text
能否先将点云重建为 mesh 或连续表面渲染，
再把更像真实物体的多视角图给 VLM 看，
从而提升 VLM 对候选区域的语义判断能力。
```

本实验不改动 v2 pipeline。mesh 只作为 VLM 视觉辅助，不直接作为 ground truth。

## 2. 核心原则

| 原则 | 说明 |
| --- | --- |
| 不替代原始点云 | 最终 mask 仍然必须落回原始点云索引 `[N,4]` |
| 不直接把 mesh 当标签 | mesh 可能补出不存在的面，也可能填掉孔洞 |
| 只做视觉增强 | mesh render 用于帮助 VLM 理解物体结构和部件语义 |
| 必须人工复核 | VLM 基于 mesh render 的判断仍只是 candidate proposal |
| 和 v2 隔离 | 本目录只做实验验证，不改现有 v2 输出格式 |

## 3. 推荐验证对象

优先从已经暴露问题的 pilot 开始：

```text
vlm_pilot_005
Bag / lift_carry / hook
```

原因：

- bag handle 在点云中很稀疏；
- A 候选大部分正确但有 false positive；
- E 候选召回较高但过宽；
- 适合验证 mesh / surface render 是否能帮助 VLM 更稳定地区分 handle 和 bag body。

## 4. 实验脚本

| 脚本 | 作用 |
| --- | --- |
| `reconstruct_pointcloud_mesh.py` | 从 `points.npy` 重建 mesh，支持 Poisson / Ball Pivoting / Alpha Shape |
| `render_mesh_views.py` | 对 mesh 进行多视角渲染，生成 VLM-friendly 图片 |

## 5. 运行方式

示例命令：

```bash
python MultiEEAffordance/experiments/mesh_surface_vlm/reconstruct_pointcloud_mesh.py \
  --dataset-root MultiEEAffordance \
  --pilot-id vlm_pilot_005 \
  --output-dir processed/mesh_surface_vlm/vlm_pilot_005 \
  --method all \
  --overwrite
```

如果不使用 pilot metadata，也可以直接传入点云路径：

```bash
python MultiEEAffordance/experiments/mesh_surface_vlm/reconstruct_pointcloud_mesh.py \
  --dataset-root MultiEEAffordance \
  --points processed/points/YOUR_POINTS.npy \
  --output-dir processed/mesh_surface_vlm/manual_sample \
  --method all \
  --overwrite
```

渲染 mesh：

```bash
python MultiEEAffordance/experiments/mesh_surface_vlm/render_mesh_views.py \
  --dataset-root MultiEEAffordance \
  --mesh processed/mesh_surface_vlm/vlm_pilot_005/poisson_mesh.ply \
  --output-dir processed/mesh_surface_vlm/vlm_pilot_005/renders_poisson \
  --overwrite
```

## 5.1 Component-aware Mesh + 原始点叠加实验

如果纯 mesh 把 handle 吃掉，可以把候选点从 mesh 重建输入中排除，只用主体点重建 mesh，再在渲染时把原始点云和候选点叠加回来。

例如保留候选 `A,E`，只用剩余点重建 body mesh：

```bash
python MultiEEAffordance/experiments/mesh_surface_vlm/reconstruct_pointcloud_mesh.py \
  --dataset-root MultiEEAffordance \
  --pilot-id vlm_pilot_005 \
  --output-dir processed/mesh_surface_vlm/vlm_pilot_005_component_aware \
  --method all \
  --preserve-candidates A,E \
  --overwrite
```

该命令会额外输出：

```text
preserved_candidate_points.npy
preserved_candidate_mask.npy
body_points_for_mesh.npy
poisson_body_mesh.ply
ball_pivoting_body_mesh.ply
alpha_shape_body_mesh.ply
```

渲染时将 mesh 作为半透明背景，并叠加原始点云与候选 `A,E`：

```bash
python MultiEEAffordance/experiments/mesh_surface_vlm/render_mesh_views.py \
  --dataset-root MultiEEAffordance \
  --pilot-id vlm_pilot_005 \
  --mesh processed/mesh_surface_vlm/vlm_pilot_005_component_aware/poisson_body_mesh.ply \
  --output-dir processed/mesh_surface_vlm/vlm_pilot_005_component_aware/renders_poisson_hybrid_AE \
  --overlay-points \
  --overlay-candidates A,E \
  --mesh-alpha 0.55 \
  --point-alpha 0.25 \
  --overwrite
```

如果不想显示全部原始点，只想显示候选点，可以去掉 `--overlay-points`：

```bash
python MultiEEAffordance/experiments/mesh_surface_vlm/render_mesh_views.py \
  --dataset-root MultiEEAffordance \
  --pilot-id vlm_pilot_005 \
  --mesh processed/mesh_surface_vlm/vlm_pilot_005_component_aware/poisson_body_mesh.ply \
  --output-dir processed/mesh_surface_vlm/vlm_pilot_005_component_aware/renders_poisson_candidates_AE \
  --overlay-candidates A,E \
  --mesh-alpha 0.55 \
  --overwrite
```

这个 hybrid render 的目标不是得到更真实的 mesh，而是让 VLM 和人工同时看到：

```text
主体连续表面 + 原始真实点 + 被保留的候选功能部件
```

这比“纯 mesh”更适合 hook / handle / ring 这类依赖细结构的 affordance 判断。

## 6. 如何判断实验是否值得继续

人工检查 mesh render 时重点看：

| 检查项 | 通过标准 |
| --- | --- |
| 物体主体是否更连续 | VLM 能明显看出 bag body、handle 等结构 |
| handle 是否更清晰 | handle 不再只是几个孤立点 |
| 孔洞/环是否被保留 | hook 关键结构不能被 mesh 填平 |
| 是否产生假面 | mesh 不能大面积补出不存在的连接或封闭面 |
| 是否优于 dense point splatting | 如果只是变平滑但部件更假，就不值得接入 |

## 7. 初步技术判断

### Poisson Reconstruction

优点：

- 生成连续表面，VLM 视觉上更像真实物体；
- 对主体表面友好。

风险：

- 容易把孔洞、handle loop、薄结构填平；
- 对 hook 任务可能误导，因为 hook 依赖“能插入/挂住”的开放结构。

### Ball Pivoting

优点：

- 更依赖原始点分布，较少凭空补面；
- 对稀疏点云的细结构可能更保守。

风险：

- 点太稀疏时 mesh 破碎；
- 法向估计不稳定会导致面片质量差。

### Alpha Shape

优点：

- 参数可控，能得到较紧的外壳；
- 适合快速试验。

风险：

- alpha 过大填洞，过小碎裂；
- 对不同类别需要调参。

## 8. 目前建议

mesh route 可以作为一个独立实验，但不建议马上替换 v2。

更稳妥的下一步是并行比较三种视觉输入：

```text
1. 原始点云 dense render
2. mesh render
3. silhouette / surface splatting render
```

如果 mesh render 能让 VLM 更稳定地识别 `bag handle`，但不会填掉 hook 所需的孔洞/loop，再考虑把它作为 v2 的可选视觉输入，而不是替换候选生成和回投逻辑。
