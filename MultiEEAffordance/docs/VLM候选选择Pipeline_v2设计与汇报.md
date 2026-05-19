# VLM 候选选择 Pipeline v2 设计与汇报

更新时间：2026-05-19 +08:00

## 1. 当前问题背景

本项目目标是构建物体级、多执行器、多标签 3D affordance 数据集。当前输出不是单一 affordance mask，而是：

```text
P, q -> [M_gripper, M_suction, M_hook, M_dexterous_hand]
```

其中每个通道都表示某类 canonical end-effector 在给定任务下应优先接触、施力或形成机械约束的物体表面区域。

在前一版 Qwen3-VL + Florence-2 + SAM2 pipeline 中，我们尝试让 VLM 或开放词汇 grounding 模型直接从点云渲染图中定位目标部件，再由 SAM2 生成 2D mask 并回投到 3D。远程 `vlm_pilot_005: Bag / lift_carry / hook` 实验说明该路线可以跑通，但候选区域质量不稳定。

## 2. 旧 Pipeline 的主要缺陷

| 缺陷 | 表现 | 对数据构建的影响 |
| --- | --- | --- |
| 过度依赖 2D grounding 精确框 | Florence-2 能识别 `bag handle` 文本，但 box 经常覆盖整个 bag 或包体上半部分 | SAM2 会在大框内分割主体，生成过大的错误 mask |
| VLM 被要求做不擅长的像素定位 | Qwen3-VL 对稀疏点云图输出 box/point 不稳定 | 小部件、孔洞、提手等结构容易漏掉或偏移 |
| 点云渲染图不是自然图像 | 3D AffordanceNet 每个物体通常只有 2048 点，细结构只有少量点 | VLM/SAM2 难以像处理真实照片一样理解连续轮廓 |
| SAM2 prompt 质量受上游强约束 | 上游 box 偏大或偏错时，SAM2 会分割最显著的大区域 | 输出候选看似非空，但不符合执行器机制 |
| 几何规则缺少语义判断 | hook 专项候选可以找到上方稀疏结构，但不能判断是不是功能性 handle | 单靠几何容易把普通边缘、主体上沿当成正例 |
| 旧 hook 候选路线不可泛化 | `generate_hook_candidates.py` 针对 bag handle 做了上方结构候选 | 不适合 suction、gripper、dexterous_hand 和其他物体类型 |

核心结论：

```text
失败不是因为 VLM 没有价值，而是 pipeline 分工不合理。
VLM 应负责语义和机制判断，不应直接承担稀疏点云图上的精确像素标注。
```

## 3. 新版 Pipeline 的设计原则

新版 v2 pipeline 改为：

```text
3D geometry proposals
  -> candidate overlay rendering
  -> VLM candidate selection
  -> executor mechanism filtering
  -> four-channel candidate mask
  -> human review
```

也就是说，先由可解释的几何/弱标签模块提出候选区域，再让 VLM 从候选 A/B/C/D 中选择符合当前任务和执行器机制的区域。VLM 不再输出 box、point 或 mask。

### 3.0 远程空选择反馈与 v2.1 修正

在远程 `vlm_pilot_005: Bag / lift_carry / hook` 的 v2 候选选择实验中，Qwen3-VL 对 8 个视角都返回了空的 `selected_candidates` 和 `uncertain_candidates`。这不是简单的模型失效，而是暴露出 v2 的两个关键缺口：

| 暴露问题 | 具体表现 | 修正方向 |
| --- | --- | --- |
| 候选区域缺少语义桥接 | 候选名多为 `smooth_surface`、`thin_structure`、`edge_or_boundary` 等几何描述，VLM 难以把它们和 `bag handle / handle loop` 对齐 | 在候选选择 prompt 中显式加入 Qwen3-VL 语义部件计划，让 VLM 知道当前应寻找的目标部件 |
| 候选生成缺少面向稀疏点云的视觉部件候选 | bag handle 只有少量点，纯几何候选容易把它当成普通稀疏点或弱标签残留 | 从多视角 `point_index map` 中补充 `above_main_body_structure`、`detached_upper_or_side_component`、`small_visual_component` 等视觉候选 |
| 旧弱标签来源会误导 VLM | handle 区域可能来自 gripper weak mask，但当前任务是 hook，VLM 会认为它不是 hook 候选 | 弱标签候选增加说明：gripper/hand 的空间先验可能覆盖 handle，可作为 hook 的候选但必须经机制审查 |

因此，v2.1 的核心不是让 VLM 更激进地“猜”，而是把它的输入改成更符合任务的形式：**语义部件计划告诉 VLM 应该找什么，3D 候选生成器告诉 VLM 可以在哪些真实点云区域中选择**。

### 3.1 模块分工

| 模块 | 负责 | 不负责 |
| --- | --- | --- |
| 3D candidate generator | 从真实点云和已有弱 mask 中生成高召回候选区域 | 判断最终正例 |
| candidate overlay renderer | 把候选 A/B/C/D 画到多视角图上，给 VLM 和人工看 | 改变真实 3D 点 |
| Qwen3-VL selector | 根据任务、执行器和候选图选择 candidate IDs | 输出像素坐标或 mask |
| executor rule filter | 检查候选是否满足 gripper/suction/hook/hand 机制 | 替代人工审查 |
| mask builder | 将通过候选写回 `[N,4]` 的目标执行器通道 | 生成 ground truth |
| human review | accept / refine / reject / uncertain | 被自动流程取代 |

### 3.1.1 规则过滤的保守原则

规则过滤器只做机制一致性检查，不能在 VLM 已经运行后重新引入 VLM 没有支持的候选。否则会出现一种隐蔽错误：VLM 只看了 A-J，但规则模块又把 K-R 中某些几何上像边界/细结构的候选写入 accepted，最终 mask 会包含没有被 VLM 或人工看过的区域。

当前默认策略：

| 情况 | 默认处理 |
| --- | --- |
| 候选获得足够 VLM selected votes | 可以进入 `accepted_candidates`，仍需人工审查 |
| 候选有 VLM 投票但低于 `--min-selected-votes` | 降为 `uncertain_candidates` |
| 候选被展示给 VLM 但未被选中或标为 uncertain | 不自动进入正例 |
| 候选没有展示给 VLM | 直接拒绝，除非后续人工明确加入 |
| 完全没有 VLM selection 文件 | 可按配置启用 rule-only weak proposal，但必须标为弱候选 |

### 3.2 通用候选类型

| 候选类型 | 生成依据 | 主要服务对象 |
| --- | --- | --- |
| `existing_weak_mask` | 已有 checked/weak 四通道 mask | 所有执行器的先验 |
| `smooth_surface` | 低曲率、较平滑局部几何 | suction、press/push 面板 |
| `smooth_extreme_patch` | 物体极值方向附近的平滑 patch | suction、可推压面 |
| `edge_or_boundary` | 高曲率点 | gripper、hook、dexterous_hand |
| `extreme_edge_or_lip` | 极值方向上的边界/凸缘 | gripper、hook |
| `thin_structure` | 局部 PCA 线性结构 | handle、rod、stem、ring |
| `protruding_or_thin_part` | 轴向极值处的细长或突出结构 | gripper、hook、hand |
| `small_protrusion` | 小型凸起或高曲率紧凑区域 | button、knob、switch |
| `central_body` | 主体中部候选 | dexterous hand 包覆抓握，需严格审查 |
| `above_main_body_structure` | 多视角中位于主体上方或外侧的稀疏结构 | handle、loop、ring、top protrusion |
| `detached_upper_or_side_component` | 多视角中与主体分离或弱连接的小部件 | handle、hookable component、thin attachment |
| `small_visual_component` | 渲染图中的小型连通视觉部件 | button、knob、ring、handle fragment |

这些候选不是正例。它们的目标是尽量保证“正确区域在候选集合中”，后续再通过 VLM、规则和人工审查过滤。

需要强调的是，新增的视觉候选仍然是通用候选，不是 bag/handle 特例。它们只利用多视角渲染和 `point_index map` 找出“主体外的小型结构”，可服务于 hook 的环/孔/提手，也可服务于 gripper 的细柄、dexterous hand 的旋钮按钮等。

## 4. 四执行器机制过滤

| 执行器 | accept 倾向 | reject 倾向 |
| --- | --- | --- |
| `gripper` | 细长结构、可夹持边缘、凸起、柄部 | 大平面中心、过大主体区域、无相对接触潜力的面 |
| `suction` | 低曲率、连续、面积足够的平滑 patch | 细杆、边缘、孔洞、高曲率区域、把手 |
| `hook` | 可进入、可挂住、可沿任务方向施力的边界、孔洞、环、凸缘 | 普通平面、普通外轮廓、主体表面、大面积 smooth patch |
| `dexterous_hand` | 任务相关的包覆、捏取、按压、旋转或精细操作区域 | 所有可接触表面、无任务意义的大平面 |

规则过滤只决定候选是否进入 candidate mask，不决定 ground truth。最终仍需要网页人工审查。

## 5. 新增脚本

| 脚本 | 作用 |
| --- | --- |
| `tools/generate_3d_candidate_regions.py` | 从真实点云、已有弱 mask 和局部几何特征生成通用 3D 候选 |
| `tools/render_candidate_overlays_v2.py` | 把候选区域渲染成 VLM 可读的多视角 overlay 和 selector panel |
| `tools/run_vlm_candidate_selection_v2.py` | 让 Qwen3-VL 结合语义部件计划从候选 ID 中选择，而不是输出 box/mask |
| `tools/filter_candidates_by_executor_rules.py` | 按四类执行器机制对 VLM 选择结果做规则过滤 |
| `tools/build_v2_candidate_masks.py` | 将通过过滤的候选写回 `[N,4]` mask，并生成网页审查 JSONL |
| `tools/visualize_v2_candidates.py` | 为人工审查生成已选候选定位图、单候选网格图和 HTML 索引页 |

新增配置：

```text
configs/vlm_candidate_pipeline_v2.yaml
```

## 6. 推荐运行顺序

先确保已生成多视角 VLM-friendly render：

```bash
python MultiEEAffordance/tools/render_vlm_friendly_views.py \
  --dataset-root MultiEEAffordance \
  --pilot-id vlm_pilot_005 \
  --overwrite
```

生成通用 3D candidates：

```bash
python MultiEEAffordance/tools/generate_3d_candidate_regions.py \
  --dataset-root MultiEEAffordance \
  --pilot-id vlm_pilot_005 \
  --overwrite
```

生成候选 overlay：

```bash
python MultiEEAffordance/tools/render_candidate_overlays_v2.py \
  --dataset-root MultiEEAffordance \
  --pilot-id vlm_pilot_005 \
  --overwrite
```

远程运行 Qwen3-VL 候选选择：

```bash
CUDA_VISIBLE_DEVICES=1,2 python MultiEEAffordance/tools/run_vlm_candidate_selection_v2.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml \
  --pilot-id vlm_pilot_005 \
  --overwrite
```

如果只是检查文件路径：

```bash
python MultiEEAffordance/tools/run_vlm_candidate_selection_v2.py \
  --dataset-root MultiEEAffordance \
  --pilot-id vlm_pilot_005 \
  --validate-only
```

执行规则过滤：

```bash
python MultiEEAffordance/tools/filter_candidates_by_executor_rules.py \
  --dataset-root MultiEEAffordance \
  --pilot-id vlm_pilot_005 \
  --overwrite
```

生成四通道候选 mask：

```bash
python MultiEEAffordance/tools/build_v2_candidate_masks.py \
  --dataset-root MultiEEAffordance \
  --pilot-id vlm_pilot_005 \
  --overwrite
```

生成候选人工审查可视化：

```bash
python MultiEEAffordance/tools/visualize_v2_candidates.py \
  --dataset-root MultiEEAffordance \
  --pilot-id vlm_pilot_005 \
  --selected-candidates A \
  --overwrite
```

输出目录：

```text
processed/vlm_candidate_v2/review_visualizations/vlm_pilot_005/
```

其中：

| 文件 | 用途 |
| --- | --- |
| `index.html` | 汇总查看所有视角的人工审查页面 |
| `{view}_selected_context.png` | 左侧原图、中间仅显示已选候选、右侧局部放大 |
| `{view}_candidate_grid.png` | 每个候选编号单独成图，方便判断 A/B/C... 的位置 |

用现有网页工具复核：

```powershell
python MultiEEAffordance\tools\serve_review_app.py `
  --dataset-root MultiEEAffordance `
  --samples processed\metadata\v2_candidate_samples_v0_1.jsonl `
  --review-csv processed\metadata\v2_candidate_review_v0_1.csv `
  --host 127.0.0.1 `
  --port 8768 `
  --max-points 4096
```

## 7. 向导师汇报口径

当前阶段不是追求全自动标注，而是在搭建一个更可靠的 candidate mask 生成闭环。旧 pipeline 的失败说明：直接让 VLM 或 Florence-2 在稀疏点云渲染图上输出精确区域不稳定，SAM2 也会被错误大框带偏。

新版 pipeline 将问题拆成三层：

1. **几何层**：先从真实 3D 点云中提出可解释候选，保证候选和最终点云索引一致。
2. **语义层**：让 Qwen3-VL 判断哪些候选符合当前任务和执行器机制，充分利用 VLM 的知识。
3. **机制层**：用 gripper/suction/hook/dexterous_hand 的作用机制过滤候选，避免“能接触”被误标为“能完成任务”。

最终输出仍然只是 candidate proposal，必须进入人工审查。这样既保留了 VLM 的语义能力，也降低了其在稀疏点云图上做精确像素标注的不稳定性。

## 8. 后续验证重点

- 检查不同执行器的候选生成是否具有足够召回率。
- 检查 VLM 是否能稳定从候选编号中选择，而不是被图像稀疏性误导。
- 检查规则过滤是否过严或过松。
- 对 5 到 10 条 pilot 样本做完整闭环，而不是只看 `vlm_pilot_005`。
- 将网页人工审查结果反向用于调整候选生成器和规则阈值。
