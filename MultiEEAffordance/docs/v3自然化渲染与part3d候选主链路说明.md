# v3 自然化渲染与 part3d 候选主链路说明

更新时间：2026-05-25

## 1. 当前目标

v3 主链路保持“语义规划 -> 候选生成 -> VLM 选择 -> 规则过滤 -> 人工审查 -> `[N,4]` mask”的总体结构，但把视觉输入和候选生成方式做了升级：

- 自然化点云表面渲染用于给 VLM 看更接近自然图像的多视角表面图。
- `part3d` 候选生成用于替代默认的 VLM 坐标定位路径。
- VLM 只负责判断 target/reject 语义和从候选 ID 中选择，不直接输出最终点级标注。
- 所有候选最终仍只映射回原始点云索引，原始点云数量 `N` 不变。

## 2. 自然化渲染输出

正式 v3 入口：

```bash
python MultiEEAffordance/tools/render_natural_surface_views.py \
  --dataset-root MultiEEAffordance \
  --pilot-csv processed/metadata/v3_large_scale_review_queue_v0_1.csv \
  --samples processed/metadata/samples_v3_large_batch_v0_1.jsonl \
  --overwrite
```

默认输出目录：

```text
processed/vlm_candidate_v3/natural_renders/<sample_id>/
  view_manifest.json
  yaw000_elev20/
    natural_render.png
    point_index.npy
    exact_point_index.npy
    confidence.npy
    source.npy
    panel.png
```

字段约定：

| 文件 | 用途 |
| --- | --- |
| `natural_render.png` | 给 VLM、候选 overlay、人工诊断使用的自然化视图 |
| `point_index.npy` | 每个有效像素回指的原始 3D 点索引，范围必须是 `[0, N)` |
| `exact_point_index.npy` | 只记录原始稀疏点直接投影位置，用于诊断 |
| `confidence.npy` | 像素回投可信度，范围 `[0, 1]` |
| `source.npy` | 像素来源：`0=background`，`1=direct splat`，`2=filled/interpolated` |
| `panel.png` | 人工检查图，展示自然图、置信度和前景放大 |

自然化渲染可以生成 render-only midpoint 和填补像素，但这些像素必须回指原始点云中的某个端点或可信近邻，不能产生新的真实点。

## 3. 新 v3 默认链路

推荐服务器命令：

```bash
CUDA_VISIBLE_DEVICES=1,2 python MultiEEAffordance/tools/run_v3_pipeline.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml \
  --pilot-csv processed/metadata/v3_large_scale_review_queue_v0_1.csv \
  --samples processed/metadata/samples_v3_large_batch_v0_1.jsonl \
  --include-tasks pick_up,open_pull,press_push \
  --exclude-tasks lift_carry \
  --candidate-source part3d \
  --view-renderer natural \
  --stages views,plan,part_propose,render,part_select,part_filter,build \
  --allow-empty \
  --overwrite
```

阶段含义：

| 阶段 | 作用 |
| --- | --- |
| `views` | 生成自然化多视角图和 `point_index/confidence/source` 回投图 |
| `plan` | VLM 判断 target 部件、reject 部件和任务/执行器机制约束 |
| `part_propose` | 生成 3D 候选 mask，输出 `candidate_masks[K,N]` |
| `render` | 生成候选 overlay，供 VLM 和人工查看 |
| `part_select` | VLM 只选择/拒绝候选 ID，不输出 box/point |
| `part_filter` | 执行器规则保守过滤，并更新 `default_selected_candidates` |
| `build` | 生成审查系统可加载的 `[N,4]` candidate mask 样本 |

旧的 `ground -> project -> grow` 路径保留为 fallback：

```bash
--candidate-source grounding --stages views,plan,ground,project,grow,render,coverage,build
```

## 4. part3d 候选生成

正式工具：

```bash
python MultiEEAffordance/tools/propose_v3_part_candidates.py \
  --dataset-root MultiEEAffordance \
  --pilot-csv processed/metadata/v3_large_scale_review_queue_v0_1.csv \
  --samples processed/metadata/samples_v3_large_batch_v0_1.jsonl \
  --renders-root processed/vlm_candidate_v3/natural_renders \
  --backend natural_cc \
  --overwrite
```

第一版 `natural_cc` 后端复用高召回几何/视觉候选生成逻辑，包括：

- 自然化前景候选；
- 连通组件、detached / upper structure；
- thin / loop-like / edge / extreme component；
- source weak mask fallback；
- 执行器相关候选 family 标记。

预留的 `sam3d` 和 `partslippp` adapter 已在参数层占位，但第一版不强制安装外部模型。后续接入 SAM3D / PartSLIP++ 时，应保持统一输出协议：`candidate_manifest.json` 和 `candidates.npz`，其中 `candidate_masks` 仍为 `[K,N]`。

## 5. 鲁棒性修复

`build_v3_candidate_masks.py` 已增加 `safe_dict(value)` 和 row-level nonfatal 处理：

- `feasibility / label_source / negative_reason` 即使是字符串、列表或空值，也不会触发 `dict(value)` 崩溃。
- `build empty` 循环单条失败时会写入 `failed_needs_review` 占位样本，不中断整批 pipeline。
- 普通候选行失败会进入 `build_errors`，便于后续定位。

这用于解决服务器端出现的：

```text
ERROR: dictionary update sequence element #0 has length 1; 2 is required
```

## 6. 本地已完成检查

- `python -m py_compile` 已覆盖自然化渲染、part3d 候选、v3 pipeline、VLM selection、规则过滤和 build 脚本。
- 自然化渲染 smoke test 已确认：
  - `point_index.shape == (768, 768)`；
  - 有效索引全部 `< N`；
  - `source` 只包含 `{0,1,2}`；
  - `confidence` 范围在 `[0,1]`。
- part3d smoke test 已确认：
  - 能生成 v3 兼容 `candidate_manifest.json`；
  - `candidate_masks.shape == [K,N]`；
  - 默认不把自动候选直接当作 GT。

## 7. 后续注意

- 自然化图像只是视觉代理，不是 ground truth。
- `point_index/confidence/source` 是回投依据，最终标注仍在原始点云 `N` 个点上。
- VLM 的输出只能作为候选选择建议，必须经过规则过滤和人工审查。
- 对 scissors、bag、door、faucet 等样本，应优先人工检查 `panel.png`，确认 handle、loop、ring、button、flat panel 是否比旧稀疏点图更容易识别。

## 8. 小批测试到全量运行流程

本文件中的命令不是天然“全量数据命令”。实际处理范围由 `--samples` 和 `--pilot-csv` 决定：

- `samples_v3_large_batch_v0_1.jsonl` 是当前从 3D AffordanceNet 转出的样本集合。
- `v3_large_scale_review_queue_v0_1.csv` 是从 samples 展开的审查队列。
- 如果构建队列时带 `--limit 300`，那么后续 pipeline 只会处理这 300 条队列。
- 若要全量，需要重新构建不带 `--limit` 的队列，或显式换成全量 queue 文件。

### 8.1 先生成小批测试队列

建议先用 12 到 24 条，类别/任务/执行器尽量分散：

```bash
python MultiEEAffordance/tools/build_large_scale_review_queue.py \
  --dataset-root MultiEEAffordance \
  --samples processed/metadata/samples_v3_large_batch_v0_1.jsonl \
  --output-csv processed/metadata/v3_test_review_queue_v0_1.csv \
  --summary-json processed/metadata/v3_test_review_queue_summary_v0_1.json \
  --include-tasks pick_up,open_pull,press_push \
  --exclude-tasks lift_carry \
  --executor-scope all \
  --quality-scope all \
  --empty-policy review \
  --common-sense-filter \
  --limit 24 \
  --limit-strategy round_robin_category_task_executor \
  --overwrite
```

先看 summary：

```bash
cat MultiEEAffordance/processed/metadata/v3_test_review_queue_summary_v0_1.json
```

重点确认：

- `counts_by_task` 不要只集中在一个任务；
- `counts_by_executor` 四类执行器是否都有；
- `counts_by_decision` 中 `review` 和 `empty_review_required` 是否符合预期；
- `skipped` 是否只包含明显不合理的物体-任务组合。

### 8.2 先只跑非 VLM 可视化候选

这一步用于快速检查自然化渲染和 part3d 候选是否像样，不先消耗大量 VLM 时间：

```bash
python MultiEEAffordance/tools/run_v3_pipeline.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml \
  --pilot-csv processed/metadata/v3_test_review_queue_v0_1.csv \
  --samples processed/metadata/samples_v3_large_batch_v0_1.jsonl \
  --include-tasks pick_up,open_pull,press_push \
  --exclude-tasks lift_carry \
  --candidate-source part3d \
  --view-renderer natural \
  --stages views,part_propose,render \
  --limit 6 \
  --allow-empty \
  --overwrite
```

人工先看：

```text
processed/vlm_candidate_v3/natural_renders/<sample_id>/<view>/panel.png
processed/vlm_candidate_v3/candidate_overlays/<pilot_id>/
processed/vlm_candidate_v3/3d_candidates/<pilot_id>/candidate_manifest.json
```

判断标准：

- 自然化图是否比原始稀疏点云更容易看出 handle、loop、ring、button、flat panel；
- 候选是否覆盖目标部件，而不是只覆盖普通边缘、刀刃、主体表面；
- 候选数量是否适合人工审查，通常希望 top-k 内有 3 到 8 个可比较区域。

### 8.3 小批完整跑 VLM 选择和 build

如果 8.2 的候选质量能接受，再跑完整小批：

```bash
CUDA_VISIBLE_DEVICES=1,2 python MultiEEAffordance/tools/run_v3_pipeline.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml \
  --pilot-csv processed/metadata/v3_test_review_queue_v0_1.csv \
  --samples processed/metadata/samples_v3_large_batch_v0_1.jsonl \
  --include-tasks pick_up,open_pull,press_push \
  --exclude-tasks lift_carry \
  --candidate-source part3d \
  --view-renderer natural \
  --stages views,plan,part_propose,render,part_select,part_filter,build \
  --limit 6 \
  --allow-empty \
  --overwrite
```

输出重点：

```text
processed/vlm_candidate_v3/semantic_plans/<pilot_id>/combined_semantic_plan.json
processed/vlm_candidate_v3/vlm_selection/<pilot_id>/combined_selection.json
processed/vlm_candidate_v3/3d_candidates/<pilot_id>/candidate_manifest.json
processed/metadata/v3_candidate_samples_v0_1.jsonl
processed/metadata/v3_candidate_summary_v0_1.json
```

检查标准：

- semantic plan 是否把任务相关部件列为 target，并明确 reject 普通主体/刀刃/无关平面；
- `combined_selection.json` 中 VLM 是否只选择候选 ID；
- `candidate_manifest.json` 中 `default_selected_candidates` 是否经过 VLM 和规则过滤更新；
- `v3_candidate_summary_v0_1.json` 是否没有大面积 `failed_needs_review`。

### 8.4 打开审查系统看人工体验

```bash
python MultiEEAffordance/tools/serve_v2_annotation_app.py \
  --dataset-root MultiEEAffordance \
  --samples processed/metadata/v3_candidate_samples_v0_1.jsonl \
  --review-jsonl processed/metadata/v3_point_level_review_records.jsonl \
  --output-mask-root processed/vlm_candidate_v3/manual_refined_masks \
  --output-samples processed/metadata/v3_manual_refined_samples_v0_1.jsonl \
  --port 8765 \
  --max-points 0 \
  --top-k-candidates 8
```

浏览器打开：

```text
http://127.0.0.1:8765
```

建议至少人工检查 6 类情况：

- Bag / hook 或 gripper；
- Scissors / hook；
- Door 或 Dishwasher / open_pull；
- Keyboard 或 Button-like object / press_push；
- Knife / pick_up；
- 一个 `empty_review_required` 空标签样本。

如果这些样本的候选已经能把人工工作从“从零画点”变成“选择候选组合 + 少量点级修正”，再扩大规模。

### 8.5 扩大到 300 条队列

```bash
python MultiEEAffordance/tools/build_large_scale_review_queue.py \
  --dataset-root MultiEEAffordance \
  --samples processed/metadata/samples_v3_large_batch_v0_1.jsonl \
  --output-csv processed/metadata/v3_large_scale_review_queue_v0_1.csv \
  --summary-json processed/metadata/v3_large_scale_review_queue_summary_v0_1.json \
  --include-tasks pick_up,open_pull,press_push \
  --exclude-tasks lift_carry \
  --executor-scope all \
  --quality-scope all \
  --empty-policy review \
  --common-sense-filter \
  --limit 300 \
  --limit-strategy round_robin_category_task_executor \
  --overwrite
```

然后运行：

```bash
CUDA_VISIBLE_DEVICES=1,2 python MultiEEAffordance/tools/run_v3_pipeline.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml \
  --pilot-csv processed/metadata/v3_large_scale_review_queue_v0_1.csv \
  --samples processed/metadata/samples_v3_large_batch_v0_1.jsonl \
  --include-tasks pick_up,open_pull,press_push \
  --exclude-tasks lift_carry \
  --candidate-source part3d \
  --view-renderer natural \
  --stages views,plan,part_propose,render,part_select,part_filter,build \
  --allow-empty \
  --overwrite
```

### 8.6 真正全量运行

确认 300 条效果稳定后，重新构建不带 `--limit` 的全量队列：

```bash
python MultiEEAffordance/tools/build_large_scale_review_queue.py \
  --dataset-root MultiEEAffordance \
  --samples processed/metadata/samples_v3_large_batch_v0_1.jsonl \
  --output-csv processed/metadata/v3_full_review_queue_v0_1.csv \
  --summary-json processed/metadata/v3_full_review_queue_summary_v0_1.json \
  --include-tasks pick_up,open_pull,press_push \
  --exclude-tasks lift_carry \
  --executor-scope all \
  --quality-scope all \
  --empty-policy review \
  --common-sense-filter \
  --limit-strategy round_robin_category_task_executor \
  --overwrite
```

再跑全量：

```bash
CUDA_VISIBLE_DEVICES=1,2 python MultiEEAffordance/tools/run_v3_pipeline.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml \
  --pilot-csv processed/metadata/v3_full_review_queue_v0_1.csv \
  --samples processed/metadata/samples_v3_large_batch_v0_1.jsonl \
  --include-tasks pick_up,open_pull,press_push \
  --exclude-tasks lift_carry \
  --candidate-source part3d \
  --view-renderer natural \
  --stages views,plan,part_propose,render,part_select,part_filter,build \
  --allow-empty \
  --overwrite
```

全量之前不要删除 `--common-sense-filter`。它只过滤明显不匹配的物体-任务组合，能减少大量无意义 VLM 调用；对于同一物体任务下不可行的执行器，`--executor-scope all --empty-policy review` 会保留为空标签审查样本。
