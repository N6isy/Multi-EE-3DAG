# v3 部件分割候选与 PartSLIP++ 接入说明

更新时间：2026-05-25

## 1. 当前决策

v3 正式链路不再包含自然化点云表面渲染模块。后续候选生成收敛为：

```text
原始点云 / 弱标签 / 可选部件分割模型
  -> 3D part candidates [K, N]
  -> VLM 只选择候选 ID
  -> 执行器规则过滤
  -> 人工审查
  -> [N,4] mask
```

其中 `N` 始终是原始点云点数。任何自动候选都只是 candidate，不是 ground truth。

## 2. 保留模块

正式候选入口：

```bash
python MultiEEAffordance/tools/propose_v3_part_candidates.py \
  --dataset-root MultiEEAffordance \
  --pilot-csv processed/metadata/v3_test_review_queue_v0_1.csv \
  --samples processed/metadata/samples_v3_large_batch_v0_1.jsonl \
  --backend geometry \
  --overwrite
```

当前支持：

| backend | 状态 | 说明 |
| --- | --- | --- |
| `geometry` | 可运行 | 基于原始点云几何、已有弱标签和通用候选 family 生成高召回 `[K,N]` 候选 |
| `partslippp` | 预留 | 后续接入 PartSLIP++ 部件分割，输出同一套 `candidate_manifest.json` 和 `candidates.npz` |

`partslippp` 后端当前会明确报 `NotImplementedError`，避免误以为已经完成外部模型集成。

## 3. v3 默认测试流程

### 3.1 生成小批测试队列

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

先检查 summary：

```bash
cat MultiEEAffordance/processed/metadata/v3_test_review_queue_summary_v0_1.json
```

重点看任务、执行器和类别是否分散，避免小批样本只覆盖一种情况。

### 3.2 先跑非 VLM 候选可视化

这一步最快，用来判断候选是否能帮助人工，而不是先消耗 VLM 时间：

```bash
python MultiEEAffordance/tools/run_v3_pipeline.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml \
  --pilot-csv processed/metadata/v3_test_review_queue_v0_1.csv \
  --samples processed/metadata/samples_v3_large_batch_v0_1.jsonl \
  --include-tasks pick_up,open_pull,press_push \
  --exclude-tasks lift_carry \
  --candidate-source partseg \
  --part-proposal-backend geometry \
  --stages views,part_propose,render \
  --limit 6 \
  --allow-empty \
  --overwrite
```

检查输出：

```text
processed/vlm_semantic_part/renders/<sample_id>/view_manifest.json
processed/vlm_candidate_v3/3d_candidates/<pilot_id>/candidate_manifest.json
processed/vlm_candidate_v3/candidate_overlays/<pilot_id>/
```

判断标准：

- 候选是否覆盖目标部件，而不是只覆盖普通边缘、主体大平面或明显无关区域；
- 候选 top-k 是否适合人工选择，通常 3 到 8 个候选最舒服；
- 剪刀、包、门、龙头这类样本是否能出现 handle / loop / ring / button / panel 相关候选。

### 3.3 小批完整跑 VLM 选择

候选质量可以后，再跑 VLM 语义规划和候选选择：

```bash
CUDA_VISIBLE_DEVICES=1,2 python MultiEEAffordance/tools/run_v3_pipeline.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml \
  --pilot-csv processed/metadata/v3_test_review_queue_v0_1.csv \
  --samples processed/metadata/samples_v3_large_batch_v0_1.jsonl \
  --include-tasks pick_up,open_pull,press_push \
  --exclude-tasks lift_carry \
  --candidate-source partseg \
  --part-proposal-backend geometry \
  --stages views,plan,part_propose,render,part_select,part_filter,build \
  --limit 6 \
  --allow-empty \
  --overwrite
```

重点检查：

```text
processed/vlm_candidate_v3/semantic_plans/<pilot_id>/combined_semantic_plan.json
processed/vlm_candidate_v3/vlm_selection/<pilot_id>/combined_selection.json
processed/vlm_candidate_v3/rule_filter/<pilot_id>/combined_rule_filter.json
processed/metadata/v3_candidate_samples_v0_1.jsonl
processed/metadata/v3_candidate_summary_v0_1.json
```

期望结果：

- VLM 在 `plan` 阶段说明 target/reject 部件；
- VLM 在 `part_select` 阶段只选择候选 ID，不输出坐标；
- 规则过滤后 `candidate_manifest.json` 中的 `default_selected_candidates` 被更新；
- build 阶段即使遇到空样本也不中断。

### 3.4 打开人工审查系统

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

如果人工审查体验仍然像“从零点点云”，说明当前 geometry fallback 不够，下一步应优先接 PartSLIP++，而不是恢复自然化渲染。

## 4. 扩大运行

小批通过后，先跑 300 条：

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

```bash
CUDA_VISIBLE_DEVICES=1,2 python MultiEEAffordance/tools/run_v3_pipeline.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml \
  --pilot-csv processed/metadata/v3_large_scale_review_queue_v0_1.csv \
  --samples processed/metadata/samples_v3_large_batch_v0_1.jsonl \
  --include-tasks pick_up,open_pull,press_push \
  --exclude-tasks lift_carry \
  --candidate-source partseg \
  --part-proposal-backend geometry \
  --stages views,plan,part_propose,render,part_select,part_filter,build \
  --allow-empty \
  --overwrite
```

全量运行时，重新构建不带 `--limit` 的队列即可。保留 `--common-sense-filter`，但同一物体任务下不可行的执行器仍会通过 `--executor-scope all --empty-policy review` 进入空标签确认流程。

## 5. PartSLIP++ 接入约定

后续真正接 PartSLIP++ 时，adapter 只需要满足 `propose_v3_part_candidates.py` 的输出协议：

```text
processed/vlm_candidate_v3/3d_candidates/<pilot_id>/
  candidates.npz
  candidate_manifest.json
```

`candidates.npz` 至少包含：

| key | 形状/类型 |
| --- | --- |
| `candidate_masks` | `[K,N] uint8` |
| `candidate_ids` | `[K]` |
| `candidate_names` | `[K]` |
| `candidate_families` | `[K]` |

`candidate_manifest.json` 中每个候选应包含：

| 字段 | 说明 |
| --- | --- |
| `candidate_id` | A/B/C... |
| `candidate_name` | 部件名或 adapter 输出名 |
| `candidate_family` | 如 `partslippp_part`、`handle_like_part`、`flat_panel_part` |
| `point_count` | 候选覆盖点数 |
| `point_fraction` | 占原始点云比例 |
| `recommended_executors` | 可选，给规则过滤使用 |
| `source` | 建议写 `partslippp` |

adapter 不能改变原始点云数量 `N`，不能输出代理点或渲染像素作为标注点。
