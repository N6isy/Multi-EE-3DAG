# VLM 语义部件引导标注 Pipeline

更新时间：2026-05-15 15:02 +08:00

## 1. 设计动机

当前项目要构建的是物体级多执行器、多标签 3D affordance 数据集。初版 Qwen3-VL + SAM2 pipeline 直接让 VLM 在点云渲染图上输出 box / point，再交给 SAM2 生成 2D mask。但在 `vlm_pilot_005: Bag / lift_carry / hook` 上，该路线没有稳定识别出包体上方提手，输出为空。

这并不说明 VLM 没有价值，而是说明使用方式不合适。VLM 擅长语义理解、常识推理和任务-部件关系判断；不擅长在稀疏点云渲染图上直接给出精确像素坐标。因此新一版 pipeline 将 VLM 的角色从“像素坐标标注器”改为“语义部件规划器”。

核心原则：

```text
VLM 负责回答：要找什么部件、为什么这个部件适合当前任务和执行器。
Grounding/SAM2 负责回答：这个部件在 2D 图像哪里。
point_index map 负责回答：2D mask 对应真实 3D 点云中的哪些点。
几何规则负责校验和过滤，而不是单独决定正样本。
人工审查负责最终确认。
```

## 2. 初版失败原因

| 失败点 | 具体表现 | 本质原因 |
| --- | --- | --- |
| 稀疏点云图不像自然图像 | bag handle 在图中只是少量点构成的弧线 | 3D AffordanceNet full-shape 每个物体只有 2048 个点，细结构点数极少 |
| 目标区域占比太小 | VLM 多次回答 “no hookable structure visible” | handle 在整物体视图中占比太低，视觉显著性不足 |
| VLM 直接输出坐标不稳定 | box / point 经常落在背景或主体边缘 | Qwen3-VL 更擅长语义判断，不擅长在非自然图像上精确定位 |
| SAM2 不适合直接分割稀疏点 | 即使有点提示，也可能无法形成合理连续区域 | SAM2 面向自然图像连续区域，稀疏点云图缺少真实表面纹理 |
| 几何候选过于粗糙 | A/B/C 候选能找到上方结构，但语义边界仍不稳定 | 几何规则没有 VLM 的部件知识，容易混淆提手、普通上边缘和主体表面 |
| source dataset 没有 hook mask | 无法从已有 3D AffordanceNet mask 中直接修正 | full-shape 标签不包含明确的 hookable hole / ring / handle boundary |

结论：

```text
初版失败不是因为 VLM 没有用，而是因为把 VLM 放在了“直接输出像素坐标”的弱项上。
新一版应把 VLM 放回“语义部件识别 + 任务机制判断”的强项上。
```

## 3. 新版总体流程

```text
1. 从点云生成多视角 VLM-friendly 图像
   - dense render：给 VLM / grounding 模型看
   - silhouette / depth render：增强形状理解
   - zoom panel：突出候选或细结构
   - point_index map：保留真实 3D 点回投索引

2. Qwen3-VL 做语义部件规划
   输入：多视角图像、object_category、task、executor、标注规范
   输出：target_part_names、部件解释、reject_parts、uncertain_parts

3. 开放词汇 grounding
   输入：dense render + target_part_text
   可选模型：
   - GroundingDINO
   - Florence-2
   - 后续可测试其他开放词汇定位模型

4. SAM2 根据 grounding box 生成 2D mask

5. 2D mask 回投真实 3D 点云
   只使用 point_index map 或最近真实投影点，不把补密像素当成真实点

6. 多视角融合
   对真实 3D 点累计 vote，并结合 visibility / grounding confidence / VLM confidence

7. 几何与机制校验
   gripper / suction / hook / dexterous_hand 分别执行机制约束检查

8. 生成 candidate 3D mask

9. 网页人工审查
   accept / refine / reject / uncertain
```

## 4. 两套图像的分工

| 图像类型 | 目的 | 是否用于最终 3D 回投 |
| --- | --- | --- |
| dense render | 让 VLM / Grounding 模型看起来更接近连续物体 | 否 |
| silhouette | 辅助理解整体轮廓和细结构 | 否 |
| zoom panel | 放大 handle / ring / button 等小结构 | 否 |
| point_index map | 记录每个像素对应的真实点云 point id | 是 |

关键约束：

```text
可以给 VLM 看更连续、更友好的图；
但最终 3D mask 只能落到真实点云 P 的 N 个点上。
```

## 5. 模块职责

| 模块 | 负责 | 不负责 |
| --- | --- | --- |
| Qwen3-VL Part Planner | 部件识别、任务-执行器机制判断、生成 grounding 文本 | 精确像素坐标 |
| GroundingDINO / Florence-2 | 根据部件文本找 2D box | 判断机器人执行器机制 |
| SAM2 | 根据 box/point 生成 2D mask | 理解 hook/gripper/suction 语义 |
| point_index projection | 将 2D mask 映射到真实 3D 点 | 生成不存在的补密点 |
| geometry validator | 过滤明显不满足机制的候选 | 单独作为最终标注来源 |
| 人工审查 | 最终确认候选质量 | 被 VLM 或规则替代 |

## 6. pilot_005 的预期行为

样本：

```text
pilot_id: vlm_pilot_005
object_category: Bag
task: lift_carry
executor: hook
```

Qwen3-VL Part Planner 预期输出：

```json
{
  "target_part_names": ["bag handle", "top handle loop", "handle loop"],
  "grounding_queries": ["bag handle", "top handle loop", "hookable handle loop"],
  "mechanism_reason": "A hook can enter or catch the top handle loop and apply lifting force for lift_carry.",
  "reject_parts": ["bag body surface", "flat side panel", "ordinary top edge"]
}
```

如果 Qwen3-VL 仍然不能识别提手，则记录为 VLM failure case，而不是直接否定 hook 标签。此时可以进入人工指定部件文本或人工候选审查。

## 7. 输出文件规划

```text
processed/vlm_semantic_part/
  renders/
    {sample_id}/
      view_manifest.json
      {view}_dense.png
      {view}_silhouette.png
      {view}_selector.png
      {view}_point_index.npy
  part_plans/
    {pilot_id}/
      {view}_part_plan.json
      combined_part_plan.json
  grounded_2d/
    {pilot_id}/
      {view}_{query_id}_boxes.json
      {view}_{query_id}_mask.npy
      {view}_{query_id}_mask.png
  projected_3d/
    {pilot_id}_votes.npz
  fused_masks/
    {sample_id}_{pilot_id}_semantic_part_candidate.npy
  metadata/
    semantic_part_candidate_samples_v0_1.jsonl
```

## 8. 当前阶段的实现范围

当前先实现可运行骨架，不一开始全量跑：

1. 只处理 `vlm_pilot_005`。
2. Qwen3-VL part planner 可以真实运行，也支持 `--dry-run` / `--validate-only`。
3. GroundingDINO / Florence-2 先做接口骨架，允许用手写 boxes JSON 作为替代输入。
4. 2D-to-3D projection 和 fusion 先使用通用格式，保证后续接入真实 grounding 输出时不需要重写下游。
5. 所有输出均标记为 candidate proposal，不作为 ground truth。

## 9. 失败处理策略

| 情况 | 处理 |
| --- | --- |
| Qwen3-VL 识别不出目标部件 | 记录 `part_planner_failed`，可人工输入部件文本继续后续流程 |
| GroundingDINO / Florence-2 找不到 box | 记录 `grounding_failed`，不直接生成正例 |
| SAM2 mask 为空 | 记录 `segmentation_empty`，进入 uncertain |
| 2D mask 回投后点数过少 | 保留为 weak candidate，必须人工复核 |
| 几何 validator 判定机制不满足 | 标记 rejected 或 uncertain |

这条 pipeline 的目标不是自动生成最终标注，而是最大化利用 VLM 的语义能力产生更好的 candidate mask，并通过规则和人工审查保证数据集质量。

## 10. pilot_005 单样本运行顺序

先生成 VLM-friendly 图像：

```bash
python MultiEEAffordance/tools/render_vlm_friendly_views.py \
  --dataset-root MultiEEAffordance \
  --pilot-id vlm_pilot_005 \
  --overwrite
```

运行 Qwen3-VL 语义部件规划：

```bash
CUDA_VISIBLE_DEVICES=1,2 python MultiEEAffordance/tools/run_qwen3vl_part_planner.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml \
  --pilot-id vlm_pilot_005 \
  --overwrite
```

如果只是检查格式，可用 dry-run：

```bash
python MultiEEAffordance/tools/run_qwen3vl_part_planner.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml \
  --pilot-id vlm_pilot_005 \
  --dry-run \
  --overwrite
```

执行 grounding + SAM2。当前骨架先支持 `manual-json` 和 `dry-run`；后续在服务器上接入 GroundingDINO / Florence-2 adapter：

```bash
python MultiEEAffordance/tools/run_grounding_sam2.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml \
  --pilot-id vlm_pilot_005 \
  --dry-run \
  --box-mask-only \
  --overwrite
```

回投到 3D：

```bash
python MultiEEAffordance/tools/project_grounded_masks_to_3d.py \
  --dataset-root MultiEEAffordance \
  --pilot-id vlm_pilot_005 \
  --overwrite
```

融合成四通道候选 mask：

```bash
python MultiEEAffordance/tools/fuse_semantic_part_masks.py \
  --dataset-root MultiEEAffordance \
  --pilot-id vlm_pilot_005 \
  --overwrite
```

检查候选样本格式：

```bash
python MultiEEAffordance/tools/check_dataset.py \
  --dataset-root MultiEEAffordance \
  --samples processed/metadata/semantic_part_candidate_samples_v0_1.jsonl \
  --split-dir splits_semantic_part_candidates
```

进入网页人工复核：

```powershell
python MultiEEAffordance\tools\serve_review_app.py `
  --dataset-root MultiEEAffordance `
  --samples processed\metadata\semantic_part_candidate_samples_v0_1.jsonl `
  --review-csv processed\metadata\semantic_part_candidate_review_v0_1.csv `
  --host 127.0.0.1 `
  --port 8768 `
  --max-points 4096
```
