# Qwen3-VL + SAM2 候选标注流程图

记录日期：2026-05-12 +08:00

本文档只保留当前用于汇报的候选标注主流程。该流程从 pilot 样本读取开始，到 VLM/SAM2 生成 2D 候选 mask、回投融合为 3D 候选 mask，再进入网页人工审查和 metadata 更新。

```mermaid
flowchart TD
  F["读取 pilot 样本表<br/>vlm_pilot_samples_v0_1.csv"]

  F --> G["读取多视角渲染<br/>front/back/left/right/top/iso"]
  G --> H["Qwen3-VL 逐视角理解图像"]
  H --> I["输出 SAM2 prompts<br/>box / positive point / negative point"]
  I --> J["SAM2 生成每视角 2D binary mask"]

  J --> K["保存 VLM/SAM2 候选结果<br/>vlm_2d_masks + qwen3vl_sam2_responses"]
  K --> L["project_2d_masks_to_3d / build_vlm_pilot_candidates<br/>利用 point_index map 回投到 3D 点云"]
  L --> M["多视角融合<br/>生成候选 [N,4] mask"]
  M --> N["生成候选样本 metadata<br/>vlm_pilot_candidate_samples_v0_1.jsonl"]

  N --> O["本地网页复核<br/>serve_review_app.py"]
  O --> P["人工审查候选 mask"]
  P --> Q{"审查结论"}

  Q -- "通过" --> R["标记 checked / human_verified<br/>进入后续可用样本"]
  Q -- "需要修正" --> S["进入 refine 队列<br/>rule_refined / manual_refinement"]
  Q -- "信息不足" --> T["标记 uncertain<br/>等待补视角/补部件/人工确认"]
  Q -- "不符合机制" --> U["标记 rejected<br/>不进入正样本"]

  R --> V["更新 metadata / provenance / quality flag"]
  S --> V
  T --> V
  U --> V
```

## 汇报时需要强调的边界

- Qwen3-VL + SAM2 输出的是候选标注，不是最终 ground truth。
- 2D mask 回投到 3D 依赖多视角渲染阶段保存的 `point_index map`。
- 多视角融合后的 `[N,4]` mask 仍然需要人工审查。
- 人工审查结论分为通过、需要修正、信息不足和不符合机制四类，并写回 metadata 的 provenance 与 quality flag。
