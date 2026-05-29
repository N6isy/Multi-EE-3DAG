# 维护者协作标注 README NEW

更新时间：2026-05-29

当前维护者流程已经统一到五任务版本，正式文档请阅读：

```text
MultiEEAffordance/docs/维护者协作标注README_5tasks.md
```

本文件只作为兼容入口，避免旧对话或旧记录继续引用 `README_NEW` 时找不到文档。

当前关键约定：

- 人工标注任务：`lift`、`open`、`pull`、`press`、`push`。
- 旧候选任务：`pick_up`、`open_pull`、`press_push`、`lift_carry`。
- 旧候选到五任务映射：`pick_up -> lift`，`open_pull -> open + pull`，`press_push -> press + push`，`lift_carry -> lift`。
- 旧任务候选只能作为人工审查 proposal，不能作为五任务 ground truth。
- 统一 taxonomy 文件：`MultiEEAffordance/utils/task_taxonomy.py`。
- 当前 taxonomy 版本：`v0_2_5tasks`。
- 当前审查批次目录：`processed/annotation_batches/v0_2_5tasks/`。

维护者最常用的三个入口命令：

```bash
python MultiEEAffordance/tools/expand_legacy_tasks_to_5tasks.py \
  --input /home/lzq/data/MultiEEAffordance/processed/metadata/v3_candidate_samples_v0_1.jsonl \
  --output /home/lzq/data/MultiEEAffordance/processed/metadata/v3_candidate_samples_v0_2_5tasks.jsonl \
  --summary-json /home/lzq/data/MultiEEAffordance/processed/metadata/v3_candidate_samples_v0_2_5tasks_summary.json \
  --overwrite
```

```bash
python MultiEEAffordance/tools/package_annotation_batches_from_samples.py \
  --dataset-root /home/lzq/data/MultiEEAffordance \
  --input processed/metadata/v3_candidate_samples_v0_2_5tasks.jsonl \
  --batch-dir processed/annotation_batches/v0_2_5tasks \
  --reviewers reviewer_a,reviewer_b \
  --calibration-objects 0 \
  --archive-format tar.gz \
  --overwrite
```

```bash
python MultiEEAffordance/tools/serve_v2_annotation_app.py \
  --dataset-root MultiEEAffordance \
  --samples processed/annotation_batches/v0_2_5tasks/reviewer_a_samples.jsonl \
  --review-jsonl processed/annotation_batches/v0_2_5tasks/reviewer_a_review_records.jsonl \
  --output-mask-root processed/annotation_batches/v0_2_5tasks/manual_refined_masks_reviewer_a \
  --output-samples processed/annotation_batches/v0_2_5tasks/reviewer_a_refined_samples.jsonl \
  --port 8765 \
  --top-k-candidates 12
```
