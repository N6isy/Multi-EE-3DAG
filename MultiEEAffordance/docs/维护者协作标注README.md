# 维护者协作标注 README

更新时间：2026-05-27

本文档只面向维护者。维护者负责准备数据、生成自动候选、拆分标注批次、发给两位审查者、收回结果、合并并发布已审查数据集。

审查者不需要理解整条 v3 pipeline，也不需要运行 VLM 或 GPU 推理。审查者只需要按照 `双人协作标注README.md` 本地打开网页做点级审查。维护者要保证他们拿到的数据包可以直接运行。

当前协作身份统一使用：

- `reviewer_a`
- `reviewer_b`

不要把真实姓名写进路径名、文件名或脚本参数。这样合并和排查时更稳定。

## 0. 维护者到底要做什么

维护者的工作可以理解成 6 件事：

1. 准备原始数据。
2. 把原始数据转换成项目内部样本格式。
3. 生成一批需要人工审查的样本队列。
4. 跑轻量自动候选，生成网页能读取的候选 mask。
5. 把队列拆成 `reviewer_a` 和 `reviewer_b` 两份本地数据包。
6. 收回两份结果，检查、合并、生成已审查数据集。

推荐流程是：

```text
raw 3D AffordanceNet
  -> convert_3d_affordancenet.py
  -> build_samples_jsonl.py
  -> build_large_scale_review_queue.py
  -> run_v3_pipeline.py
  -> split reviewer packages
  -> reviewers local annotation
  -> collect reviewer result packages
  -> merge reviewed release
```

## 1. 当前原始数据在哪里

服务器上当前原始数据已经解压在：

```text
/home/lzq/Multi-EE-3DAG/MultiEEAffordance/raw/3d_affordancenet/full_shape_train_data.pkl
/home/lzq/Multi-EE-3DAG/MultiEEAffordance/raw/3d_affordancenet/full_shape_val_data.pkl
```

原始 zip 也建议保留：

```text
/home/lzq/Multi-EE-3DAG/MultiEEAffordance/raw/3d_affordancenet/full-shape.zip
```

这里需要注意一个实现细节：

当前 `MultiEEAffordance/tools/convert_3d_affordancenet.py` 是直接读取 `full-shape.zip`，然后从 zip 内部找：

```text
full_shape_train_data.pkl
full_shape_val_data.pkl
```

所以即使 pkl 文件已经解压出来，当前转换命令仍然应该使用：

```bash
--zip raw/3d_affordancenet/full-shape.zip
```

不要直接把 `--zip` 改成：

```text
raw/3d_affordancenet/full_shape_val_data.pkl
```

否则当前脚本会把 pkl 当 zip 打开，导致报错。

如果后面只保留了解压后的 pkl、没有保留 `full-shape.zip`，有两种选择：

1. 给 `convert_3d_affordancenet.py` 增加 `--pkl` 参数，让它直接读取 pkl。
2. 重新打包一个满足当前脚本预期的 zip。

临时重新打包命令如下：

```bash
cd /home/lzq/Multi-EE-3DAG/MultiEEAffordance/raw/3d_affordancenet
zip -j full-shape.zip full_shape_train_data.pkl full_shape_val_data.pkl
```

当前建议保留 zip，不急着改 converter。

## 2. 这些 pkl 是不是当前需要的全部数据

对当前这条 3D AffordanceNet full-shape 标注流程来说，需要的数据主要就是：

```text
full_shape_train_data.pkl
full_shape_val_data.pkl
```

其中：

- `train` 是训练 split。
- `val` 是验证 split。
- 当前人工审查批次一般先从 `val` 开始做，因为体量更可控，适合先验证流程。

但这不等于整个项目所有可能用到的数据都只有这两个文件。项目里还有其他来源或未来可能接入的数据，例如 PartNet-Mobility。它们不属于当前 3D AffordanceNet full-shape 批次的主输入。

当前维护者可以先把目标明确为：

```text
先基于 3D AffordanceNet full-shape 的 val split，生成一版稳定可审查数据集。
```

流程稳定后，再扩大到 train split 或其他数据源。

## 3. 维护者环境要求

维护者需要两类环境。

第一类是普通 Python 环境，用于数据转换、队列生成、拆包、合并：

```text
Python
numpy
项目脚本
```

第二类是带 GPU 和 VLM 依赖的环境，用于跑自动候选：

```text
Qwen3-VL
torch
CUDA
configs/qwen3vl_sam2_pilot.yaml 中配置的模型路径
```

审查者本地不需要第二类环境。

维护者在服务器上操作时，建议固定进入项目根目录：

```bash
cd /home/lzq/Multi-EE-3DAG
```

后续所有命令都默认从这个目录执行。

## 4. 推荐目录结构

当前建议把人工标注批次统一放在：

```text
MultiEEAffordance/processed/annotation_batches/v0_1/
```

推荐结构：

```text
processed/annotation_batches/v0_1/
  batch_manifest.json
  batch_summary.json
  reviewer_a_samples.jsonl
  reviewer_b_samples.jsonl
  reviewer_a_review_records.jsonl
  reviewer_b_review_records.jsonl
  reviewer_a_refined_samples.jsonl
  reviewer_b_refined_samples.jsonl
  manual_refined_masks_reviewer_a/
  manual_refined_masks_reviewer_b/
  packages/
    reviewer_a_annotation_package_v0_1.tar.gz
    reviewer_b_annotation_package_v0_1.tar.gz
  returned/
    reviewer_a_result_YYYYMMDD.tar.gz
    reviewer_b_result_YYYYMMDD.tar.gz
  merged/
    merged_review_records.jsonl
    merged_refined_samples.jsonl
    conflict_samples.jsonl
    reviewed_dataset_summary.json
```

原则：

- Git 管代码和文档。
- `raw/`、`processed/`、`.npy`、`.npz`、大压缩包不要直接提交到普通 Git。
- 给审查者的数据包和审查者回传的结果包都放到批次目录，方便以后复现。

## 5. 第一步：把原始数据转换成项目样本

先用 `val` split 做一批。下面命令会从 `full-shape.zip` 读取 `full_shape_val_data.pkl`，并把点云、弱 mask、manifest 写到项目内部目录。

```bash
python MultiEEAffordance/tools/convert_3d_affordancenet.py \
  --dataset-root MultiEEAffordance \
  --zip raw/3d_affordancenet/full-shape.zip \
  --source-split val \
  --target-split val \
  --tasks pick_up,open_pull,press_push \
  --max-per-category 20 \
  --points-dir processed/points/3d_affordancenet_full_shape_val_batch_v3 \
  --candidate-dir processed/candidates/3d_affordancenet_full_shape_val_batch_v3 \
  --mask-dir processed/masks/3d_affordancenet_full_shape_val_batch_v3 \
  --manifest manifests/3d_affordancenet_full_shape_val_batch_v3_manifest.jsonl \
  --summary processed/metadata/3d_affordancenet_full_shape_val_batch_v3_summary.json \
  --overwrite
```

你应该重点看输出里的这些字段：

```text
objects_loaded
objects_converted
samples_written
tasks
categories
feasible_counts_by_task
```

含义：

- `objects_loaded`：原始 pkl 里读到了多少个物体。
- `objects_converted`：实际转换了多少个物体。
- `samples_written`：写出了多少个物体-任务样本。
- `tasks`：每个任务各写出多少条。
- `categories`：每个物体类别各写出多少个物体。
- `feasible_counts_by_task`：弱标签阶段认为哪些任务和执行器可能有正例。

如果 `samples_written` 明显为 0，通常是路径、split 或任务过滤参数错了。

## 6. 第二步：生成统一 samples JSONL

转换后的 manifest 还不是审查系统直接使用的最终样本表。需要再构建 `samples_v3_large_batch_v0_1.jsonl`。

```bash
python MultiEEAffordance/tools/build_samples_jsonl.py \
  --dataset-root MultiEEAffordance \
  --manifest manifests/3d_affordancenet_full_shape_val_batch_v3_manifest.jsonl \
  --output processed/metadata/samples_v3_large_batch_v0_1.jsonl \
  --default-split val \
  --default-quality weak \
  --write-splits \
  --split-dir splits_v3_large_batch
```

输出里要看：

```text
samples_written
splits
```

正常情况下，`samples_written` 应该和上一步的 `samples_written` 一致。

## 7. 第三步：生成人工审查队列

这一步决定哪些样本进入这次人工审查批次。

当前任务策略：

- 保留：`pick_up,open_pull,press_push`
- 删除：`lift_carry`

原因是 `lift_carry` 和 `pick_up` 在当前定义里过于接近，先不作为独立任务推进。

示例命令：

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
  --limit-strategy round_robin_category_task_executor \
  --limit 300 \
  --overwrite
```

这里几个参数要理解清楚：

- `--executor-scope all`：同一个物体和任务下，不只保留原始弱标签中有正例的执行器，也保留其他执行器组合。
- `--empty-policy review`：即使某个任务-执行器组合没有候选区域，也保留为待审查空样本。这样才能体现“这个执行器不适合这个任务”。
- `--common-sense-filter`：先过滤明显不合理的物体-任务组合，例如某些物体根本不存在可 open_pull 的结构。
- `--limit 300`：这次先抽 300 条做批次。全量跑时可以去掉或调大。

输出里要看：

```text
rows
counts_by_task
counts_by_executor
counts_by_decision
counts_by_category
skipped
```

如果 `counts_by_category` 只出现很少类别，说明队列抽样策略或 limit 可能偏向了前几个类别，需要调整 `--limit-strategy` 或增大 `--limit`。

## 8. 第四步：先小批 smoke test

不要一上来全量跑 VLM。先用 3 到 5 条样本确认 pipeline 能跑完。

```bash
CUDA_VISIBLE_DEVICES=1,2 python MultiEEAffordance/tools/run_v3_pipeline.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml \
  --pilot-csv processed/metadata/v3_large_scale_review_queue_v0_1.csv \
  --samples processed/metadata/samples_v3_large_batch_v0_1.jsonl \
  --include-tasks pick_up,open_pull,press_push \
  --exclude-tasks lift_carry \
  --candidate-source partseg \
  --part-proposal-backend high_recall \
  --stages views,plan,part_propose,render,part_select,part_filter,build \
  --limit 5 \
  --proposal-max-candidates 64 \
  --max-candidates 12 \
  --part-top-k 5 \
  --allow-empty \
  --overwrite
```

检查重点：

1. `views` 是否正常输出多视角图。
2. `part_propose` 是否能产生候选，而不是大量 `candidate_count=0`。
3. `part_select` 是否只选择候选 ID，不直接生成坐标。
4. `build` 是否能生成待审查 samples。
5. 空样本是否能被保留下来，而不是让 pipeline 崩溃。

如果小批都不稳定，不要继续全量跑。

## 9. 第五步：正式生成自动候选

小批通过后，再跑完整队列。

```bash
CUDA_VISIBLE_DEVICES=1,2 python MultiEEAffordance/tools/run_v3_pipeline.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml \
  --pilot-csv processed/metadata/v3_large_scale_review_queue_v0_1.csv \
  --samples processed/metadata/samples_v3_large_batch_v0_1.jsonl \
  --include-tasks pick_up,open_pull,press_push \
  --exclude-tasks lift_carry \
  --candidate-source partseg \
  --part-proposal-backend high_recall \
  --stages views,plan,part_propose,render,part_select,part_filter,build \
  --proposal-max-candidates 64 \
  --max-candidates 12 \
  --part-top-k 5 \
  --allow-empty \
  --overwrite
```

维护者要记录这次运行的：

```text
运行日期
git commit hash
命令完整文本
输入 samples 路径
输入 review queue 路径
输出 candidate samples 路径
错误样本列表
```

建议写到：

```text
processed/annotation_batches/v0_1/batch_manifest.json
```

## 10. 第六步：检查候选质量

自动候选不是最终标注。维护者至少要抽查几类样本：

```text
Bag / pick_up / gripper
Faucet / open_pull / dexterous_hand
Scissors / pick_up / hook
Door / open_pull / suction
Keyboard / press_push / dexterous_hand
Mug / pick_up / suction
```

检查时重点看：

- 自动候选有没有完全空掉。
- 自动候选有没有把大面积主体都选成正例。
- 小部件是否保留下来了，例如 button、handle、ring、knob、switch。
- 对 hook 来说，blade、普通边缘、长杆尖端不应被默认认为可 hook。
- 对 suction 来说，候选应更偏向平整可吸附面。
- 对 dexterous_hand 来说，候选可以比 gripper 更宽，但不能把完全无任务意义的主体都选进去。

如果自动候选质量一般，也可以继续进入人工审查；但要保证网页里有足够候选可选，并且人工可以方便增删点。

## 11. 第七步：拆分 reviewer_a / reviewer_b 样本

拆分原则：

1. 两个人不要写同一个输出文件。
2. 尽量按物体分组拆分，不要把同一个 object 的多个任务-执行器组合分散给不同人。
3. 可以保留少量重叠样本用于校准，但重叠样本合并时必须进入冲突列表，不自动覆盖。

推荐输出：

```text
processed/annotation_batches/v0_1/reviewer_a_samples.jsonl
processed/annotation_batches/v0_1/reviewer_b_samples.jsonl
```

如果当前没有专门拆分脚本，可以先用一个简单规则：

- 按 `sample_id` 去重后排序。
- 奇数 object 给 `reviewer_a`。
- 偶数 object 给 `reviewer_b`。
- 每人额外给 5 条相同的 calibration samples。

拆分后检查：

```text
reviewer_a 样本数
reviewer_b 样本数
重叠 sample_key 数
每个 reviewer 的任务分布
每个 reviewer 的执行器分布
```

## 12. 第八步：给审查者准备本地数据包

审查者本地需要三类东西：

1. GitHub 仓库代码。
2. 自己那份 `reviewer_x_samples.jsonl`。
3. `reviewer_x_samples.jsonl` 引用到的点云、mask、candidate 文件。

数据包解压后，建议仍保持项目目录结构：

```text
Multi-EE-3DAG/
  MultiEEAffordance/
    processed/
      annotation_batches/v0_1/reviewer_a_samples.jsonl
      points/...
      masks/...
      vlm_candidate_v3/...
```

这样审查网页里的相对路径不用改。

维护者不要只发 `reviewer_a_samples.jsonl`。如果不带点云和候选文件，审查者本地网页会找不到 `.npy/.npz`。

## 13. 第九步：打包 reviewer 数据

如果批次较小，可以直接打包必要目录。示例：

```bash
tar -czf processed/annotation_batches/v0_1/packages/reviewer_a_annotation_package_v0_1.tar.gz \
  MultiEEAffordance/processed/annotation_batches/v0_1/reviewer_a_samples.jsonl \
  MultiEEAffordance/processed/points/3d_affordancenet_full_shape_val_batch_v3 \
  MultiEEAffordance/processed/masks/3d_affordancenet_full_shape_val_batch_v3 \
  MultiEEAffordance/processed/vlm_candidate_v3
```

`reviewer_b` 同理：

```bash
tar -czf processed/annotation_batches/v0_1/packages/reviewer_b_annotation_package_v0_1.tar.gz \
  MultiEEAffordance/processed/annotation_batches/v0_1/reviewer_b_samples.jsonl \
  MultiEEAffordance/processed/points/3d_affordancenet_full_shape_val_batch_v3 \
  MultiEEAffordance/processed/masks/3d_affordancenet_full_shape_val_batch_v3 \
  MultiEEAffordance/processed/vlm_candidate_v3
```

如果数据包过大，后续应该写一个精确打包脚本，只复制 `samples.jsonl` 实际引用到的文件。当前第一版可以先以稳定为主，允许包稍大一些。

## 14. 第十步：审查者本地怎么运行

维护者给审查者的 README 中应明确告诉他们：

`reviewer_a` 本地运行：

```bash
python MultiEEAffordance/tools/serve_v2_annotation_app.py \
  --dataset-root MultiEEAffordance \
  --samples processed/annotation_batches/v0_1/reviewer_a_samples.jsonl \
  --review-jsonl processed/annotation_batches/v0_1/reviewer_a_review_records.jsonl \
  --output-mask-root processed/annotation_batches/v0_1/manual_refined_masks_reviewer_a \
  --output-samples processed/annotation_batches/v0_1/reviewer_a_refined_samples.jsonl \
  --port 8765 \
  --top-k-candidates 8
```

`reviewer_b` 本地运行：

```bash
python MultiEEAffordance/tools/serve_v2_annotation_app.py \
  --dataset-root MultiEEAffordance \
  --samples processed/annotation_batches/v0_1/reviewer_b_samples.jsonl \
  --review-jsonl processed/annotation_batches/v0_1/reviewer_b_review_records.jsonl \
  --output-mask-root processed/annotation_batches/v0_1/manual_refined_masks_reviewer_b \
  --output-samples processed/annotation_batches/v0_1/reviewer_b_refined_samples.jsonl \
  --port 8765 \
  --top-k-candidates 8
```

审查者本地各跑各的，所以两个人都用 `8765` 没问题。只有在同一台机器上同时开两个审查网页时，才需要一个用 `8765`、另一个用 `8766`。

## 15. 第十一步：审查者每天要交付什么

维护者应要求每天回传这些文件：

```text
processed/annotation_batches/v0_1/reviewer_x_review_records.jsonl
processed/annotation_batches/v0_1/reviewer_x_refined_samples.jsonl
processed/annotation_batches/v0_1/manual_refined_masks_reviewer_x/
```

其中：

- `review_records.jsonl`：每次点击保存都会追加一条审查记录。
- `refined_samples.jsonl`：当前已经完成审查的样本索引。
- `manual_refined_masks_reviewer_x/`：真正的人工修正后 mask。

如果只回传 JSONL、不回传 mask 目录，结果是不完整的。

## 16. 第十二步：维护者收包后检查

收到结果包后，先不要直接覆盖已有目录。建议放到：

```text
processed/annotation_batches/v0_1/returned/
```

解压到临时目录后检查：

```text
review_records 是否存在
refined_samples 是否存在
manual_refined_masks 是否存在
mask 文件数量是否大于 0
JSONL 是否每行都是合法 JSON
样本数是否和审查者当天报告一致
```

可以用下面命令快速看行数：

```bash
wc -l processed/annotation_batches/v0_1/reviewer_a_review_records.jsonl
wc -l processed/annotation_batches/v0_1/reviewer_a_refined_samples.jsonl
find processed/annotation_batches/v0_1/manual_refined_masks_reviewer_a -name "*.npy" | wc -l
```

Windows 本地检查可用 PowerShell：

```powershell
(Get-Content MultiEEAffordance\processed\annotation_batches\v0_1\reviewer_a_review_records.jsonl).Count
(Get-Content MultiEEAffordance\processed\annotation_batches\v0_1\reviewer_a_refined_samples.jsonl).Count
(Get-ChildItem MultiEEAffordance\processed\annotation_batches\v0_1\manual_refined_masks_reviewer_a -Filter *.npy -Recurse).Count
```

## 17. 第十三步：处理两人冲突

如果同一个 `sample_key` 被两个人都标了，不要自动覆盖。

冲突样本应写入：

```text
processed/annotation_batches/v0_1/merged/conflict_samples.jsonl
```

冲突处理原则：

- 如果两人都标记为空，并且理由一致，可以合并为空。
- 如果一人为空、一人有正例，需要复审。
- 如果两人都有正例但区域差异很大，需要复审。
- 如果两人差异很小，可以保留更干净、更符合标注规范的一版。

## 18. 第十四步：生成已审查数据集 release

合并后建议输出：

```text
processed/annotation_batches/v0_1/merged/merged_refined_samples.jsonl
processed/annotation_batches/v0_1/merged/reviewed_dataset_summary.json
processed/reviewed_dataset/v0_1/
```

已审查数据集至少要包含：

- 原始点云路径。
- `[N,4]` refined mask 路径。
- object category。
- task。
- executor。
- review status。
- reviewer id。
- review decision。
- 是否为空样本。
- 如果为空，空样本原因。

发布前必须抽查：

```text
pos=0 样本
正例点数异常大的样本
正例点数异常小的样本
hook 选到 blade/普通边缘的样本
suction 选到非平面/边缘的样本
open_pull 选到主体而不是把手/开关的样本
```

## 19. 每日维护者检查清单

每天建议记录一次：

```text
日期
reviewer_a 完成多少条
reviewer_b 完成多少条
今天新增 refined mask 数量
今天 pos=0 数量
今天发现的可疑样本
今天是否有网页或路径报错
是否已备份当天结果包
```

建议写到：

```text
processed/annotation_batches/v0_1/daily_logs/YYYYMMDD.md
```

## 20. 常见问题

### 20.1 已经解压 pkl 了，为什么还要用 zip

因为当前 converter 的实现就是读 zip 内部的 pkl entry。解压出来的 pkl 可以留作备用，但当前命令不要直接指向 pkl。

### 20.2 审查者网页打开后点云不显示

优先检查：

1. 数据包是否解压在仓库根目录下。
2. `--dataset-root MultiEEAffordance` 是否正确。
3. `reviewer_x_samples.jsonl` 中引用的 `.npy/.npz` 文件是否真实存在。
4. 是否只发了 samples JSONL，没有发 points/masks/candidates。

### 20.3 为什么有些样本 pos=0

这是允许的。对于某个物体、任务、执行器组合，如果没有合适区域，应该保留空样本。它能表达“这个执行器不适合这个任务或物体”。

但 `pos=0` 需要人工确认，不应该全靠自动结果。

### 20.4 自动候选很差，还能标吗

可以，但效率会下降。当前版本的定位是轻量自动候选 + 人工点级审查。自动候选只是辅助，不是最终答案。

如果某一类样本自动候选持续很差，维护者应记录下来，放到后续 `dev/high-recall-candidate-v0.2` 研发分支改进，不要在稳定标注分支里频繁大改。

### 20.5 什么东西不能提交到 Git

不要提交：

```text
raw/3d_affordancenet/full-shape.zip
raw/3d_affordancenet/full_shape_train_data.pkl
raw/3d_affordancenet/full_shape_val_data.pkl
processed/points/
processed/masks/
processed/candidates/
processed/vlm_candidate_v3/
processed/annotation_batches/
external/
*.npy
*.npz
*.pkl
*.tar.gz
```

这些都属于大文件或生成结果。GitHub 仓库只放代码、文档、少量小型配置文件。

## 21. 推荐维护节奏

第一阶段：10 条样本校准。

```text
两人各标 5 条，另外 5 条重叠校准。
```

第二阶段：50 条样本试运行。

```text
检查两人标注速度、网页稳定性、pos=0 比例、候选质量。
```

第三阶段：300 条样本正式小批。

```text
每天收包，每 50 或 100 条生成一次 summary。
```

第四阶段：扩大到更多类别和 train split。

```text
只有当前流程稳定后再扩大，不要在自动候选还频繁崩溃时直接全量跑。
```

## 22. 维护者最终要交付什么

一个完整批次结束后，维护者应能交付：

```text
1. 本批次输入样本列表
2. reviewer_a / reviewer_b 原始审查记录
3. 合并后的 refined samples
4. 合并后的 refined masks
5. conflict samples
6. reviewed dataset summary
7. 本批次运行命令和 git commit hash
8. 已知问题列表
```

这样后续无论是继续标注、训练模型，还是写论文实验部分，都能追溯这批数据是怎么来的。
