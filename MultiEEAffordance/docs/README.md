# Multi-EE-3DAG 文档索引

更新时间：2026-06-09

## 当前数据服务器与数据规模

从 2026-06-01 起，五任务下采样、网页审查系统和人工输出统一在数据存储服务器执行：

```text
服务器：10.24.1.11
数据根目录：/home/lzq/data/MultiEEAffordance
```

当前数据集目标口径：

```text
3D AffordanceNet：8000+ 五任务审查样本行
PartNet-Mobility：4000+ 五任务审查样本行
总计：约 1.2w 样本行
```

其中 PartNet-Mobility 的 `4000+` 是后续展开后的目标样本行，不是 raw object 数。解压、点云采样和 URDF link 候选转换流程见：

```text
PartNet-Mobility数据解压与格式转换.md
```

## 当前任务体系状态

当前项目同时保留两条概念链路：

1. 旧任务候选生成链路：`pick_up`、`open_pull`、`press_push`，以及历史兼容的 `lift_carry`。这些数据已经生成过，不能删除或覆盖。
2. 五任务人工标注链路：`lift`、`open`、`pull`、`press`、`push`。这是后续人工审查和新数据版本的主线。

旧任务到五任务的映射为：

| 旧任务 | 五任务 |
| --- | --- |
| `pick_up` | `lift` |
| `open_pull` | `open` + `pull` |
| `press_push` | `press` + `push` |
| `lift_carry` | `lift` |

旧任务候选只能作为五任务人工标注的 proposal，不能直接当作五任务 ground truth。统一任务定义集中在 `MultiEEAffordance/utils/task_taxonomy.py`，当前版本号为 `v0_2_5tasks`。

## 当前推荐命令速查

当前 v3 主线已经切换为自研高召回 3D part candidate generator。默认不要再跑旧的 `ground,project,grow` 候选生长链路，除非是在复现实验。

## 最终五任务训练实验入口

当前人工标注已经完成整理与合并。最终清洗后的训练样本文件为：

```text
/home/lzq/data/MultiEEAffordance/processed/annotation_batches/final_5tasks/all_sources_5tasks_4exec_complete_aligned_posfixed.jsonl
```

该 JSONL 每一行是 `object_id + task + executor`，训练前必须先压缩为 `object_id + task -> [N,4] mask` manifest。训练实验准备统一在数据服务器执行：

```text
服务器：10.24.1.11
Python 包父目录：/home/lzq/data
数据根目录 / 代码包目录：/home/lzq/data/MultiEEAffordance
```

训练侧只接受五任务人工标注结果，不接受旧任务候选作为 GT。正式训练前必须先完成：

1. `validate_final_5task_rows.py`：检查最终 JSONL 的五任务字段、执行器字段、路径、mask shape 和 positive count。
2. `prepare_final_5task_training_dataset.py --split-unit source_asset`：合并成 canonical object-task `[N,4]` manifest，并按 CAD asset 划分 train/val/test。
3. `audit_splits.py --fail-on-leakage`：确认 `asset_uid/object_id` 无泄漏。
4. `train.py/evaluate.py/collect_experiment_table.py`：训练、评估并汇总 AAAI 主表。

完整可执行命令见：

```text
../training/README.md
最终五任务训练数据接入README.md
AAAI投稿导向模型训练Pipeline规划.md
```

特别注意：3D AffordanceNet 中一个原始 object shape/model 就是一个 CAD asset。如果只有 `object_id=3danet_full_xxx`，则先把这个 `object_id` 作为 `source_asset_id`；同一 asset 派生出的所有 task、executor、mask 和重复标注样本必须进入同一个 split。

## 当前双人协作标注版本

当前建议把轻量自动候选 + 人工点级审查版本冻结为双人协作标注 v0.2 五任务版：

- 稳定标注分支：`annotation/mvp-v0.1`
- 稳定标注标签：`annotation-mvp-v0.1`
- 候选生成研发分支：`dev/high-recall-candidate-v0.2`
- 双人协作输出目录：`processed/annotation_batches/v0_2_5tasks/`

两位审查者统一使用中性身份 `reviewer_a` 和 `reviewer_b`。当前默认模式是服务器网页协作：维护者在 `10.24.1.11` 生成候选、拆分五任务批次并启动两个独立网页端口，审查者通过浏览器完成标注，结果直接保存在 `/home/lzq/data/MultiEEAffordance/processed/annotation_batches/`。GitHub + 本地数据包方式仅作为离线备用。

双人协作标注优先阅读：

1. `双人协作标注README_5tasks.md`
2. `维护者协作标注README_5tasks.md`
3. `标注版本冻结与研发分支说明.md`
4. `大规模人工审查与已审查数据集可视化.md`

其中 `双人协作标注README_5tasks.md` 是审查者操作手册，包含本地环境、数据包解压、页面操作、保存检查和每日交付；`维护者协作标注README_5tasks.md` 是维护者批次管理手册，包含原始 pkl/zip 数据说明、旧候选生成、五任务展开、分包、收包和合并；`标注版本冻结与研发分支说明.md` 是版本管理手册，包含分支/tag、hotfix、批次目录和后续研发隔离条件。

### 1. 先小批查看候选 overlay

用于先看候选区域是否覆盖 handle / loop / button / flat panel / small part 等目标部件：

```bash
python MultiEEAffordance/tools/run_v3_pipeline.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml \
  --pilot-csv processed/metadata/v3_large_scale_review_queue_v0_1.csv \
  --samples processed/metadata/samples_v3_large_batch_v0_1.jsonl \
  --include-tasks pick_up,open_pull,press_push \
  --exclude-tasks lift_carry \
  --candidate-source partseg \
  --part-proposal-backend high_recall \
  --stages views,part_propose,render \
  --limit 6 \
  --proposal-max-candidates 64 \
  --max-candidates 24 \
  --part-top-k 5 \
  --allow-empty \
  --overwrite
```

### 2. 跑完整候选生成与构建

用于进入 VLM 候选选择、规则过滤、构建待人工审查样本：

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

### 3. 展开为五任务人工标注 samples

旧任务候选生成完成后，不直接进入正式标注。先把旧任务候选 samples 展开成五任务 samples：

```bash
python MultiEEAffordance/tools/expand_legacy_tasks_to_5tasks.py \
  --input MultiEEAffordance/processed/metadata/v3_candidate_samples_v0_1.jsonl \
  --output MultiEEAffordance/processed/metadata/v3_candidate_samples_v0_2_5tasks.jsonl \
  --summary-json MultiEEAffordance/processed/metadata/v3_candidate_samples_v0_2_5tasks_summary.json \
  --overwrite
```

输出中的 `source_task/source_sample_id/task_taxonomy_version/task_split_source` 用于追溯旧任务候选来源。

### 4. 启动人工审查系统

单人调试或本地验证可以使用下面命令。正式双人协作标注请优先使用 `双人协作标注README_5tasks.md` 中的 `reviewer_a/reviewer_b` 独立输出路径，并在网页打开后选择当前 reviewer 身份。

```bash
python MultiEEAffordance/tools/serve_v2_annotation_app.py \
  --dataset-root MultiEEAffordance \
  --samples processed/metadata/v3_candidate_samples_v0_2_5tasks.jsonl \
  --review-jsonl processed/metadata/v3_point_level_review_records.jsonl \
  --output-mask-root processed/vlm_candidate_v3/manual_refined_masks \
  --output-samples processed/metadata/v3_manual_refined_samples_v0_1.jsonl \
  --port 8765 \
  --top-k-candidates 8 \
  --candidate-min-selected-votes 2
```

## 当前主线文档

优先阅读这些文档：

| 文档 | 用途 |
| --- | --- |
| `双人协作标注README_5tasks.md` | 五任务审查者操作流程：访问 `10.24.1.11` 网页、选择 reviewer 身份、点级审查、保存结果；本地数据包作为离线备用 |
| `维护者协作标注README_5tasks.md` | 五任务维护者批次管理流程：准备原始数据、旧候选生成、展开五任务、服务器端下采样、启动网页、收包合并 |
| `标注版本冻结与研发分支说明.md` | `annotation/mvp-v0.1` 稳定标注版本和 `dev/high-recall-candidate-v0.2` 研发分支约定 |
| `v3自研高召回3D候选生成器说明.md` | 当前 v3 主线：自研高召回 3D part candidate generator、VLM 候选选择、人工审查 |
| `大规模人工审查与已审查数据集可视化.md` | 大规模队列、审查系统、已审查数据集 release 和只读可视化 |
| `异构末端执行器标注规范.md` | gripper / suction / hook / dexterous_hand 的标注语义 |
| `候选区域格式说明.md` | candidate mask、manifest、`[N,4]` 格式约定 |

## 服务器与运行手册

| 文档 | 用途 |
| --- | --- |
| `本地Codex与远程服务器工作流.md` | 本地改代码、服务器运行、同步结果的协作流程 |
| `Qwen3-VL+SAM2远程运行说明.md` | Qwen3-VL / SAM2 相关环境与远程运行记录 |
| `报错问题记录.md` | 已遇到的环境、磁盘、pipeline 报错和处理记录 |

## 数据转换、审查与可视化

| 文档 | 用途 |
| --- | --- |
| `3D-AffordanceNet转换报告.md` | 3D AffordanceNet 转换过程和统计 |
| `PartNet-Mobility数据解压与格式转换.md` | 在 `10.24.1.11` 解压 PartNet-Mobility，转换 URDF link-level proposal，并直接生成五任务双人审查批次 |
| `可视化检查报告.md` | 数据和 mask 可视化检查记录 |
| `人工审查表字段说明.md` | 人工审查字段含义 |
| `v0.1人工审查总结.md` | 早期人工审查结论 |
| `10条Pilot人工审查工作清单.md` | 小规模 pilot 审查记录 |

## 设计与汇报材料

| 文档 | 用途 |
| --- | --- |
| `V3语义目标拒绝候选Pipeline设计与汇报.md` | 旧 target/reject seed-growth 方案的历史设计说明 |
| `VLM候选选择Pipeline_v2设计与汇报.md` | v2 候选选择方案和问题总结 |
| `v2模块化Pipeline与点级审查系统设计.md` | 点级审查系统和 v2 模块化方案 |
| `AAAI投稿导向模型训练Pipeline规划.md` | AAAI 训练实验计划：asset-level split、人工一致性、empty/feasibility 建模、baseline 和消融 |
| `最终五任务训练数据接入README.md` | 标注合并完成后的最终 JSONL 接入训练：校验、manifest 生成、split 审计、baseline 训练和评估 |
| `AAAI投稿时间规划_2026-06-01.md` | 从 2026-06-01 开始的投稿排期：标注、训练、实验、论文写作、投稿材料和风险降级策略 |
| `../training/README.md` | 独立五任务训练目录操作手册：数据合并、训练环境、服务器运行、baseline 训练和评估 |
| `Qwen3-VL+SAM2候选标注流程图.md` | 早期 Qwen3-VL + SAM2 流程图 |
| `Qwen3-VL+SAM2候选标注与标注规范汇报稿.md` | 汇报稿草案 |
| `选题思路跟进报告.md` | 研究问题和选题思路跟进 |

## 历史实验与参考

这些文档记录过往探索，不代表当前推荐主线：

| 文档 | 说明 |
| --- | --- |
| `v3部件分割候选与PartSLIP++接入说明.md` | PartSLIP++ adapter 和外部模型接入记录，当前不再作为推荐主线 |
| `VLM试验运行手册.md` | 早期 VLM 多视角试验命令和排查 |
| `VLM语义部件引导标注Pipeline.md` | 早期 VLM 语义部件引导方案 |
| `VLM多视角试验设计.md` | 早期多视角 VLM 方案 |
| `协作者阅读论文清单.md` | 阅读材料列表 |

## 推荐阅读顺序

1. `双人协作标注README.md`
2. `维护者协作标注README.md`
3. `标注版本冻结与研发分支说明.md`
4. `大规模人工审查与已审查数据集可视化.md`
5. `v3自研高召回3D候选生成器说明.md`
6. `异构末端执行器标注规范.md`
7. `候选区域格式说明.md`
8. 需要排查环境或服务器运行时，再看运行手册和报错记录。
