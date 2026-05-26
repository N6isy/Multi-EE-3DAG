# Multi-EE-3DAG 文档索引

更新时间：2026-05-26

## 当前推荐命令速查

当前 v3 主线已经切换为自研高召回 3D part candidate generator。默认不要再跑旧的 `ground,project,grow` 候选生长链路，除非是在复现实验。

## 当前双人协作标注版本

当前建议把轻量自动候选 + 人工点级审查版本冻结为双人协作标注 v0.1：

- 稳定标注分支：`annotation/mvp-v0.1`
- 稳定标注标签：`annotation-mvp-v0.1`
- 候选生成研发分支：`dev/high-recall-candidate-v0.2`
- 双人协作输出目录：`processed/annotation_batches/v0_1/`

两位审查者统一使用中性身份 `reviewer_a` 和 `reviewer_b`。正式标注时不要让两个人写入同一个 `review_records.jsonl` 或同一个 refined mask 目录；应分别启动不同端口、写入不同输出路径，再在批次结束后合并。

双人协作标注优先阅读：

1. `双人协作标注README.md`
2. `标注版本冻结与研发分支说明.md`
3. `大规模人工审查与已审查数据集可视化.md`

其中 `双人协作标注README.md` 是审查者操作手册，包含样本分配、端口访问、页面操作、保存检查、冲突检测和每日交付；`标注版本冻结与研发分支说明.md` 是维护者版本管理手册，包含分支/tag、hotfix、批次目录和 v0.2 升级条件。

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

### 3. 启动人工审查系统

单人调试或本地验证可以使用下面命令。正式双人协作标注请优先使用 `双人协作标注README.md` 中的 `reviewer_a/reviewer_b` 独立输出路径。

```bash
python MultiEEAffordance/tools/serve_v2_annotation_app.py \
  --dataset-root MultiEEAffordance \
  --samples processed/metadata/v3_candidate_samples_v0_1.jsonl \
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
| `双人协作标注README.md` | 当前双人协作标注操作流程：分配样本、启动两个端口、分别保存、合并 release |
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
2. `标注版本冻结与研发分支说明.md`
3. `大规模人工审查与已审查数据集可视化.md`
4. `v3自研高召回3D候选生成器说明.md`
5. `异构末端执行器标注规范.md`
6. `候选区域格式说明.md`
7. 需要排查环境或服务器运行时，再看运行手册和报错记录。
