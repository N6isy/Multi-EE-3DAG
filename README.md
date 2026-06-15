# Multi-EE-3DAG

面向异构末端执行器的多标签 3D Affordance Grounding 研究原型。

## 当前数据服务器

从 2026-06-01 起，大规模中间文件、五任务下采样批次和网页人工审查统一在数据存储服务器执行：

```text
服务器：10.24.1.11
Python 包父目录：/home/lzq/data
数据根目录 / 代码包目录：/home/lzq/data/MultiEEAffordance
```

GitHub 仓库只管理代码、文档和小型元数据。大规模 `.npy`、`.npz`、候选目录、人工 refined mask 和审查日志保存在数据服务器，不提交到普通 Git 仓库。

当前规划的数据来源为：

```text
3D AffordanceNet：8000+ 五任务审查样本行
PartNet-Mobility：4000+ 五任务审查样本行
总计：约 1.2w 样本行
```

这里的数量是后续按任务和执行器组织的样本行目标，不等同于 raw object 数量。PartNet-Mobility 解压和转换流程见：

```text
MultiEEAffordance/docs/PartNet-Mobility数据解压与格式转换.md
```

PartNet-Mobility 当前只作为 3D AffordanceNet 的类别补充。转换脚本默认保留 `Box`、`Bucket`、`Cabinet`、`Camera`、`CoffeeMachine`、`Dispenser`、`Kettle`、`Lighter`、`Mouse`、`Oven`、`Phone`、`Pliers`、`Remote`、`Safe`、`Stapler`、`Suitcase`、`Switch`、`Toaster`、`Toilet`、`WashingMachine`、`Window` 共 21 类。

当前项目目标是从物体点云和任务指令出发，构建不同末端执行器在同一任务下的可操作区域标注：

```text
P, q -> [M_gripper, M_suction, M_hook, M_dexterous_hand]
```

其中 `P` 表示单个物体的 3D 点云，`q` 表示人工标注阶段采用的五任务之一：`lift`、`open`、`pull`、`press`、`push`。输出是四通道点级 mask：

| 通道 | 执行器 | 含义 |
| --- | --- | --- |
| 0 | `gripper` | 平行二指夹爪可夹持区域 |
| 1 | `suction` | 单吸盘可吸附区域 |
| 2 | `hook` | 单钩可插入、挂住、拉动或提拉区域 |
| 3 | `dexterous_hand` | 多指灵巧手任务相关操作区域 |

> 注意：本项目标注的是“任务相关的可操作接触区域”，不是物体上所有能被接触的表面。

## 当前阶段

项目目前已经完成五任务人工标注整理与合并，进入 AAAI 第一批 baseline 实验准备阶段。当前最终训练样本文件为：

```text
/home/lzq/data/MultiEEAffordance/processed/annotation_batches/final_5tasks/all_sources_5tasks_4exec_complete_aligned_posfixed.jsonl
```

该文件每一行表示一个 `object_id + task + executor` 组合。训练前需要先用 `training/prepare_final_5task_training_dataset.py` 合并成 `object_id + task -> [N,4] mask` 的 canonical manifest，并按 CAD asset 做 train/val/test split。

当前任务体系已经从旧候选生成任务切换到五任务人工标注体系：

| 旧候选任务 | 五任务人工标注任务 | 说明 |
| --- | --- | --- |
| `pick_up` | `lift` | 旧候选继续作为 lift 的 proposal |
| `open_pull` | `open` + `pull` | 一对多展开，不能直接重命名 |
| `press_push` | `press` + `push` | 一对多展开，不能直接重命名 |
| `lift_carry` | `lift` | 仅作为历史兼容任务 |

旧任务生成的候选、mask 和 pipeline 输出不得删除或覆盖。它们只作为五任务人工审查的候选 proposal，最终标签以 `reviewer_a/reviewer_b` 在网页中按五任务语义审查后的 refined mask 为准。

已经完成的内容包括：

- 基于 3D AffordanceNet full-shape 数据构建了第一版物体级样本。
- 完成了 61 条样本的网页端人工审查，并形成 cleaned v0.1 数据格式。
- 明确了四类末端执行器和五任务人工标注规范。
- 构建了候选区域生成、VLM 辅助筛选、规则过滤和网页人工审查流程。
- 当前正在验证 v3 pipeline：让 VLM 先判断目标部件和应排除部件，再生成更适合人工审查的 3D 候选区域。

当前优先级是执行最终训练数据接入和第一批 baseline：

```text
validate final 5task row-level JSONL
prepare canonical object-task [N,4] manifests
audit CAD asset-level split
train first PointNet baseline
```

训练侧只接受 `lift/open/pull/press/push`，不会读取旧复合任务。正式 split 以 CAD asset 为单位：3D AffordanceNet 中一个 `3danet_full_xxx` 原始 object shape/model 就是一个 CAD asset；如果没有额外 asset 字段，则 `source_asset_id=object_id`。同一 asset 派生出的所有 task、executor、mask 和重复标注样本必须进入同一个 split。

## 仓库结构

```text
Multi-EE-3DAG/
  README.md
  AGENT.md
  MultiEEAffordance/
    configs/                  # VLM / SAM2 / pipeline 配置
    docs/                     # 项目文档、标注规范、进度记录
    manifests/                # 数据 manifest 模板和中间清单
    processed/                # 已处理点云、mask、metadata、候选结果
    raw/                      # 原始数据目录，本仓库通常不上传大数据
    splits/                   # train / val / test split
    training/                 # 独立五任务训练 pipeline、配置和运行说明
    tools/                    # 主要脚本
    taxonomy.yaml             # 任务和执行器定义
    requirements-vlm.txt      # 远程 VLM 环境依赖参考
```

训练入口说明见：

```text
MultiEEAffordance/training/README.md
```

大文件数据、模型权重和远程运行结果通常不放进 GitHub，需要在本地或服务器端单独准备。

## 建议阅读顺序

如果你是第一次看这个项目，建议按下面顺序阅读：

1. `MultiEEAffordance/docs/双人协作标注README_5tasks.md`
   - 审查者本地标注操作手册：拉仓库、解压数据包、启动网页、选择身份、保存结果。
2. `MultiEEAffordance/docs/维护者协作标注README_5tasks.md`
   - 维护者准备候选、展开五任务、分包、回收和合并结果。
3. `MultiEEAffordance/docs/异构末端执行器标注规范.md`
   - 了解四类执行器和五任务标注标准。
4. `MultiEEAffordance/docs/项目进度日志.md`
   - 了解项目目前已经完成了什么、失败过什么、下一步做什么。
5. `MultiEEAffordance/docs/V3语义目标拒绝候选Pipeline设计与汇报.md`
   - 了解当前最新 v3 pipeline 为什么这样设计。
6. `MultiEEAffordance/tools/serve_v2_annotation_app.py`
   - 网页人工审查系统，支持候选组合选择和点级增删。
7. `MultiEEAffordance/tools/run_v3_pipeline.py`
   - 当前推荐的一键式 v3 候选生成流程。
8. `MultiEEAffordance/tools/run_v3_semantic_part_planner.py`
   - VLM 如何先输出目标部件和拒绝部件。
9. `MultiEEAffordance/tools/grow_v3_reject_aware_candidates.py`
   - 如何从 target seed 生长候选，并用 reject veto 排除错误区域。

## 主要脚本说明

| 脚本 | 作用 |
| --- | --- |
| `convert_partnet_mobility.py` | 解压 PartNet-Mobility，并转换为标准点云和 URDF link-level 部件 proposal |
| `build_partnet_5task_review_samples.py` | 将 PartNet-Mobility 物体级 proposal 直接转换为五任务、四执行器人工审查样本 |
| `render_vlm_friendly_views.py` | 生成多视角点云渲染图和 point-index map |
| `run_v3_semantic_part_planner.py` | 用 Qwen3-VL 输出 target parts 和 reject parts |
| `run_v3_target_reject_grounding.py` | 对 target/reject 进行 2D 粗定位 |
| `project_v3_grounding_to_3d.py` | 利用 point-index map 将 2D 区域回投到 3D 点云 |
| `grow_v3_reject_aware_candidates.py` | 从 target seed 生长候选，并排除 reject 区域 |
| `build_v3_candidate_masks.py` | 构建待人工审查的四通道 candidate mask |
| `run_v3_pipeline.py` | 串联 v3 全流程 |
| `serve_v2_annotation_app.py` | 网页端人工审查和点级精修工具 |
| `training/validate_reviewed_samples.py` | 标注完成后检查 refined samples、路径、reviewer 和 mask shape |
| `training/validate_final_5task_rows.py` | 检查最终清洗 JSONL 的 task、executor、路径、shape 和 positive count |
| `training/prepare_final_5task_training_dataset.py` | 将最终 `object-task-executor` JSONL 合并为训练用 `object-task -> [N,4]` manifest |
| `training/prepare_training_dataset.py` | 历史兼容入口：合并早期 reviewer refined samples，生成 canonical `[N,4]` mask 和 split |
| `training/audit_splits.py` | 审计 train/val/test 是否存在 CAD asset 或 object 泄漏 |
| `training/audit_annotation_consistency.py` | 统计双人重叠标注的一致性 |
| `training/collect_experiment_table.py` | 汇总实验指标为论文主表 CSV/JSON |

## 本地环境

如果只阅读代码和运行轻量脚本，建议使用 Python 3.11：

```bash
conda create -n multieeaffordance python=3.11 -y
conda activate multieeaffordance
pip install numpy pyyaml pillow matplotlib
```

如果要运行 Qwen3-VL、SAM2 或 Florence-2 等模型，建议在远程 GPU 服务器上单独配置环境，参考：

```text
MultiEEAffordance/requirements-vlm.txt
MultiEEAffordance/docs/Qwen3-VL+SAM2远程运行说明.md
```

## 本地/远程协作方式

当前项目采用：

```text
本地修改代码和文档 -> git push 到 GitHub -> 远程服务器 git pull -> 远程运行模型和批处理脚本
```

原因是远程服务器无法直接使用 Codex 辅助改代码，因此所有代码修改应先在本地完成，再同步到服务器运行。

## v3 pipeline 运行示例

旧任务候选生成仍使用 `pick_up/open_pull/press_push`。如果只做当前人工审查输入，推荐轻量运行 `part_propose,build`：

```bash
CUDA_VISIBLE_DEVICES=1,2 python MultiEEAffordance/tools/run_v3_pipeline.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml \
  --pilot-csv processed/metadata/v3_large_scale_review_queue_v0_1.csv \
  --samples processed/metadata/samples_v3_large_batch_v0_1.jsonl \
  --include-tasks pick_up,open_pull,press_push \
  --exclude-tasks lift_carry \
  --include-decisions all \
  --candidate-source partseg \
  --part-proposal-backend high_recall \
  --stages part_propose,build \
  --proposal-max-candidates 64 \
  --max-candidates 12 \
  --part-top-k 5 \
  --allow-empty \
  --overwrite
```

生成旧任务候选 samples 后，先展开为五任务人工审查 samples：

```bash
python MultiEEAffordance/tools/expand_legacy_tasks_to_5tasks.py \
  --input MultiEEAffordance/processed/metadata/v3_candidate_samples_v0_1.jsonl \
  --output MultiEEAffordance/processed/metadata/v3_candidate_samples_v0_2_5tasks.jsonl \
  --summary-json MultiEEAffordance/processed/metadata/v3_candidate_samples_v0_2_5tasks_summary.json \
  --overwrite
```

然后启动网页审查系统。网页打开后必须选择 `reviewer_a` 或 `reviewer_b`：

```bash
python MultiEEAffordance/tools/serve_v2_annotation_app.py \
  --dataset-root MultiEEAffordance \
  --samples processed/metadata/v3_candidate_samples_v0_2_5tasks.jsonl \
  --review-jsonl processed/metadata/v3_point_level_review_records.jsonl \
  --output-mask-root processed/vlm_candidate_v3/manual_refined_masks \
  --output-samples processed/metadata/v3_manual_refined_samples_v0_2_5tasks.jsonl \
  --port 8765 \
  --top-k-candidates 12
```

打开：

```text
http://127.0.0.1:8765/
```

## 人工审查时要做什么

人工审查不是简单确认模型输出，而是要判断候选区域是否真的满足当前任务和末端执行器机制。

审查时重点看：

- 该区域是否和任务 `q` 相关。
- 该执行器是否能通过自己的典型机制作用在该区域。
- 候选区域是否覆盖了目标部件。
- 是否混入了明显错误部位，例如 hook 选到剪刀刃、suction 选到孔洞边缘。
- 如果候选基本正确，但多了一些点，可以在网页里点级删除。
- 如果候选漏了一些点，可以在网页里点级补充。

自动生成的候选 mask 只能作为 candidate proposal，不能直接当作 ground truth。

## 当前重点样本

当前重点检查的 pilot 样本包括：

- `vlm_pilot_005`: Bag / lift_carry / hook
  - 检查 bag handle 是否能被识别为 hook 可挂接区域。
- `vlm_pilot_010`: Scissors / pick_up / hook
  - 检查剪刀指环、孔洞边界是否能作为 hook target。
  - 检查剪刀刃、刀尖、普通长边界是否被 reject。

这两个样本能较好反映当前 pipeline 是否真正理解“任务 + 末端执行器机制”，而不是只依赖几何显著性。

## 重要提醒

- `processed/` 中很多文件是中间结果或本地运行结果，GitHub 上不一定完整。
- `external/`、大模型权重和原始数据通常不上传。
- 当前 v3 pipeline 仍处于验证阶段，所有自动候选都需要人工审查。
- 旧任务候选只能作为五任务人工审查 proposal。
- 不要把 VLM/SAM2 或几何规则生成的 mask 直接作为最终标注。
- 修改标注规范或 pipeline 逻辑时，需要同步更新 `docs/项目进度日志.md`。

## 项目关键词

3D Affordance Grounding, heterogeneous end-effector, multi-label affordance, point cloud annotation, VLM-assisted annotation, Qwen3-VL, SAM2, human-in-the-loop labeling.
