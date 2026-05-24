# Multi-EE-3DAG

面向异构末端执行器的多标签 3D Affordance Grounding 研究原型。

当前项目目标是从物体点云和任务指令出发，构建不同末端执行器在同一任务下的可操作区域标注：

```text
P, q -> [M_gripper, M_suction, M_hook, M_dexterous_hand]
```

其中 `P` 表示单个物体的 3D 点云，`q` 表示任务，例如 `pick_up`、`lift_carry`、`open_pull`、`press_push`。输出是四通道点级 mask：

| 通道 | 执行器 | 含义 |
| --- | --- | --- |
| 0 | `gripper` | 平行二指夹爪可夹持区域 |
| 1 | `suction` | 单吸盘可吸附区域 |
| 2 | `hook` | 单钩可插入、挂住、拉动或提拉区域 |
| 3 | `dexterous_hand` | 多指灵巧手任务相关操作区域 |

> 注意：本项目标注的是“任务相关的可操作接触区域”，不是物体上所有能被接触的表面。

## 当前阶段

项目目前处于数据集原型和候选标注流程验证阶段，还不是最终可直接训练的大规模数据集。

已经完成的内容包括：

- 基于 3D AffordanceNet full-shape 数据构建了第一版物体级样本。
- 完成了 61 条样本的网页端人工审查，并形成 cleaned v0.1 数据格式。
- 明确了四类末端执行器、四类任务下的中文标注规范。
- 构建了候选区域生成、VLM 辅助筛选、规则过滤和网页人工审查流程。
- 当前正在验证 v3 pipeline：让 VLM 先判断目标部件和应排除部件，再生成更适合人工审查的 3D 候选区域。

当前最重要的工作不是训练模型，而是把“候选区域生成 + 人工审查”这条数据构建链路跑稳定。如果小规模 pilot 样本效果较好，下一步会开始第一批正式数据标注。

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
    tools/                    # 主要脚本
    taxonomy.yaml             # 任务和执行器定义
    requirements-vlm.txt      # 远程 VLM 环境依赖参考
```

大文件数据、模型权重和远程运行结果通常不放进 GitHub，需要在本地或服务器端单独准备。

## 建议阅读顺序

如果你是第一次看这个项目，建议按下面顺序阅读：

1. `MultiEEAffordance/docs/异构末端执行器标注规范.md`
   - 了解四类执行器和四类任务的标注标准。
2. `MultiEEAffordance/docs/项目进度日志.md`
   - 了解项目目前已经完成了什么、失败过什么、下一步做什么。
3. `MultiEEAffordance/docs/V3语义目标拒绝候选Pipeline设计与汇报.md`
   - 了解当前最新 v3 pipeline 为什么这样设计。
4. `MultiEEAffordance/tools/serve_v2_annotation_app.py`
   - 网页人工审查系统，支持候选组合选择和点级增删。
5. `MultiEEAffordance/tools/run_v3_pipeline.py`
   - 当前推荐的一键式 v3 候选生成流程。
6. `MultiEEAffordance/tools/run_v3_semantic_part_planner.py`
   - VLM 如何先输出目标部件和拒绝部件。
7. `MultiEEAffordance/tools/grow_v3_reject_aware_candidates.py`
   - 如何从 target seed 生长候选，并用 reject veto 排除错误区域。

## 主要脚本说明

| 脚本 | 作用 |
| --- | --- |
| `render_vlm_friendly_views.py` | 生成多视角点云渲染图和 point-index map |
| `run_v3_semantic_part_planner.py` | 用 Qwen3-VL 输出 target parts 和 reject parts |
| `run_v3_target_reject_grounding.py` | 对 target/reject 进行 2D 粗定位 |
| `project_v3_grounding_to_3d.py` | 利用 point-index map 将 2D 区域回投到 3D 点云 |
| `grow_v3_reject_aware_candidates.py` | 从 target seed 生长候选，并排除 reject 区域 |
| `build_v3_candidate_masks.py` | 构建待人工审查的四通道 candidate mask |
| `run_v3_pipeline.py` | 串联 v3 全流程 |
| `serve_v2_annotation_app.py` | 网页端人工审查和点级精修工具 |

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

在远程服务器上，优先验证 10 条 pilot 样本：

```bash
CUDA_VISIBLE_DEVICES=1,2 python MultiEEAffordance/tools/run_v3_pipeline.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml \
  --limit 10 \
  --stages views,plan,ground,project,grow,render,build \
  --allow-empty \
  --overwrite \
  --box-shrink-ratio 0.05 \
  --max-target-box-area-fraction 0.30 \
  --expand-hops 2
```

生成候选样本后，启动网页审查系统：

```bash
python MultiEEAffordance/tools/serve_v2_annotation_app.py \
  --dataset-root MultiEEAffordance \
  --samples processed/metadata/v3_candidate_samples_v0_1.jsonl \
  --port 8765 \
  --top-k-candidates 0
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
- 不要把 VLM/SAM2 或几何规则生成的 mask 直接作为最终标注。
- 修改标注规范或 pipeline 逻辑时，需要同步更新 `docs/项目进度日志.md`。

## 项目关键词

3D Affordance Grounding, heterogeneous end-effector, multi-label affordance, point cloud annotation, VLM-assisted annotation, Qwen3-VL, SAM2, human-in-the-loop labeling.
