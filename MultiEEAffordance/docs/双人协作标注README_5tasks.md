# 双人协作标注 README

更新时间：2026-06-01
适用版本：`serve_v2_annotation_app.py` 身份选择 + 五任务字段 + pending/checked/all 切换版

本文档说明两位审查者如何使用网页完成点级 mask 审查。当前默认方式是在数据存储服务器 `10.24.1.11` 上启动网页审查系统，五任务下采样批次和人工输出统一保存到：

```text
/home/lzq/data/MultiEEAffordance/
```

审查者不需要运行自动候选生成流程。维护者负责准备 reviewer 批次、启动两个独立端口，并把访问地址交给两位审查者。审查者打开网页、选择自己的身份、完成点级修正并保存即可。

本地拉取仓库、解压审查包、在本地启动网页的方式仍然保留，但只作为离线备用方案。

两位审查者统一使用下面两个身份：

```text
reviewer_a
reviewer_b
```

文档、目录、输出文件和网页身份都使用这两个名字。不要把个人姓名写进文件名，也不要在标注过程中临时改成别的身份。这样后续合并、查重和追踪会更稳定。

---

## 0. 当前网页审查系统有哪些新变化

这次网页审查系统相比旧版有四个重要改动。

### 0.1 打开网页后必须选择审查身份

进入网页后，页面会要求先选择：

```text
reviewer_a
reviewer_b
```

没有选择身份时，不应该开始保存。新版后端也会校验 reviewer 字段，如果保存请求里没有合法 reviewer，会拒绝保存，避免产生空 reviewer 记录。

选择身份以后，每次点击“保存 refined mask”，都会把身份写入：

```text
review_records.jsonl
refined_samples.jsonl
v2_point_edit.reviewer
```

这意味着后续可以明确追踪：

```text
谁标了这条样本
什么时候保存
保存后的 mask 在哪里
本次增删了哪些点
```

### 0.2 当前标注任务从旧三任务转为五任务

旧候选生成阶段历史任务是：

```text
pick_up
open_pull
press_push
```

人工标注阶段现在使用更清楚的五任务体系：

```text
lift
open
pull
press
push
```

对应关系是：

| 旧任务 | 新任务 | 说明 |
| --- | --- | --- |
| `pick_up` | `lift` | 从支撑面抬起物体，不强调长距离搬运 |
| `open_pull` | `open` + `pull` | 一个旧候选样本在人工标注前拆成打开和拉动两个任务 |
| `press_push` | `press` + `push` | 一个旧候选样本在人工标注前拆成按压和推动两个任务 |

注意：旧候选只是 proposal。`open_pull` 拆成 `open` 和 `pull` 后，两个新任务可以复用旧候选区域作为初始参考，但不能把旧候选直接当成最终真值。最终标签以人工在五任务语义下审查后的结果为准。

### 0.3 网页会显示并保存五任务相关字段

新版网页会优先显示：

```text
task_display
```

如果样本里没有 `task_display`，再显示 `target_task` 或 `task`。

保存审查记录时，`review_records.jsonl` 会写入下面这些五任务追溯字段：

```text
task
task_display
target_task
source_task
source_sample_id
task_taxonomy_version
task_split_source
```

这些字段的含义是：

| 字段 | 含义 |
| --- | --- |
| `task` | 当前真正用于标注的新任务，例如 `open`、`pull` |
| `task_display` | 页面展示名，例如 `Open`、`Pull` |
| `target_task` | 如果保留兼容字段，可表示目标任务 |
| `source_task` | 这条新任务样本来自哪个旧任务，例如 `open_pull` |
| `source_sample_id` | 拆分前的旧样本 ID |
| `task_taxonomy_version` | 任务体系版本，例如 `v0_2_5tasks` |
| `task_split_source` | 是否来自旧任务展开，例如 `legacy_task_expansion` |

### 0.4 左侧样本列表支持 pending / checked / all 切换

左侧现在有三个按钮：

```text
pending
checked
all
```

含义如下：

| 按钮 | 作用 |
| --- | --- |
| `pending` | 只看还没有完成的样本。默认进入网页时使用这个视图。 |
| `checked` | 只看已经保存为 checked 的样本，用于复查。 |
| `all` | 查看全部样本。 |

左侧的计数会跟随当前输出文件实时更新。保存成功后，样本会从 pending 进入 checked，或者显示为你保存时选择的其他状态。

---

## 1. 审查者到底要做什么

每个网页样本是一组：

```text
物体 object + 任务 task + 执行器 executor
```

在五任务版本中，样本看起来类似：

```text
Faucet + open + dexterous_hand
Door + pull + hook
Keyboard + press + dexterous_hand
Laptop + push + suction
Mug + lift + gripper
```

审查者的任务不是训练模型，也不是重新跑 VLM、SAM2 或 PartSLIP++。审查者只需要在网页里检查自动候选区域是否符合当前任务和当前执行器的作用机制，然后通过点级编辑得到最终 refined mask。

完整流程是：

```text
维护者在 10.24.1.11 生成候选和五任务批次
  -> 维护者把旧任务候选展开成五任务标注样本
  -> 维护者在 10.24.1.11 为 reviewer_a / reviewer_b 启动独立网页端口
  -> 审查者打开维护者提供的网址
  -> 审查者打开网页并选择 reviewer_a / reviewer_b
  -> 审查者逐条检查、增删点、保存 refined mask
  -> 审查者每天检查输出文件
  -> 维护者合并 reviewer_a / reviewer_b 的结果
```

审查者不需要运行：

```text
run_v3_pipeline.py
convert_3d_affordancenet.py
build_large_scale_review_queue.py
Qwen3-VL
SAM2
PartSLIP++
CUDA/GPU 推理环境
```

默认服务器模式下，网页脚本由维护者运行：

```text
MultiEEAffordance/tools/serve_v2_annotation_app.py
```

审查者只需要使用浏览器。离线备用模式下，审查者才需要在本地运行该脚本。这个网页只依赖 Python 标准库和 `numpy`。

---

## 2. 角色分工

## 2.1 维护者负责

维护者需要提前完成：

1. 在服务器或本地工作环境中生成候选样本。
2. 将旧任务候选样本展开成五任务版本：`lift/open/pull/press/push`。
3. 按 object 分组把样本拆成 `reviewer_a` 和 `reviewer_b` 两份。
4. 给每位审查者准备可直接解压的数据包。
5. 确认数据包解压后可以被网页读取。
6. 收到两位审查者的结果包后，合并 refined samples 和 refined masks。
7. 检查冲突样本、空样本、异常正例数量样本。

维护者不要在标注进行中随意重跑同名候选文件。否则审查者本地的 JSONL 可能会和候选 `.npz` 或点云 `.npy` 对不上。

## 2.2 审查者负责

每位审查者只处理自己分到的数据包：

```text
reviewer_a 只处理 reviewer_a_samples.jsonl
reviewer_b 只处理 reviewer_b_samples.jsonl
```

审查者需要做：

1. 从 GitHub 拉取稳定标注版本。
2. 配置轻量 Python 环境。
3. 解压自己的数据包。
4. 启动本地网页服务。
5. 打开浏览器后选择自己的身份。
6. 在网页中逐条审查样本。
7. 保存 refined mask。
8. 每天检查输出文件是否正常增加。
9. 打包并回传结果。

审查者不要改别人的输出路径。例如 `reviewer_a` 不要把结果写进 `reviewer_b_refined_samples.jsonl`。

---

## 3. 审查者第一次配置环境

审查者电脑需要：

```text
Git
Python 3.9 或更高版本
Chrome / Edge / Firefox 等现代浏览器
```

不需要 GPU。

### 3.1 拉取仓库

第一次操作：

```bash
git clone <repo_url>
cd Multi-EE-3DAG
git checkout annotation/mvp-v0.1
```

如果仓库已经拉过：

```bash
cd Multi-EE-3DAG
git fetch origin
git checkout annotation/mvp-v0.1
git pull
```

标注期间不要随便切到候选生成研发分支。网页版本和数据包是配套的，分支不一致可能导致字段不匹配。

### 3.2 创建轻量 Python 环境

Windows PowerShell：

```powershell
cd D:\VSCode\Multi-EE-3DAG
python -m venv .venv-review
.\.venv-review\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy
```

如果 PowerShell 提示不能执行脚本，在当前窗口先执行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv-review\Scripts\Activate.ps1
```

Linux / macOS：

```bash
cd ~/Multi-EE-3DAG
python -m venv .venv-review
source .venv-review/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy
```

检查环境：

```bash
python -c "import numpy as np; print(np.__version__)"
```

能打印 numpy 版本就可以继续。

---

## 4. 数据包从哪里来，解压到哪里

审查者不需要下载原始 `full-shape.zip`，也不需要自己生成候选。维护者会给每位审查者准备一个数据包，例如：

```text
annotation_batch_v0_2_5tasks_reviewer_a.zip
annotation_batch_v0_2_5tasks_reviewer_b.zip
```

数据包通常包含：

```text
MultiEEAffordance/processed/annotation_batches/v0_2_5tasks/reviewer_a_samples.jsonl
MultiEEAffordance/processed/annotation_batches/v0_2_5tasks/reviewer_b_samples.jsonl
MultiEEAffordance/processed/points/...
MultiEEAffordance/processed/vlm_candidate_v3.../3d_candidates/...
MultiEEAffordance/processed/vlm_candidate_v3.../fused_masks/...
MultiEEAffordance/processed/annotation_batches/v0_2_5tasks/BATCH_INFO.md
```

最重要的是：

```text
reviewer_a_samples.jsonl
reviewer_b_samples.jsonl
```

它们不是点云本身，而是“样本清单”。网页会读取清单，再根据里面的路径找到点云、候选和初始 mask。

路径读取规则是：

```text
--dataset-root MultiEEAffordance
```

也就是说，如果 JSONL 里写：

```text
processed/points/xxx/object.npy
```

那么本地必须存在：

```text
MultiEEAffordance/processed/points/xxx/object.npy
```

### 4.1 解压数据包

把维护者给的数据包放到仓库根目录，也就是 `Multi-EE-3DAG/` 下。

Windows PowerShell：

```powershell
cd D:\VSCode\Multi-EE-3DAG
Expand-Archive .\annotation_batch_v0_2_5tasks_reviewer_b.zip -DestinationPath . -Force
```

Linux / macOS：

```bash
cd ~/Multi-EE-3DAG
unzip annotation_batch_v0_2_5tasks_reviewer_b.zip
```

解压后检查样本清单是否存在。

Windows PowerShell：

```powershell
Test-Path .\MultiEEAffordance\processed\annotation_batches\v0_2_5tasks\reviewer_b_samples.jsonl
```

Linux / macOS：

```bash
test -f MultiEEAffordance/processed/annotation_batches/v0_2_5tasks/reviewer_b_samples.jsonl && echo ok
```

如果这里显示不存在，不要继续启动网页，先确认数据包是否解压到了正确位置。

---

## 5. 启动本地审查网页

启动网页前，先激活 Python 环境。命令行前面应能看到类似：

```text
(.venv-review)
```

### 5.1 reviewer_a 启动命令

Windows PowerShell：

```powershell
cd /home/lzq/Multi-EE-3DAG

python MultiEEAffordance/tools/serve_v2_annotation_app.py \
  --dataset-root /home/lzq/data/MultiEEAffordance \
  --samples processed/annotation_batches/v0_1_5tasks/reviewer_a_samples.jsonl \
  --review-jsonl processed/annotation_batches/v0_1_5tasks/reviewer_a_review_records.jsonl \
  --output-mask-root processed/annotation_batches/v0_1_5tasks/manual_refined_masks_reviewer_a \
  --output-samples processed/annotation_batches/v0_1_5tasks/reviewer_a_refined_samples.jsonl \
  --port 8765 \
  --top-k-candidates 8
```

Linux / macOS / Git Bash：

```bash
python MultiEEAffordance/tools/serve_v2_annotation_app.py \
  --dataset-root MultiEEAffordance \
  --samples processed/annotation_batches/v0_2_5tasks/reviewer_a_samples.jsonl \
  --review-jsonl processed/annotation_batches/v0_2_5tasks/reviewer_a_review_records.jsonl \
  --output-mask-root processed/annotation_batches/v0_2_5tasks/manual_refined_masks_reviewer_a \
  --output-samples processed/annotation_batches/v0_2_5tasks/reviewer_a_refined_samples.jsonl \
  --port 8765 \
  --top-k-candidates 8 \
  --candidate-min-selected-votes 2
```

### 5.2 reviewer_b 启动命令

Windows PowerShell：

```powershell
python .\MultiEEAffordance\tools\serve_v2_annotation_app.py `
  --dataset-root MultiEEAffordance `
  --samples processed/annotation_batches/v0_2_5tasks/reviewer_b_samples.jsonl `
  --review-jsonl processed/annotation_batches/v0_2_5tasks/reviewer_b_review_records.jsonl `
  --output-mask-root processed/annotation_batches/v0_2_5tasks/manual_refined_masks_reviewer_b `
  --output-samples processed/annotation_batches/v0_2_5tasks/reviewer_b_refined_samples.jsonl `
  --port 8765 `
  --top-k-candidates 8 `
  --candidate-min-selected-votes 2
```

Linux / macOS / Git Bash：

```bash
python MultiEEAffordance/tools/serve_v2_annotation_app.py \
  --dataset-root MultiEEAffordance \
  --samples processed/annotation_batches/v0_2_5tasks/reviewer_b_samples.jsonl \
  --review-jsonl processed/annotation_batches/v0_2_5tasks/reviewer_b_review_records.jsonl \
  --output-mask-root processed/annotation_batches/v0_2_5tasks/manual_refined_masks_reviewer_b \
  --output-samples processed/annotation_batches/v0_2_5tasks/reviewer_b_refined_samples.jsonl \
  --port 8765 \
  --top-k-candidates 8 \
  --candidate-min-selected-votes 2
```

启动成功后终端会显示：

```text
Serving v2 annotation app at http://127.0.0.1:8765
```

然后浏览器打开：

```text
http://127.0.0.1:8765
```

审查过程中不要关闭终端。关闭终端后，网页服务会停止。

---

## 6. 打开网页后的第一步：选择身份

网页打开后，先选择当前审查身份：

```text
reviewer_a
reviewer_b
```

选择原则很简单：

```text
如果你启动的是 reviewer_a_samples.jsonl，就选择 reviewer_a。
如果你启动的是 reviewer_b_samples.jsonl，就选择 reviewer_b。
```

不要出现下面这种错配：

```text
启动 reviewer_a_samples.jsonl，但网页身份选择 reviewer_b
启动 reviewer_b_samples.jsonl，但网页身份选择 reviewer_a
```

如果选错身份，后续 `review_records.jsonl` 里会把样本记录到错误 reviewer 名下，合并时会造成混乱。

如果不确定自己当前开的是什么包，看启动命令里的 `--samples`：

```text
--samples .../reviewer_a_samples.jsonl  -> 选择 reviewer_a
--samples .../reviewer_b_samples.jsonl  -> 选择 reviewer_b
```

---

## 7. 网页界面怎么用

网页分三块：

```text
左侧：样本列表和筛选
中间：点云、当前 mask、点级编辑
右侧：样本信息、候选区域、状态选择、保存按钮
```

### 7.1 左侧 pending / checked / all

左侧顶部有三个筛选按钮：

```text
pending
checked
all
```

建议正常标注时使用 `pending`，只看未完成样本。每天结束或复查时再切到 `checked` 或 `all`。

每条样本会显示：

```text
sample_id
object_category
task_display 或 task
source_task
task_taxonomy_version
executor
review_status
pos=正例点数量
```

`pos=0` 不一定是错误。某些任务-执行器组合确实可能没有合适区域，人工确认后可以保存为空标签。

### 7.2 中间点云区域

中间显示当前点云和当前目标 executor 通道的 mask。

常用操作：

| 操作 | 说明 |
| --- | --- |
| 鼠标滚轮 | 缩放 |
| 右键拖动 | 旋转点云 |
| 左键 | 按当前模式增删点 |
| `brush` 滑条 | 调整一次编辑的圆形范围 |
| `查看/旋转` | 只观察，不编辑点 |
| `点击切换点` | 正例变负例，负例变正例 |
| `只添加` | 只补正例点 |
| `只删除` | 只删除正例点 |
| `撤销` | 撤回上一次编辑 |
| `重置视角` | 回到默认视角 |

### 7.3 右侧信息栏

右侧重点看：

```text
sample_id
object_category
task
target_executor
task_key / task_display / source_task / taxonomy
selected_candidates
候选列表
review_status
review_decision
quality_after_review
notes
保存 refined mask
```

审查前先确认当前任务和执行器。不要只按物体形状标，要按“这个执行器完成这个任务应接触哪里”来判断。

---

## 8. 五任务标注标准简要说明

这里是网页操作用的简明标准。更详细标准应以《异构末端执行器标注规范》为准。

### 8.1 lift

`lift` 表示把物体从支撑面抬起，不强调长距离搬运。

不同执行器关注点：

| executor | 正例区域 |
| --- | --- |
| `gripper` | 可二指夹住并稳定抬起的两侧、柄部、边缘厚结构 |
| `suction` | 面积足够、较平整、远离边缘、可吸附抬起的面 |
| `hook` | 孔、环、提手开口、把手内侧等可挂住并承重的结构 |
| `dexterous_hand` | 类人手可抓握、包覆、托住并抬起的区域 |

不要为了非空而把普通接触面都标为正例。

### 8.2 open

`open` 表示使可开合部件从关闭状态变为打开状态。

典型正例：

```text
门把手
柜门/冰箱门把手
微波炉门把手
笔记本可开合边缘
水龙头开关
垃圾桶盖可打开部位
```

不要把整个门板、整个外壳或无关大平面直接标为 open 正例，除非该执行器确实能通过该区域完成打开。

### 8.3 pull

`pull` 表示拉动目标结构，使其沿拉力方向产生位移或受力。

典型正例：

```text
拉环
把手
袋子提手
剪刀环
抽屉/门类拉手
可被钩子扣住的孔洞或内边界
```

`pull` 不一定等于 `open`。例如拉住一个环或手柄可以是 pull，但不一定导致开合状态变化。

### 8.4 press

`press` 表示按压按钮、按键、开关等小型可按压部件。

典型正例：

```text
键盘按键
按钮
开关
微波炉控制按键
显示器按键
水龙头小型按压/切换部件
```

`press` 通常更适合 `dexterous_hand`。`hook` 大多不适合 press。`suction` 通常也不适合按小按钮，除非任务定义明确允许大面接触按压。

### 8.5 push

`push` 表示推动目标部件、面板或物体表面。

典型正例：

```text
可推动面板
可推开的门板区域
可推动开关/大按钮
可推动物体的稳定接触面
```

`push` 和 `press` 的区别：

```text
press 更偏小按钮/按键/开关
push 更偏大面板/推动表面/推开动作
```

---

## 9. 如何审查一条样本

建议每条样本按下面顺序处理。

### 第一步：确认身份、任务、执行器

先确认页面身份是否正确，然后看：

```text
object_category
task
target_executor
source_task
task_taxonomy_version
```

心里问一句：

```text
这个执行器要完成当前任务，应该接触哪里？
```

不要一上来就直接保存自动结果。

### 第二步：检查自动候选

右侧候选列表里，每个候选都有勾选框。勾选框表示“准备采用这个候选”。

常见操作：

1. 勾选一个候选，观察点云上亮起的位置。
2. 勾选多个候选，看它们的合集。
3. 点击“预览勾选组合”，确认组合是否覆盖目标区域。
4. 点击“应用勾选候选”，把勾选组合变成当前 mask。

自动候选只是起点，不是最终 GT。如果候选覆盖主体、无关边缘、刀刃、普通外壳，必须删掉或取消。

### 第三步：点级精修

| 情况 | 处理方式 |
| --- | --- |
| 候选基本正确 | 少量删除错误点，补上漏掉小区域 |
| 候选太大 | 取消大候选，或应用后用 `只删除` 清理 |
| 候选太少 | 用 `只添加` 补点 |
| 候选全错 | 清空当前 mask，再手动补正确区域 |
| 没有合适区域 | 保持空 mask，保存为空标签确认 |

### 第四步：设置状态

一般样本：

```text
review_status = checked
review_decision = accept_refined
quality_after_review = checked
```

确认应该为空的样本：

```text
review_status = checked
review_decision = confirm_empty
quality_after_review = checked
```

如果当前页面没有 `confirm_empty` 选项，保持 `accept_refined` 也可以，但要在 notes 里写明：

```text
confirmed empty: no feasible region for this task-executor pair
```

不确定样本：

```text
review_status = refine_needed
review_decision = uncertain
quality_after_review = weak
```

并在 notes 里写清原因。

### 第五步：保存

点击：

```text
保存 refined mask
```

保存成功后，页面会提示 positive 点数变化。该样本会从 pending 切换到 checked 或你选择的状态。保存后再进入下一条。

---

## 10. 保存后会写哪些文件

网页不会把结果只存在浏览器里，而是写到本地仓库目录。

以 `reviewer_b` 为例，保存后会生成或更新：

```text
MultiEEAffordance/processed/annotation_batches/v0_2_5tasks/reviewer_b_review_records.jsonl
MultiEEAffordance/processed/annotation_batches/v0_2_5tasks/reviewer_b_refined_samples.jsonl
MultiEEAffordance/processed/annotation_batches/v0_2_5tasks/manual_refined_masks_reviewer_b/
```

三者含义：

| 文件或目录 | 作用 |
| --- | --- |
| `reviewer_b_review_records.jsonl` | 审查流水。每次保存都会追加一条记录。 |
| `reviewer_b_refined_samples.jsonl` | 当前最新 refined 样本清单。后续合并主要读这个文件。 |
| `manual_refined_masks_reviewer_b/` | 真正的人工 refined mask，里面是 `.npy` 文件。 |

简单理解：

```text
review_records = 每次操作记录
refined_samples = 每条样本的最新结果清单
manual_refined_masks = 真正人工修正后的 mask 文件
```

这三类都必须保留，回传结果时不能只回传其中一个。

---

## 11. review_records.jsonl 里应包含哪些字段

新版网页保存时，`review_records.jsonl` 每条记录应至少包含：

```text
created_at
row_key
sample_id
object_id
object_category
task
task_display
target_task
source_task
source_sample_id
task_taxonomy_version
task_split_source
executor
reviewer
review_status
review_decision
quality_after_review
notes
positive_points_before
positive_points_after
selected_candidate_ids
added_points
removed_points
output_mask_path
```

重点检查两类字段。

第一类是 reviewer 字段：

```text
reviewer = reviewer_a 或 reviewer_b
```

如果 reviewer 是空字符串，说明打开网页后没有正确选择身份，或者网页版本不是身份选择版。

第二类是五任务字段：

```text
task
task_display
source_task
task_taxonomy_version
task_split_source
```

如果这些字段缺失，说明使用的可能是旧版网页或旧版 samples。

---

## 12. 每天结束前必须检查

每天结束前，不要只关网页。必须检查结果文件有没有增加。

### 12.1 Windows PowerShell 检查数量

以 `reviewer_b` 为例：

```powershell
cd D:\VSCode\Multi-EE-3DAG

(Get-Content .\MultiEEAffordance\processed\annotation_batches\v0_2_5tasks\reviewer_b_refined_samples.jsonl).Count
(Get-Content .\MultiEEAffordance\processed\annotation_batches\v0_2_5tasks\reviewer_b_review_records.jsonl).Count
(Get-ChildItem .\MultiEEAffordance\processed\annotation_batches\v0_2_5tasks\manual_refined_masks_reviewer_b -Filter *.npy -Recurse).Count
```

如果当天审查了 20 条，`refined_samples` 行数和 `.npy` 数量应该明显增加。

### 12.2 Linux / macOS 检查数量

```bash
cd ~/Multi-EE-3DAG

wc -l MultiEEAffordance/processed/annotation_batches/v0_2_5tasks/reviewer_b_refined_samples.jsonl
wc -l MultiEEAffordance/processed/annotation_batches/v0_2_5tasks/reviewer_b_review_records.jsonl
find MultiEEAffordance/processed/annotation_batches/v0_2_5tasks/manual_refined_masks_reviewer_b -name "*.npy" | wc -l
```

### 12.3 检查 reviewer 和五任务字段是否写入

Windows PowerShell 可运行：

```powershell
python - <<'PY'
import json
from pathlib import Path

path = Path("MultiEEAffordance/processed/annotation_batches/v0_2_5tasks/reviewer_b_review_records.jsonl")
required = [
    "reviewer", "task", "task_display", "source_task",
    "task_taxonomy_version", "task_split_source", "executor",
]

if not path.exists():
    print("review_records not found:", path)
    raise SystemExit

rows = []
with path.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

print("records:", len(rows))
if not rows:
    raise SystemExit

last = rows[-1]
print("last row key:", last.get("row_key"))
for k in required:
    print(f"{k} = {last.get(k)!r}")

missing = [k for k in required if k not in last]
empty = [k for k in required if last.get(k) in (None, "") and k not in ("source_task", "task_split_source")]
print("missing fields:", missing)
print("empty important fields:", empty)
PY
```

Linux / macOS 同样可以运行上面的 Python 片段。

如果输出里：

```text
reviewer = ''
```

说明身份没有正确选择。请停止继续标注，先确认网页版本。

---

## 13. 每天记录什么

建议每天写一个简单记录，例如：

```text
MultiEEAffordance/processed/annotation_batches/v0_2_5tasks/daily_notes_reviewer_b_20260528.md
```

内容不用很正式，但要能帮助维护者复查。

模板：

```text
日期：2026-05-28
审查者：reviewer_b
今日完成：35 条
累计完成：120 条

今日确认空标签：
- sample_id=xxx, task=pull, executor=hook, reason=没有孔/环/把手

今日不确定样本：
- sample_id=yyy, task=open, executor=suction, question=平面能否吸住后打开不确定

自动候选典型问题：
- faucet open dexterous_hand 自动候选覆盖主体，实际只保留上方开关
- scissors pull hook 自动候选包含刀刃，已删除

网页或数据问题：
- 无
```

---

## 14. 每天或每批如何回传结果

建议每天结束后打一个结果包。

### 14.1 Windows PowerShell 打包

以 `reviewer_b` 为例：

```powershell
cd D:\VSCode\Multi-EE-3DAG

Compress-Archive -Path `
  .\MultiEEAffordance\processed\annotation_batches\v0_2_5tasks\reviewer_b_review_records.jsonl, `
  .\MultiEEAffordance\processed\annotation_batches\v0_2_5tasks\reviewer_b_refined_samples.jsonl, `
  .\MultiEEAffordance\processed\annotation_batches\v0_2_5tasks\manual_refined_masks_reviewer_b, `
  .\MultiEEAffordance\processed\annotation_batches\v0_2_5tasks\daily_notes_reviewer_b_20260528.md `
  -DestinationPath .\reviewer_b_annotation_v0_2_5tasks_20260528.zip `
  -Force
```

### 14.2 Linux / macOS 打包

```bash
cd ~/Multi-EE-3DAG

tar -czf reviewer_b_annotation_v0_2_5tasks_20260528.tar.gz \
  MultiEEAffordance/processed/annotation_batches/v0_2_5tasks/reviewer_b_review_records.jsonl \
  MultiEEAffordance/processed/annotation_batches/v0_2_5tasks/reviewer_b_refined_samples.jsonl \
  MultiEEAffordance/processed/annotation_batches/v0_2_5tasks/manual_refined_masks_reviewer_b \
  MultiEEAffordance/processed/annotation_batches/v0_2_5tasks/daily_notes_reviewer_b_20260528.md
```

结果包命名建议：

```text
reviewer_a_annotation_v0_2_5tasks_YYYYMMDD.zip
reviewer_b_annotation_v0_2_5tasks_YYYYMMDD.zip
```

不要只回传 `review_records.jsonl`。必须同时包含：

```text
review_records.jsonl
refined_samples.jsonl
manual_refined_masks_reviewer_x/
daily_notes_reviewer_x_YYYYMMDD.md
```

---

## 15. 维护者如何合并两份结果

维护者收到两份结果包后，建议解压到：

```text
MultiEEAffordance/processed/annotation_batches/v0_2_5tasks/
```

先检查 reviewer 字段和五任务字段，再合并。

### 15.1 检查两份结果的 reviewer 是否正确

```bash
python - <<'PY'
import json
from pathlib import Path

paths = [
    Path("MultiEEAffordance/processed/annotation_batches/v0_2_5tasks/reviewer_a_review_records.jsonl"),
    Path("MultiEEAffordance/processed/annotation_batches/v0_2_5tasks/reviewer_b_review_records.jsonl"),
]

for path in paths:
    if not path.exists():
        print("missing", path)
        continue
    reviewers = set()
    tasks = set()
    missing_taxonomy = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            reviewers.add(row.get("reviewer", ""))
            tasks.add(row.get("task", ""))
            if "task_taxonomy_version" not in row:
                missing_taxonomy += 1
    print(path)
    print("  reviewers:", reviewers)
    print("  tasks:", sorted(tasks))
    print("  missing_taxonomy:", missing_taxonomy)
PY
```

期望看到：

```text
reviewer_a_review_records.jsonl -> reviewers: {'reviewer_a'}
reviewer_b_review_records.jsonl -> reviewers: {'reviewer_b'}
tasks 包含 lift/open/pull/press/push
missing_taxonomy 为 0 或很少
```

### 15.2 合并 refined samples

```bash
mkdir -p MultiEEAffordance/processed/annotation_batches/v0_2_5tasks/merged

cat \
  MultiEEAffordance/processed/annotation_batches/v0_2_5tasks/reviewer_a_refined_samples.jsonl \
  MultiEEAffordance/processed/annotation_batches/v0_2_5tasks/reviewer_b_refined_samples.jsonl \
  > MultiEEAffordance/processed/annotation_batches/v0_2_5tasks/merged/merged_refined_samples.jsonl
```

### 15.3 检查重复或冲突样本

```bash
python - <<'PY'
import json
from collections import defaultdict
from pathlib import Path

paths = [
    Path("MultiEEAffordance/processed/annotation_batches/v0_2_5tasks/reviewer_a_refined_samples.jsonl"),
    Path("MultiEEAffordance/processed/annotation_batches/v0_2_5tasks/reviewer_b_refined_samples.jsonl"),
]

seen = defaultdict(list)
for path in paths:
    if not path.exists():
        continue
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("row_key") or "|".join([
                str(row.get("pilot_id") or row.get("review_id") or ""),
                str(row.get("sample_id") or ""),
                str(row.get("task") or ""),
                str(row.get("target_executor") or row.get("executor") or ""),
            ])
            seen[key].append((str(path), line_no))

conflicts = {key: locs for key, locs in seen.items() if len(locs) > 1}
print("conflicts:", len(conflicts))
for key, locs in list(conflicts.items())[:50]:
    print(key, locs)
PY
```

如果 `conflicts > 0`，不要自动覆盖。先人工复查冲突样本。

---

## 16. 常见问题

| 问题 | 可能原因 | 处理方式 |
| --- | --- | --- |
| 打开网页后没有身份选择 | 使用了旧版 `serve_v2_annotation_app.py` | 更新到身份选择版网页代码 |
| 保存时报 reviewer 错误 | 没选择 `reviewer_a/reviewer_b`，或选择值异常 | 刷新页面，重新选择身份后再保存 |
| 保存后 reviewer 为空 | 旧版网页或前端未传 reviewer | 停止继续标注，更新网页代码 |
| 左侧 checked 里没有刚保存的样本 | 保存失败，或 review_status 没设为 checked | 看终端报错，检查 `review_records.jsonl` 是否新增 |
| 点云不显示 | 数据包缺点云或路径不对 | 把终端报错路径发给维护者 |
| 候选列表为空 | 该样本没有高置信候选，或候选文件缺失 | 可以手动点级编辑；严重漏召回写 daily notes |
| `pos=0` | 当前 mask 为空 | 判断是否确实无可用区域；确认后可保存空标签 |
| 任务显示 source_task=open_pull | 正常，说明新任务从旧 `open_pull` 展开而来 | 按当前显示的新任务 `open` 或 `pull` 标注 |
| 不确定该不该标 | 执行器机制或点云结构不清楚 | 保存为 `uncertain`，并写 notes |

---

## 17. 不要做的事

- 不要自己运行 VLM pipeline。
- 不要自己重新生成候选。
- 不要把 `reviewer_a` 的结果写到 `reviewer_b` 文件里。
- 不要打开 `reviewer_a_samples.jsonl` 却在网页里选择 `reviewer_b`。
- 不要把自动候选直接当成最终 GT。
- 不要为了让样本非空而强行补点。
- 不要只回传 `review_records.jsonl`，必须同时回传 refined samples 和 mask 文件夹。
- 不要把大量 `.npy/.npz` 文件直接提交到普通 Git 仓库。

---

## 18. 当前原始数据说明

当前原始数据由维护者使用，例如：

```text
/home/lzq/Multi-EE-3DAG/MultiEEAffordance/raw/3d_affordancenet/full-shape.zip
```

审查者通常不需要这个原始 zip。审查者需要的是维护者已经处理好的数据包，里面包含当前批次要审查的点云、候选和样本清单。

简单说：

```text
full-shape.zip = 维护者生成数据用
annotation_batch_v0_2_5tasks_reviewer_x.zip = 审查者本地标注用
reviewer_x_annotation_v0_2_5tasks_YYYYMMDD.zip = 审查者回传结果用
```

---

## 19. 最后检查清单

审查者每天结束前确认：

```text
[ ] 网页里选择的身份正确：reviewer_a / reviewer_b
[ ] pending 数量减少，checked 数量增加
[ ] reviewer_x_review_records.jsonl 行数增加
[ ] reviewer_x_refined_samples.jsonl 行数增加
[ ] manual_refined_masks_reviewer_x/ 下 .npy 数量增加
[ ] review_records 最后一条 reviewer 字段不是空
[ ] review_records 最后一条包含 task/task_display/source_task/task_taxonomy_version
[ ] 今日不确定样本已经写进 daily notes
[ ] 结果包包含 review_records、refined_samples、manual_refined_masks、daily_notes
```

维护者合并前确认：

```text
[ ] reviewer_a 和 reviewer_b 的 reviewer 字段没有错配
[ ] 五任务字段存在
[ ] 没有同一个 row_key 被两人重复覆盖
[ ] pos=0 样本已抽查
[ ] hook/suction/dexterous_hand 的典型错误已抽查
[ ] 合并后的 merged_refined_samples.jsonl 可正常读取
```
