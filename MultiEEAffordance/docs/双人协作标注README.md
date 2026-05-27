# 双人协作标注 README

更新时间：2026-05-27

本文档说明当前标注版本如何由两位审查者通过 GitHub 协作完成。当前方式不依赖 SSH，也不要求审查者登录服务器。审查者只需要在自己的电脑上拉取代码、解压维护者准备好的数据包、本地打开审查网页、完成点级审查、再把结果包回传给维护者。

当前两位审查者统一记为：

- `reviewer_a`
- `reviewer_b`

文档、目录、输出文件都使用这两个名字。不要把个人姓名写进文件名，后续合并数据时会更清楚。

## 0. 先理解这件事在做什么

每个样本是一组：

```text
物体 object + 任务 task + 执行器 executor
```

例如：

```text
Faucet + open_pull + dexterous_hand
Scissors + pick_up + hook
Door + open_pull + suction
```

审查者的任务不是重新跑模型，也不是训练网络，而是在网页里检查自动候选区域是否合理，并把最终认为正确的点云区域保存下来。

完整流程是：

```text
维护者生成候选和数据包
  -> 审查者从 GitHub 拉代码
  -> 审查者解压自己的数据包
  -> 审查者本地运行审查网页
  -> 审查者逐条检查点云 mask
  -> 审查者每天检查输出文件
  -> 审查者把结果包回传
  -> 维护者合并 reviewer_a / reviewer_b 的结果
```

审查者不需要运行：

- `run_v3_pipeline.py`
- Qwen3-VL
- SAM2
- PartSLIP++
- CUDA/GPU 推理环境

审查者只运行一个轻量网页：

```text
MultiEEAffordance/tools/serve_v2_annotation_app.py
```

这个网页只依赖 Python 标准库和 `numpy`。

## 1. GitHub 协作方式

当前协作分成三类东西：

| 类型 | 放在哪里 | 作用 |
| --- | --- | --- |
| 代码和文档 | GitHub 仓库 | 审查网页、脚本、说明文档 |
| 标注数据包 | GitHub Release 附件、网盘或其他文件传输方式 | 点云、候选、样本 JSONL |
| 审查结果包 | GitHub Release 附件、Issue 附件、网盘或压缩包回传 | 人工 refined mask、审查记录 |

不要把完整原始数据 zip、大规模点云目录、大量 `.npy/.npz` 候选结果直接提交到普通 Git 仓库。普通 GitHub 仓库主要保存代码、文档和小型索引文件。大文件用压缩包传。

## 2. 角色分工

### 2.1 维护者负责

维护者需要提前做好这些事：

1. 在服务器或自己的工作环境中生成候选样本。
2. 把样本拆成 `reviewer_a` 和 `reviewer_b` 两份。
3. 给每位审查者准备一个最小数据包。
4. 确认数据包解压后能被本地审查网页读取。
5. 收到审查结果后合并两份结果。
6. 处理冲突样本和异常样本。

维护者不应该在标注进行中随意重跑同名候选文件，否则审查者本地的样本 JSONL 可能和候选 `.npz` 对不上。

### 2.2 审查者负责

每位审查者只处理自己分到的数据包：

```text
reviewer_a 只处理 reviewer_a_samples.jsonl
reviewer_b 只处理 reviewer_b_samples.jsonl
```

审查者需要做：

1. 从 GitHub 拉取稳定标注版本。
2. 配置一个轻量 Python 环境。
3. 解压自己的数据包。
4. 启动本地审查网页。
5. 按要求逐条审查样本。
6. 每天检查保存文件是否正常增加。
7. 把结果包回传给维护者。

审查者不要改别人的输出目录，也不要把自己的结果写到另一个 reviewer 的文件名里。

## 3. 审查者第一次配置环境

审查者电脑需要：

- Git
- Python 3.9 或更高版本
- 一个现代浏览器，例如 Chrome、Edge、Firefox

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

标注期间不要切到候选生成研发分支。审查网页和数据包是按稳定标注版本准备的，分支不一致可能导致页面字段不匹配。

### 3.2 创建轻量 Python 环境

Windows PowerShell 推荐：

```powershell
cd D:\VSCode\Multi-EE-3DAG
python -m venv .venv-review
.\.venv-review\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy
```

如果 PowerShell 提示不能执行脚本，可以先在当前窗口执行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv-review\Scripts\Activate.ps1
```

Linux 或 macOS 推荐：

```bash
cd ~/Multi-EE-3DAG
python -m venv .venv-review
source .venv-review/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy
```

检查环境是否正常：

```bash
python -c "import numpy as np; print(np.__version__)"
```

能打印版本号就可以继续。

## 4. 数据包从哪里来，解压到哪里

审查者不需要下载原始的 `full-shape.zip`，也不需要自己生成候选。维护者会给每位审查者一个标注数据包，例如：

```text
annotation_batch_v0_1_reviewer_a.zip
annotation_batch_v0_1_reviewer_b.zip
```

这个数据包通常包含：

```text
MultiEEAffordance/processed/annotation_batches/v0_1/reviewer_b_samples.jsonl
MultiEEAffordance/processed/points/...
MultiEEAffordance/processed/vlm_candidate_v3/3d_candidates/...
MultiEEAffordance/processed/vlm_candidate_v3/fused_masks/...
MultiEEAffordance/processed/annotation_batches/v0_1/BATCH_INFO.md
```

其中最重要的是 `reviewer_b_samples.jsonl` 或 `reviewer_a_samples.jsonl`。它不是点云本身，而是一张“清单”。网页会读取这张清单，再根据里面的路径找到点云、候选和初始 mask。

路径读取规则是：

```text
--dataset-root MultiEEAffordance
```

也就是说，样本 JSONL 里的路径都默认从 `MultiEEAffordance/` 下面开始找。例如 JSONL 里写：

```text
processed/points/xxx/object.npy
```

那么本地电脑上必须存在：

```text
MultiEEAffordance/processed/points/xxx/object.npy
```

### 4.1 解压数据包

把维护者给的数据包放到仓库根目录，也就是 `Multi-EE-3DAG/` 下面。

Windows PowerShell 示例：

```powershell
cd D:\VSCode\Multi-EE-3DAG
Expand-Archive .\annotation_batch_v0_1_reviewer_b.zip -DestinationPath . -Force
```

Linux 或 macOS 示例：

```bash
cd ~/Multi-EE-3DAG
unzip annotation_batch_v0_1_reviewer_b.zip
```

解压后检查样本清单是否存在。

Windows PowerShell：

```powershell
Test-Path .\MultiEEAffordance\processed\annotation_batches\v0_1\reviewer_b_samples.jsonl
```

Linux 或 macOS：

```bash
test -f MultiEEAffordance/processed/annotation_batches/v0_1/reviewer_b_samples.jsonl && echo ok
```

如果这里显示不存在，不要继续启动网页，先确认数据包是否解压到了正确位置。

## 5. 启动本地审查网页

启动网页前，先确认 Python 环境已经激活。

Windows PowerShell 里命令行前面应该能看到类似：

```text
(.venv-review)
```

### 5.1 `reviewer_a` 启动命令

Windows PowerShell 可以直接复制：

```powershell
python .\MultiEEAffordance\tools\serve_v2_annotation_app.py `
  --dataset-root MultiEEAffordance `
  --samples processed/annotation_batches/v0_1/reviewer_a_samples.jsonl `
  --review-jsonl processed/annotation_batches/v0_1/reviewer_a_review_records.jsonl `
  --output-mask-root processed/annotation_batches/v0_1/manual_refined_masks_reviewer_a `
  --output-samples processed/annotation_batches/v0_1/reviewer_a_refined_samples.jsonl `
  --port 8765 `
  --top-k-candidates 8 `
  --candidate-min-selected-votes 2
```

### 5.2 `reviewer_b` 启动命令

Windows PowerShell 可以直接复制：

```powershell
python .\MultiEEAffordance\tools\serve_v2_annotation_app.py `
  --dataset-root MultiEEAffordance `
  --samples processed/annotation_batches/v0_1/reviewer_b_samples.jsonl `
  --review-jsonl processed/annotation_batches/v0_1/reviewer_b_review_records.jsonl `
  --output-mask-root processed/annotation_batches/v0_1/manual_refined_masks_reviewer_b `
  --output-samples processed/annotation_batches/v0_1/reviewer_b_refined_samples.jsonl `
  --port 8765 `
  --top-k-candidates 8 `
  --candidate-min-selected-votes 2
```

每个人在自己的电脑上运行，所以都可以使用 `8765` 端口。如果同一台电脑同时开两个审查服务，再把其中一个端口改成 `8766`。

Linux、macOS 或 Git Bash 用户也可以运行同样参数，只是把 PowerShell 的反引号换成反斜杠：

```bash
python MultiEEAffordance/tools/serve_v2_annotation_app.py \
  --dataset-root MultiEEAffordance \
  --samples processed/annotation_batches/v0_1/reviewer_b_samples.jsonl \
  --review-jsonl processed/annotation_batches/v0_1/reviewer_b_review_records.jsonl \
  --output-mask-root processed/annotation_batches/v0_1/manual_refined_masks_reviewer_b \
  --output-samples processed/annotation_batches/v0_1/reviewer_b_refined_samples.jsonl \
  --port 8765 \
  --top-k-candidates 8 \
  --candidate-min-selected-votes 2
```

启动成功后，终端会显示类似：

```text
Serving v2 annotation app at http://127.0.0.1:8765
```

然后打开浏览器访问：

```text
http://127.0.0.1:8765
```

审查过程中不要关闭这个终端窗口。关闭后网页服务就停止了。

## 6. 打开网页后先看哪里

网页一般分三块：

```text
左侧：样本列表
中间：点云和当前 mask
右侧：当前样本信息、候选区域、保存按钮
```

### 6.1 左侧样本列表

左侧会显示待审查和已审查样本。每个样本会显示：

- `sample_id`
- 物体类别，例如 `Faucet`、`Door`、`Scissors`
- 任务，例如 `pick_up`、`open_pull`、`press_push`
- 执行器，例如 `gripper`、`suction`、`hook`、`dexterous_hand`
- `pos=数字`

`pos` 表示当前 mask 里有多少个正例点。`pos=0` 不一定是错误，有些任务和执行器组合本来就可能没有可用区域。

### 6.2 中间点云区域

中间显示当前物体点云。

常用操作：

- 鼠标滚轮：缩放。
- 右键拖动：旋转点云。
- 左键：根据当前模式增删点。
- `brush` 滑条：调整一次增删的圆形范围。
- `查看/旋转`：主要用于观察，不编辑点。
- `点击切换点`：左键点击圆形范围内的点，正例变负例，负例变正例。
- `只添加`：左键只补正例点。
- `只删除`：左键只删正例点。
- `撤销`：撤回上一次编辑。
- `重置视角`：回到默认视角。

### 6.3 右侧信息栏

右侧最重要的是：

- 当前 `sample_id`
- `object_category`
- `task`
- `target_executor`
- 自动候选来源
- 候选选择列表
- `review_status`
- `review_decision`
- `quality_after_review`
- `保存 refined mask`

审查前先看清楚当前任务和执行器。不要只按物体形状标。例如同样是剪刀：

- `pick_up + hook` 应该看孔、环、内边界等可挂接位置。
- `pick_up + gripper` 更关注可夹住、可稳定约束的位置。
- 刀刃或尖端通常不应该作为 hook 正例。

## 7. 如何审查一条样本

建议每条样本按这个顺序做。

### 第一步：确认任务和执行器

先看页面顶部或右侧：

```text
object_category
task
target_executor
```

心里先问一句：

```text
这个执行器要完成这个任务，应该接触哪里？
```

不要一上来就保存自动结果。

### 第二步：看自动候选

右侧候选列表里，每个候选都有一个勾选框。勾选框表示“准备采用这个候选”。

常见操作：

1. 勾选一个候选，看点云上亮起的位置。
2. 勾选多个候选，看它们的合集。
3. 点击“预览勾选组合”，确认当前组合是否覆盖目标区域。
4. 点击“应用勾选候选”，把当前勾选组合变成当前 mask。

如果自动候选覆盖了大主体、无关边缘、刀刃、普通外壳，不要直接保存。先取消不合理候选，或者应用后用 `只删除` 清掉错误点。

### 第三步：点级精修

自动候选只是起点，不是最终标注。

常见情况：

| 情况 | 怎么做 |
| --- | --- |
| 自动候选基本正确 | 少量删除错误点，补上漏掉的小区域 |
| 自动候选太大 | 取消大候选，或应用后用 `只删除` 清理主体误标 |
| 自动候选太少 | 用 `只添加` 手动补点 |
| 候选全错 | 点击“清空当前 mask”，然后手动补正确区域 |
| 没有合适区域 | 保持空 mask，保存为空标签确认 |

### 第四步：设置审查状态

一般情况：

```text
review_status = checked
review_decision = accept_refined
quality_after_review = checked
```

如果当前样本没有合适区域，并且你已经确认应该为空：

```text
review_status = checked
review_decision = confirm_empty
quality_after_review = checked
```

如果网页里暂时没有显示 `confirm_empty`，保持 `accept_refined` 也可以，但要在当天记录里写明这个样本是空标签确认。

如果你不确定：

```text
review_status = refine_needed
review_decision = uncertain
quality_after_review = weak
```

并在当天记录里写下原因。

### 第五步：保存

点击：

```text
保存 refined mask
```

保存成功后，网页会提示 positive 点数变化。左侧样本状态也应该从 pending 变成 checked 或更新为当前状态。

保存后再切换到下一条样本。

## 8. 如何判断标注是否合理

### 8.1 `pick_up`

`pick_up` 是拿起物体。要看执行器类型：

- `gripper`：能被夹住、夹稳的位置。
- `suction`：较平、较完整、能吸住的位置。
- `hook`：孔、环、把手内边界、可挂住的位置。
- `dexterous_hand`：手能抓握、托住、包覆的位置。

### 8.2 `open_pull`

`open_pull` 是打开或拉动。通常正例是：

- 门把手
- 抽屉把手
- 水龙头开关
- 可拉的环、孔、柄、旋钮

通常不是正例：

- 大主体面
- 装饰边
- 普通平滑外壳
- 不能受力打开的区域

### 8.3 `press_push`

`press_push` 是按压或推动。通常正例是：

- button
- switch
- 键盘按键
- 可按压面板
- 明确的推压区域

通常不是正例：

- 整个物体主体
- 和任务无关的把手
- 细长边缘

### 8.4 空 mask 是有意义的

空 mask 不是失败。它表示这个物体在当前任务和当前执行器下没有合适区域。

例如：

- 剪刀如果没有适合 suction 的稳定平面，可以为空。
- 某些 `hook` 任务如果没有孔、环、把手，可以为空。
- 某些 `press_push + hook` 组合本来就不合理，可以为空。

不要为了让 `pos` 不等于 0 而强行标注。

## 9. 保存后文件会写到哪里

网页不会把结果保存到浏览器里，而是写到本地仓库的 `MultiEEAffordance/processed/annotation_batches/v0_1/` 下。

以 `reviewer_b` 为例，保存后会产生或更新：

```text
MultiEEAffordance/processed/annotation_batches/v0_1/reviewer_b_review_records.jsonl
MultiEEAffordance/processed/annotation_batches/v0_1/reviewer_b_refined_samples.jsonl
MultiEEAffordance/processed/annotation_batches/v0_1/manual_refined_masks_reviewer_b/
```

这三个东西分别表示：

| 文件或目录 | 作用 |
| --- | --- |
| `reviewer_b_review_records.jsonl` | 审查日志。每次保存都会追加一条记录，便于追踪谁保存了什么。 |
| `reviewer_b_refined_samples.jsonl` | 精修后的样本索引。后续合并 reviewed dataset 主要读这个文件。 |
| `manual_refined_masks_reviewer_b/` | 真正的人工 refined mask，里面是 `.npy` 文件。 |

简单理解：

```text
review_records = 操作记录
refined_samples = 最终样本清单
manual_refined_masks = 真正的人工 mask 文件
```

这三者都要保留。

## 10. 每天结束前必须检查

每天审查结束前，不要只关网页。一定要检查结果文件有没有增加。

### 10.1 Windows PowerShell 检查

以 `reviewer_b` 为例：

```powershell
cd D:\VSCode\Multi-EE-3DAG

(Get-Content .\MultiEEAffordance\processed\annotation_batches\v0_1\reviewer_b_refined_samples.jsonl).Count
(Get-Content .\MultiEEAffordance\processed\annotation_batches\v0_1\reviewer_b_review_records.jsonl).Count
(Get-ChildItem .\MultiEEAffordance\processed\annotation_batches\v0_1\manual_refined_masks_reviewer_b -Filter *.npy -Recurse).Count
```

如果你当天审查了 20 条，`refined_samples` 行数和 mask 数量应该明显增加。

### 10.2 Linux 或 macOS 检查

```bash
cd ~/Multi-EE-3DAG

wc -l MultiEEAffordance/processed/annotation_batches/v0_1/reviewer_b_refined_samples.jsonl
wc -l MultiEEAffordance/processed/annotation_batches/v0_1/reviewer_b_review_records.jsonl
find MultiEEAffordance/processed/annotation_batches/v0_1/manual_refined_masks_reviewer_b -name "*.npy" | wc -l
```

### 10.3 每天记录什么

建议每天写一个简单记录，例如：

```text
MultiEEAffordance/processed/annotation_batches/v0_1/daily_notes_reviewer_b_20260527.md
```

内容可以很简单：

```text
日期：2026-05-27
审查者：reviewer_b
今日完成：35 条
累计完成：120 条

今日确认空标签：
- sample_id=xxx, task=pick_up, executor=hook, reason=没有孔/环/把手

今日不确定样本：
- sample_id=yyy, task=open_pull, executor=suction, question=平面能否吸住后拉开不确定

自动候选典型问题：
- faucet open_pull dexterous_hand 自动候选覆盖主体，实际只应保留上方开关
- scissors pick_up hook 自动候选包含刀刃，已删除

网页或数据问题：
- 无
```

这份记录不需要很正式，但要能让维护者知道哪些样本需要复查。

## 11. 每天或每批如何回传结果

建议审查者每天结束后打一个结果包。

### 11.1 Windows PowerShell 打包

以 `reviewer_b` 为例：

```powershell
cd D:\VSCode\Multi-EE-3DAG

Compress-Archive -Path `
  .\MultiEEAffordance\processed\annotation_batches\v0_1\reviewer_b_review_records.jsonl, `
  .\MultiEEAffordance\processed\annotation_batches\v0_1\reviewer_b_refined_samples.jsonl, `
  .\MultiEEAffordance\processed\annotation_batches\v0_1\manual_refined_masks_reviewer_b, `
  .\MultiEEAffordance\processed\annotation_batches\v0_1\daily_notes_reviewer_b_20260527.md `
  -DestinationPath .\reviewer_b_annotation_v0_1_20260527.zip `
  -Force
```

### 11.2 Linux 或 macOS 打包

```bash
cd ~/Multi-EE-3DAG

tar -czf reviewer_b_annotation_v0_1_20260527.tar.gz \
  MultiEEAffordance/processed/annotation_batches/v0_1/reviewer_b_review_records.jsonl \
  MultiEEAffordance/processed/annotation_batches/v0_1/reviewer_b_refined_samples.jsonl \
  MultiEEAffordance/processed/annotation_batches/v0_1/manual_refined_masks_reviewer_b \
  MultiEEAffordance/processed/annotation_batches/v0_1/daily_notes_reviewer_b_20260527.md
```

### 11.3 回传方式

推荐顺序：

1. 小批量校准：可以把结果放到 GitHub PR 里。
2. 正式批量：把压缩包上传到 GitHub Release 附件、Issue 附件或网盘。
3. 不建议把大量 `.npy` 文件直接提交到普通 Git 仓库。

结果包文件名建议统一：

```text
reviewer_a_annotation_v0_1_YYYYMMDD.zip
reviewer_b_annotation_v0_1_YYYYMMDD.zip
```

维护者收到结果包后，会解压、检查重复样本、合并两位审查者的 refined samples，并生成 reviewed dataset。

## 12. 维护者如何合并两份结果

维护者收到两份结果包后，先分别解压到同一个批次目录：

```text
MultiEEAffordance/processed/annotation_batches/v0_1/
```

然后合并：

```bash
cat \
  MultiEEAffordance/processed/annotation_batches/v0_1/reviewer_a_refined_samples.jsonl \
  MultiEEAffordance/processed/annotation_batches/v0_1/reviewer_b_refined_samples.jsonl \
  > MultiEEAffordance/processed/annotation_batches/v0_1/merged_refined_samples.jsonl
```

合并前检查是否有重复样本：

```bash
python - <<'PY'
import json
from collections import defaultdict
from pathlib import Path

paths = [
    Path("MultiEEAffordance/processed/annotation_batches/v0_1/reviewer_a_refined_samples.jsonl"),
    Path("MultiEEAffordance/processed/annotation_batches/v0_1/reviewer_b_refined_samples.jsonl"),
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
            key = (
                row.get("sample_id"),
                row.get("task"),
                row.get("executor") or row.get("target_executor"),
            )
            seen[key].append((str(path), line_no))

conflicts = {key: locs for key, locs in seen.items() if len(locs) > 1}
print("conflicts:", len(conflicts))
for key, locs in list(conflicts.items())[:50]:
    print(key, locs)
PY
```

如果 `conflicts > 0`，不要自动覆盖。先人工检查冲突样本，只保留一个最终版本。

确认无冲突后再打包 reviewed dataset：

```bash
python MultiEEAffordance/tools/build_reviewed_dataset_release.py \
  --dataset-root MultiEEAffordance \
  --reviewed-samples processed/annotation_batches/v0_1/merged_refined_samples.jsonl \
  --output-samples processed/metadata/reviewed_dataset_v0_1.jsonl \
  --summary-json processed/metadata/reviewed_dataset_summary_v0_1.json \
  --output-split-dir splits_reviewed_v0_1 \
  --include-tasks pick_up,open_pull,press_push \
  --exclude-tasks lift_carry \
  --overwrite
```

## 13. 常见问题

| 问题 | 可能原因 | 怎么处理 |
| --- | --- | --- |
| 网页打不开 | 审查服务没启动，或端口写错 | 看终端是否显示 `Serving ...`，确认浏览器访问 `http://127.0.0.1:8765` |
| 点云不显示 | 数据包没解压对，或缺少 `.npy/.npz/.json` 文件 | 复制终端报错路径，发给维护者 |
| 候选列表为空 | 当前样本没有高置信候选，或候选文件缺失 | 仍可手动点级编辑；如果明显漏召回，记录到 daily notes |
| 保存后没变化 | 没点保存、终端报错、输出路径没有权限 | 看终端日志；检查 `refined_samples` 行数是否增加 |
| 浏览器很卡 | 点云或候选太多 | 降低候选显示数量，或让维护者拆小批次 |
| 不确定该不该标 | 执行器机制不清楚 | 保存为 `uncertain` 或写 daily notes，交给维护者复查 |
| `pos=0` | 当前 mask 为空 | 先判断是否确实没有合适区域；如果确认为空，可以保存空标签 |

## 14. 不要做的事

- 不要自己运行 VLM pipeline。
- 不要自己重新生成候选。
- 不要改 `reviewer_a` / `reviewer_b` 以外的输出路径。
- 不要把别人的结果文件覆盖掉。
- 不要把自动候选直接当成最终 GT。
- 不要为了让样本非空而强行补点。
- 不要把大规模 `.npy/.npz` 文件直接提交到普通 Git 仓库。
- 不要只回传 `review_records.jsonl`，必须同时回传 `refined_samples.jsonl` 和 `manual_refined_masks_*`。

## 15. 当前原始数据说明

当前 v0.1 标注主线的核心原始数据是：

```text
/home/lzq/Multi-EE-3DAG/MultiEEAffordance/raw/3d_affordancenet/full-shape.zip
```

这个文件由维护者使用。它可以转换出当前需要的点云、弱标签、候选样本和审查队列。

审查者通常不需要这个原始 zip。审查者需要的是维护者已经处理好的数据包，里面包含当前批次要审查的点云、候选和样本清单。

简单说：

```text
full-shape.zip = 维护者生成数据用
annotation_batch_v0_1_reviewer_x.zip = 审查者本地标注用
reviewer_x_annotation_v0_1_YYYYMMDD.zip = 审查者回传结果用
```
