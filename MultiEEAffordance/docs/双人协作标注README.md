# 双人协作标注 README

更新时间：2026-05-26

本文档用于当前轻量自动候选 + 人工点级审查版本。两位审查者统一记为 `reviewer_a` 和 `reviewer_b`。当前阶段的目标是先稳定产出人工审查数据，不要求审查者各自重新运行 VLM 或候选生成 pipeline。

## 0. 角色分工与命名原则

这一节先说明“谁负责什么”。如果角色分工不清楚，最容易出现的问题是：两个人改同一份输出、候选文件被覆盖、或者审查者误以为自己还要跑 VLM pipeline。当前协作的基本原则是：维护者准备数据和网页，审查者只做网页中的人工判断和保存。

当前协作中只区分三类职责：

| 角色 | 主要职责 | 不建议做的事 |
| --- | --- | --- |
| 维护者 | 生成候选、拆分审查队列、启动网页服务、合并 release、处理异常样本 | 在标注批次进行中随意重跑同名候选文件 |
| `reviewer_a` | 审查分配给自己的样本，保存 refined mask 和审查记录 | 写入 `reviewer_b` 的输出目录 |
| `reviewer_b` | 审查分配给自己的样本，保存 refined mask 和审查记录 | 写入 `reviewer_a` 的输出目录 |

命名原则：

- 文档、目录、端口、输出文件只使用 `reviewer_a` / `reviewer_b`。
- 不在当前协作流程中使用个人身份作为文件名。
- 同一个标注批次统一使用 `v0_1` 后缀；下一批重新生成候选时再使用 `v0_2`。
- `processed/annotation_batches/` 是批量标注产物目录，不进 Git，需要在服务器端备份。

简单理解：

- `reviewer_a_samples.jsonl` 是 `reviewer_a` 要看的样本清单。
- `reviewer_a_refined_samples.jsonl` 是 `reviewer_a` 保存后的结果索引。
- `manual_refined_masks_reviewer_a/` 是 `reviewer_a` 真正保存的 refined mask 文件夹。
- `reviewer_b` 同理，所有路径都换成 `reviewer_b`。
- 两边不要混用。混用后即使网页能打开，后续也很难知道某个 mask 到底是谁审的。

## 1. 协作方式

这一节说明“我们为什么采用共享服务器协作”。当前数据、候选文件、VLM 输出、点云和 mask 都比较大，让每位审查者各自在本地完整跑一套环境，会浪费时间，也容易因为环境不同导致结果不一致。因此推荐维护者在服务器统一准备数据，两位审查者通过网页完成判断。

推荐使用共享服务器模式：

1. 维护者在服务器上生成候选、审查队列和候选 mask。
2. 两位审查者分别访问各自端口上的审查网页。
3. 两位审查者写入不同的 review JSONL、refined samples JSONL 和 refined mask 目录。
4. 每轮结束后再合并两份人工审查结果，生成 reviewed dataset release。

不要让两位审查者同时写入同一个 `review_records.jsonl` 或同一个 `manual_refined_masks` 目录，否则容易覆盖或混入未排查的写入冲突。

完整流程是：

```text
维护者生成 v3_candidate_samples_v0_1.jsonl
  -> 按 object_id 拆成 reviewer_a_samples / reviewer_b_samples
  -> 启动 8765 和 8766 两个审查服务
  -> reviewer_a / reviewer_b 分别审查并保存
  -> 每日检查 refined_samples 行数和 mask 文件数
  -> 合并 refined_samples
  -> 检查重复 sample_id + executor
  -> 打包 reviewed_dataset_v0_1
  -> 用只读 viewer 抽检
```

每一步的含义：

- 生成候选：自动 pipeline 先给出可能有用的区域，但它不是最终真值。
- 拆分样本：把任务分给两位审查者，避免重复劳动。
- 启动两个服务：两个网页分别写入不同输出目录。
- 分别审查：审查者在网页里选候选、删错点、补漏点。
- 每日检查：确认保存真的落盘，不是只在浏览器里看到了变化。
- 合并结果：把两个人的 refined samples 合成一个待发布版本。
- 检查重复：如果两个人都审了同一条，不自动覆盖。
- 打包 release：生成后续训练/统计/可视化使用的数据集文件。
- 抽检：用只读 viewer 看最终效果，发现大面积错误再回到审查页面修。

## 2. 拉取稳定标注版本

这一节是给审查者看的。审查者需要拿到项目代码，是为了知道当前使用哪一版网页和文档；但审查者不需要自己跑 VLM，也不需要重新生成候选。真正的标注入口是维护者启动好的网页地址。

审查者只需要拉取仓库并切换到稳定标注分支：

```bash
git clone <repo_url>
cd Multi-EE-3DAG
git checkout annotation/mvp-v0.1
```

如果仓库已经存在：

```bash
cd Multi-EE-3DAG
git fetch origin
git checkout annotation/mvp-v0.1
git pull
```

标注期间不要随意切换到候选生成研发分支，避免审查页面、候选 manifest 和样本 JSONL 版本不一致。

审查者只需要浏览器能访问网页即可。如果审查者不需要改代码，可以不在本地配置完整 VLM 环境；真正运行 VLM 和候选生成的环境保留在服务器端。

检查自己是否在正确分支：

```bash
git branch --show-current
```

应该看到：

```text
annotation/mvp-v0.1
```

如果不是这个分支，先不要开始标注。因为不同分支的网页脚本、样本字段和文档可能不一致。

## 3. 标注批次目录

这一节说明“所有人工审查结果放在哪里”。审查工作不是只保存一个网页状态，而是会产生两类重要文件：一个是 `.jsonl` 记录，说明审查了哪些样本；另一个是 `.npy` mask，保存点级标注结果。后续打包数据集时，这两类文件都需要。

当前双人协作批次建议统一放在：

```text
MultiEEAffordance/processed/annotation_batches/v0_1/
  reviewer_a_samples.jsonl
  reviewer_b_samples.jsonl
  reviewer_a_review_records.jsonl
  reviewer_b_review_records.jsonl
  reviewer_a_refined_samples.jsonl
  reviewer_b_refined_samples.jsonl
  manual_refined_masks_reviewer_a/
  manual_refined_masks_reviewer_b/
```

`processed/annotation_batches/` 是运行产物目录，不建议提交到 Git。它需要在服务器上备份或打包保存。

建议维护者在批次目录中额外保存一个简短说明文件，例如：

```text
processed/annotation_batches/v0_1/BATCH_INFO.md
```

记录：

- 使用的 Git 分支和 tag；
- 使用的 `v3_candidate_samples_v0_1.jsonl` 生成时间；
- 样本总数、`reviewer_a` 数量、`reviewer_b` 数量；
- 本批保留的任务：`pick_up,open_pull,press_push`；
- 本批排除的任务：`lift_carry`；
- 审查开始日期和计划完成日期。

`BATCH_INFO.md` 可以写成这样：

```text
# annotation batch v0_1

git_branch: annotation/mvp-v0.1
git_tag: annotation-mvp-v0.1
candidate_samples: processed/metadata/v3_candidate_samples_v0_1.jsonl
tasks: pick_up, open_pull, press_push
excluded_tasks: lift_carry
reviewer_a_samples: 150
reviewer_b_samples: 150
start_date: 2026-05-26
notes: first dual-reviewer annotation batch
```

这个文件不是给程序读的，而是给人回头查的。几周后再看这批数据时，可以马上知道它是哪次标注、用的哪版代码、由谁负责哪部分。

## 4. 样本分配

这一节是维护者准备任务时使用的。样本分配的核心目标不是平均分行数，而是减少审查者在不同物体之间来回切换。因为同一个物体可能有多个 `task / executor` 组合，放在一起审查更容易理解。

推荐按 `object_id` 分组分配，尽量保证同一个物体的不同任务和执行器变体由同一位审查者处理。这样审查者看到物体后，可以连续判断它在不同 `task / executor` 下是否有合适区域。

示例：从完整候选样本文件按物体分组拆成两份：

```bash
mkdir -p MultiEEAffordance/processed/annotation_batches/v0_1
python - <<'PY'
import json
from collections import defaultdict
from pathlib import Path

root = Path("MultiEEAffordance")
src = root / "processed/metadata/v3_candidate_samples_v0_1.jsonl"
out = root / "processed/annotation_batches/v0_1"
groups = defaultdict(list)

with src.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        groups[str(row.get("object_id") or row.get("sample_id"))].append(row)

reviewer_rows = {"reviewer_a": [], "reviewer_b": []}
for idx, object_id in enumerate(sorted(groups)):
    reviewer = "reviewer_a" if idx % 2 == 0 else "reviewer_b"
    reviewer_rows[reviewer].extend(groups[object_id])

for reviewer, rows in reviewer_rows.items():
    path = out / f"{reviewer}_samples.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(reviewer, len(rows), path)
PY
```

如果要做一致性校准，可以额外准备一个两人都审查的 overlap 文件，例如 `calibration_overlap_samples.jsonl`。重叠样本不要直接合并进正式 release，应先比较差异。

样本分配完成后，维护者应该确认两件事：

1. 两个文件都有内容，不要一个文件为空。
2. 同一个 `object_id` 没有被拆到两边，除非它是专门用于校准的 overlap 样本。

可以简单抽查：

```bash
head -n 2 MultiEEAffordance/processed/annotation_batches/v0_1/reviewer_a_samples.jsonl
head -n 2 MultiEEAffordance/processed/annotation_batches/v0_1/reviewer_b_samples.jsonl
```

看到每行都是一条 JSON 记录即可。不要手动编辑 JSONL 的内容，除非非常确定字段格式。

推荐第一次正式协作前做一个小校准：

1. 从不同类别中挑 10 条样本。
2. 两位审查者各自独立审查同一批样本。
3. 比较以下差异：
   - 是否都判断为空 mask；
   - 是否都选择相同候选族；
   - 是否都删除了自动候选中的明显错误主体区域；
   - 小部件、孔洞、把手边界是否保留一致。
4. 维护者根据差异补充标注规范，再开始大批量分配。

## 4.1 维护者启动前检查

这一节是“开工前体检”。它的作用是提前发现路径缺失、样本文件为空、候选目录不存在等问题。不要等审查者打开网页后才发现点云不显示。

在启动两位审查者的网页前，维护者先检查输入文件和候选文件是否存在：

```bash
test -f MultiEEAffordance/processed/metadata/v3_candidate_samples_v0_1.jsonl
test -f MultiEEAffordance/processed/annotation_batches/v0_1/reviewer_a_samples.jsonl
test -f MultiEEAffordance/processed/annotation_batches/v0_1/reviewer_b_samples.jsonl
test -d MultiEEAffordance/processed/vlm_candidate_v3/3d_candidates
```

检查两份样本数量：

```bash
wc -l MultiEEAffordance/processed/annotation_batches/v0_1/reviewer_a_samples.jsonl
wc -l MultiEEAffordance/processed/annotation_batches/v0_1/reviewer_b_samples.jsonl
```

抽查样本字段：

```bash
python - <<'PY'
import json
from pathlib import Path

for name in ["reviewer_a_samples.jsonl", "reviewer_b_samples.jsonl"]:
    path = Path("MultiEEAffordance/processed/annotation_batches/v0_1") / name
    with path.open("r", encoding="utf-8") as f:
        row = json.loads(next(f))
    print(name)
    print("  sample_id:", row.get("sample_id"))
    print("  object:", row.get("object_category"))
    print("  task:", row.get("task"))
    print("  executor:", row.get("executor"))
    print("  mask:", row.get("multi_channel_mask_path"))
PY
```

如果 `sample_id / task / executor / multi_channel_mask_path` 缺失，不要开始审查，先回到候选构建阶段排查。

检查通过后，维护者再启动网页服务。建议把两个服务分别放在两个终端或两个 tmux 窗口里，这样某个服务报错时能看到对应日志。

## 5. 启动审查系统

这一节说明“怎么把网页跑起来”。审查系统是一个本地 Web 服务，默认只在服务器的 `127.0.0.1` 上监听。审查者如果不在服务器桌面上操作，就需要通过 SSH 端口转发把网页映射到自己电脑。

维护者需要启动两个服务：

- `8765` 给 `reviewer_a`
- `8766` 给 `reviewer_b`

端口不是身份本身，但固定下来以后不容易混乱。以后如果端口被占用，可以换成其他端口，但要同步告诉对应审查者。

### reviewer_a

```bash
python MultiEEAffordance/tools/serve_v2_annotation_app.py \
  --dataset-root MultiEEAffordance \
  --samples processed/annotation_batches/v0_1/reviewer_a_samples.jsonl \
  --review-jsonl processed/annotation_batches/v0_1/reviewer_a_review_records.jsonl \
  --output-mask-root processed/annotation_batches/v0_1/manual_refined_masks_reviewer_a \
  --output-samples processed/annotation_batches/v0_1/reviewer_a_refined_samples.jsonl \
  --port 8765 \
  --top-k-candidates 8 \
  --candidate-min-selected-votes 2
```

### reviewer_b

```bash
python MultiEEAffordance/tools/serve_v2_annotation_app.py \
  --dataset-root MultiEEAffordance \
  --samples processed/annotation_batches/v0_1/reviewer_b_samples.jsonl \
  --review-jsonl processed/annotation_batches/v0_1/reviewer_b_review_records.jsonl \
  --output-mask-root processed/annotation_batches/v0_1/manual_refined_masks_reviewer_b \
  --output-samples processed/annotation_batches/v0_1/reviewer_b_refined_samples.jsonl \
  --port 8766 \
  --top-k-candidates 8 \
  --candidate-min-selected-votes 2
```

如果审查者从本地电脑访问服务器，可使用 SSH 端口转发：

```bash
ssh -N -L 8765:127.0.0.1:8765 <user>@<server_host>
ssh -N -L 8766:127.0.0.1:8766 <user>@<server_host>
```

然后在本地浏览器打开：

```text
http://127.0.0.1:8765
http://127.0.0.1:8766
```

启动后，维护者先自己打开两个页面，分别确认：

- 左侧样本数符合 `reviewer_a_samples.jsonl` / `reviewer_b_samples.jsonl`；
- 点击第一条样本可以显示点云；
- 右侧候选列表能显示候选或空标签说明；
- 点击保存后，对应的 `*_refined_samples.jsonl` 行数增加；
- refined mask 写入对应的 `manual_refined_masks_reviewer_*` 目录。

审查者拿到网页地址后，只需要做三件事：

1. 打开对应 URL。
2. 确认左侧列表里是自己的样本。
3. 按第 6 节规则逐条审查并保存。

审查者不要更换 URL 里的端口，也不要打开另一个审查者的端口进行保存。

## 6. 审查操作规范

这一节是整份文档最重要的部分。自动候选只是“建议区域”，不是最终答案。审查者要做的是判断当前任务和执行器到底应该作用在哪些点上，然后把自动候选修成更可信的 refined mask。

每个样本按以下顺序处理：

1. 先确认页面顶部显示的 `object / task / executor`，不要只看物体形状。
2. 查看自动勾选的候选组合是否符合当前执行器机制。
3. 打开候选预览，必要时补选 `need_review=true` 或低置信但位置合理的候选。
4. 用圆形 brush 做点级删除或补点。
5. 如果没有合适区域，保留空 mask 并保存为需要确认的空标签样本，不要强行标注。

常见判断：

- `pick_up + gripper`：优先可夹持、可稳定约束的两侧或局部结构。
- `pick_up + hook`：优先孔、环、内边界、可挂接结构；普通长边、尖端、刀刃不是正例。
- `open_pull`：优先把手、旋钮、可拉开的活动部件；主体面通常不是正例。
- `press_push + suction/dexterous_hand`：优先按钮、平面面板、可按压区域；不要把整物体主体都标为正例。

网页中常见元素的含义：

| 页面元素 | 含义 | 审查者要做什么 |
| --- | --- | --- |
| 左侧样本列表 | 当前分配给你的样本 | 从上到下逐条处理 |
| 顶部 `object/task/executor` | 当前物体、任务和执行器 | 每次保存前都看一眼，避免按错任务标 |
| 右侧候选列表 | 自动候选区域 | 勾选合理候选，取消明显错误候选 |
| 点云中的蓝色/高亮点 | 当前 mask 正例点 | 判断它们是否真是目标 affordance |
| brush 圆圈 | 点级增删工具 | 左键增删点，右键旋转视角 |
| 保存按钮 | 写入 refined mask 和记录 | 每条审完都要保存 |

一个样本的推荐审查顺序：

1. 先旋转点云，看清楚物体结构。
2. 看顶部任务和执行器，例如 `Faucet / open_pull / dexterous_hand`。
3. 看当前蓝色正例点是否大致落在目标部件上。
4. 如果蓝色点覆盖主体太多，先取消右侧明显过大的候选。
5. 如果缺少目标部件，用 brush 补点。
6. 如果目标区域很小，宁可保守一点，也不要把一大片无关主体标进去。
7. 保存。

### 6.1 审查状态建议

审查者保存时应把样本理解为以下几类：

| 情况 | 建议处理 |
| --- | --- |
| 自动候选基本正确 | 保留候选，少量删除错误点，保存 refined mask |
| 自动候选过大 | 先取消不合理候选，再用 brush 删除主体误选区域 |
| 自动候选漏掉小部件 | 手动补点，必要时记录该样本用于后续候选生成器改进 |
| 没有任何合适区域 | 保存空 mask，视为当前 `object + task + executor` 不可执行或暂无可标区域 |
| 不确定 | 先保存当前最保守结果，并在每日记录中列出 `sample_id / executor / 疑问` |

### 6.2 空 mask 不是失败

这一节需要特别注意。很多审查者第一次标数据时会下意识觉得“每条样本都应该有正例”。但 Multi-EE 的价值之一就是区分不同执行器是否适合某个任务，所以空 mask 是正常且有意义的标签。

对于多执行器数据集，空 mask 有意义。它表示该物体在当前任务和执行器组合下没有合适 affordance 区域。不要为了让 `pos>0` 而强行标注。

典型例子：

- `open_pull + suction` 可能没有稳定可吸并能拉开的区域；
- `pick_up + hook` 如果没有孔、环、把手或可挂接结构，应为空；
- `press_push + hook` 通常不合理，应由任务/执行器规则或人工确认为空；
- `open_pull` 中主体外壳、普通平滑面、装饰边缘通常不是正例。

### 6.3 需要重点记录的问题样本

这一节是为了反向改进自动候选生成器。审查者不需要修代码，但需要把典型问题记录出来。后续维护者会根据这些问题样本调整候选生成、VLM 选择或规则过滤。

审查时遇到以下情况，需要在每日交付中列出来：

- 自动默认正例过大，例如把 faucet 主体、door panel、scissors blade 全选进去；
- 候选完全为空，但肉眼能看到明显 handle/button/loop；
- 同一物体不同执行器的结果非常矛盾；
- 点云过稀导致无法判断；
- 页面保存成功但可视化结果明显不对。

问题样本记录建议格式：

```text
sample_id:
object/task/executor:
问题类型: 候选过大 / 候选为空 / 点云太稀 / 任务不确定 / 保存异常
简短说明:
是否已保存:
```

## 7. 保存后检查

这一节说明“怎么确认今天的工作真的保存下来了”。浏览器里看到颜色变化不等于文件已经正确落盘。每天结束前都要看 JSONL 行数和 `.npy` 文件数。

每位审查者每天结束前检查：

```bash
wc -l MultiEEAffordance/processed/annotation_batches/v0_1/reviewer_a_refined_samples.jsonl
wc -l MultiEEAffordance/processed/annotation_batches/v0_1/reviewer_b_refined_samples.jsonl
find MultiEEAffordance/processed/annotation_batches/v0_1/manual_refined_masks_reviewer_a -name "*.npy" | wc -l
find MultiEEAffordance/processed/annotation_batches/v0_1/manual_refined_masks_reviewer_b -name "*.npy" | wc -l
```

如果 `refined_samples.jsonl` 行数增加，但 refined mask 数量没有增加，需要检查保存是否报错。

建议每天结束后再备份一次批次目录：

```bash
tar -czf MultiEEAffordance/processed/annotation_batches/v0_1_backup_$(date +%Y%m%d).tar.gz \
  MultiEEAffordance/processed/annotation_batches/v0_1
```

如果服务器空间紧张，至少备份这三类文件：

```text
reviewer_a_review_records.jsonl
reviewer_b_review_records.jsonl
reviewer_a_refined_samples.jsonl
reviewer_b_refined_samples.jsonl
manual_refined_masks_reviewer_a/
manual_refined_masks_reviewer_b/
```

行数和文件数不一定完全相等，因为同一个样本可能多次保存，JSONL 会记录多次保存历史；但如果一天审了很多条，而 refined mask 目录没有新增文件，就要立刻检查。

## 8. 合并与 release

这一节是维护者在一批审查结束后使用的。合并不是简单把文件拼起来就结束，还要检查是否有重复样本、路径是否存在、release 是否能被可视化系统打开。

合并两位审查者的 refined samples：

```bash
cat \
  MultiEEAffordance/processed/annotation_batches/v0_1/reviewer_a_refined_samples.jsonl \
  MultiEEAffordance/processed/annotation_batches/v0_1/reviewer_b_refined_samples.jsonl \
  > MultiEEAffordance/processed/annotation_batches/v0_1/merged_refined_samples.jsonl
```

打包 reviewed dataset：

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

如果两位审查者都标注了同一个 `sample_id + executor`，不要自动覆盖。应把该样本列入 conflict review，人工比较后只保留一个最终版本。

合并前可以先检查重复键：

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

如果 `conflicts > 0`，先人工处理冲突，再生成 `merged_refined_samples.jsonl`。

打包 release 后启动只读可视化系统抽检：

```bash
python MultiEEAffordance/tools/serve_reviewed_dataset_viewer.py \
  --dataset-root MultiEEAffordance \
  --samples processed/metadata/reviewed_dataset_v0_1.jsonl \
  --port 8780
```

抽检重点：

- 每个 object 下是否能切换不同 `task / executor`；
- `pos=0` 的样本是否确实应该为空；
- 大主体误标是否明显减少；
- 小部件、把手、孔洞、按钮是否保留；
- mask 是否和当前执行器机制一致。

如果抽检发现问题，不要直接改 release 文件。应回到对应审查者的 refined sample，重新在审查页面修正，然后再次合并和打包 release。

## 9. 每日交付清单

这一节是为了让协作可追踪。每天不需要写长报告，但至少要让维护者知道完成了多少、遇到了什么问题、输出文件是否保存好。

每位审查者每天至少同步：

- 已审查数量；
- 遇到的典型错误候选；
- 是否有 `pos=0` 但不确定的样本；
- 自己的 `*_review_records.jsonl`；
- 自己的 `*_refined_samples.jsonl`；
- 自己的 `manual_refined_masks_*` 目录。

如果审查页面出现 500、点云不显示、保存后行数不增加，应立即暂停该样本并记录 `sample_id / task / executor / pilot_id`。

建议每日消息格式：

```text
reviewer: reviewer_a
date: 2026-05-26
checked_samples: 35
empty_masks: 8
uncertain_samples: 3
problem_samples:
  - sample_id=..., issue=候选过大
  - sample_id=..., issue=候选为空但有明显把手
outputs:
  - reviewer_a_review_records.jsonl 已更新
  - reviewer_a_refined_samples.jsonl 已更新
  - manual_refined_masks_reviewer_a/ 已更新
```

## 10. 常见问题

这一节是遇到问题时先看的排查表。大多数问题不需要立刻改代码，先记录样本、检查路径和服务日志。

| 问题 | 可能原因 | 处理方式 |
| --- | --- | --- |
| 页面打不开 | 服务没启动、端口没转发、防火墙限制 | 维护者检查服务日志；审查者检查 SSH tunnel |
| 点云不显示 | 样本路径、mask 路径或候选 manifest 缺失 | 记录 `sample_id`，维护者检查 JSONL 对应路径 |
| 候选为空但物体明显有可用部件 | 自动候选生成漏召回 | 可以手动补点，并把样本列入候选生成器改进清单 |
| 自动候选覆盖大主体 | VLM/规则默认正例过激或候选过大 | 取消大候选，手动保留目标部件，记录问题样本 |
| 保存后左侧状态没变化 | 浏览器未刷新或保存请求失败 | 看终端日志；检查 refined samples 行数 |
| 两人都审了同一条 | 样本拆分或 overlap 校准导致 | 不自动合并，进入 conflict review |
| 浏览器很卡 | 点数或候选显示太多 | 降低 `--top-k-candidates` 或让维护者拆小批次 |

## 11. 不要做的事

这一节列的是容易破坏整批数据可复现性的操作。标注工作最怕“看起来只是小改一下”，但后续不知道结果来自哪个版本。

- 不要在审查过程中重跑同名 `v3_candidate_samples_v0_1.jsonl`。
- 不要把 `processed/annotation_batches/` 提交到 Git。
- 不要两个人共用同一个端口和同一个输出路径。
- 不要为了让样本非空而强行补点。
- 不要把自动候选直接当作 GT。
- 不要在没有记录的情况下删除或覆盖 refined mask。
