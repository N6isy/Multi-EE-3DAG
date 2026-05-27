# 双人协作标注 README

更新时间：2026-05-26

本文档用于当前轻量自动候选 + 人工点级审查版本。两位审查者统一记为 `reviewer_a` 和 `reviewer_b`。当前阶段的目标是先稳定产出人工审查数据，不要求审查者各自重新运行 VLM 或候选生成 pipeline。

## 0. 角色分工与命名原则

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

## 1. 协作方式

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

### 1.1 如果不能 SSH：使用 GitHub + 数据包协作

如果审查者无法 SSH 到服务器，就不能用 SSH tunnel 访问服务器网页。这时仍然可以协作，但方式要换成：

```text
GitHub 管代码和文档
数据包/压缩包管点云、候选和 mask
审查者在自己电脑本地启动网页
审查结果通过 GitHub 分支/PR、GitHub Release 附件、网盘或压缩包回传
```

这和共享服务器模式的区别是：

| 项目 | 共享服务器模式 | GitHub + 数据包模式 |
| --- | --- | --- |
| 网页在哪里跑 | 服务器 | 审查者本地电脑 |
| 点云和候选在哪里 | 服务器 | 审查者本地数据包 |
| 审查者是否需要 SSH | 需要 | 不需要 |
| 审查者是否需要 VLM/GPU | 不需要 | 不需要 |
| 审查者是否需要下载数据 | 不需要或很少 | 需要下载维护者准备好的数据包 |
| 结果怎么回传 | 服务器直接保存 | GitHub PR、Release artifact、网盘或压缩包 |

注意：GitHub 不适合直接存放原始大 zip、VLM 中间结果和大规模点云目录。普通 GitHub 仓库主要放代码、文档、小型 metadata。大文件应使用 GitHub Release 附件、Git LFS、网盘、移动硬盘或其他文件分发方式。

GitHub + 数据包模式的推荐流程：

1. 维护者在服务器端生成候选和样本队列。
2. 维护者把 `reviewer_a` / `reviewer_b` 各自需要的最小数据打包。
3. 审查者从 GitHub 拉代码，切到 `annotation/mvp-v0.1`。
4. 审查者下载自己的数据包，并解压到仓库里的 `MultiEEAffordance/` 下。
5. 审查者在本地启动 `serve_v2_annotation_app.py`。
6. 审查者本地完成标注。
7. 审查者把结果包发回维护者，或者开 PR 提交到约定的结果目录。
8. 维护者合并两份结果并打包 reviewed dataset。

审查者本地只需要能跑审查网页，不需要跑：

- `run_v3_pipeline.py`
- Qwen3-VL
- SAM2
- PartSLIP++
- CUDA/GPU 推理环境

## 2. 拉取稳定标注版本

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

## 3. 标注批次目录

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

### 3.1 本地审查所需的最小数据包

如果使用 GitHub + 数据包模式，审查者本地至少需要这些文件或目录：

```text
MultiEEAffordance/processed/annotation_batches/v0_1/reviewer_b_samples.jsonl
MultiEEAffordance/processed/points/...                         # sample 中 point_cloud_path 指向的点云
MultiEEAffordance/processed/vlm_candidate_v3/fused_masks/...    # sample 中 multi_channel_mask_path 指向的候选 mask
MultiEEAffordance/processed/vlm_candidate_v3/3d_candidates/...   # candidate_manifest 和 candidates.npz
```

如果样本里还引用了 `checked_mask_path`、`source_mask_path` 或 `rule_filter_path`，对应文件也应一起打包。最稳妥的方式是维护者先按 `reviewer_b_samples.jsonl` 中的路径收集依赖，打成一个压缩包。

不要只给审查者 `reviewer_b_samples.jsonl`。这个 JSONL 只是索引，真正显示点云和候选还需要它里面引用的 `.npy/.npz/.json` 文件。

本地解压后，路径应该仍然保持相对 `MultiEEAffordance/` 的结构。例如样本里写的是：

```text
processed/points/xxx/object.npy
```

那么审查者电脑上也应该存在：

```text
MultiEEAffordance/processed/points/xxx/object.npy
```

否则网页会报点云或 mask 路径不存在。

## 4. 样本分配

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

## 5. 启动审查系统

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

### 5.1 本地模式启动命令

如果审查者不能 SSH，而是在自己电脑上解压了数据包，那么审查者可以在本地仓库根目录启动网页。下面以 `reviewer_b` 为例：

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

然后本地浏览器打开：

```text
http://127.0.0.1:8766
```

如果网页能打开但点云不显示，通常不是网络问题，而是数据包缺少点云、候选 npz 或 mask 文件。把报错里的路径发给维护者即可。

## 6. 审查操作规范

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

对于多执行器数据集，空 mask 有意义。它表示该物体在当前任务和执行器组合下没有合适 affordance 区域。不要为了让 `pos>0` 而强行标注。

典型例子：

- `open_pull + suction` 可能没有稳定可吸并能拉开的区域；
- `pick_up + hook` 如果没有孔、环、把手或可挂接结构，应为空；
- `press_push + hook` 通常不合理，应由任务/执行器规则或人工确认为空；
- `open_pull` 中主体外壳、普通平滑面、装饰边缘通常不是正例。

### 6.3 需要重点记录的问题样本

审查时遇到以下情况，需要在每日交付中列出来：

- 自动默认正例过大，例如把 faucet 主体、door panel、scissors blade 全选进去；
- 候选完全为空，但肉眼能看到明显 handle/button/loop；
- 同一物体不同执行器的结果非常矛盾；
- 点云过稀导致无法判断；
- 页面保存成功但可视化结果明显不对。

## 7. 保存后检查

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

## 8. 合并与 release

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

### 8.1 GitHub 模式下如何回传结果

如果审查者不能 SSH，而是在本地电脑完成标注，推荐按“代码走 GitHub，数据走压缩包”的方式回传。

小规模试运行时，可以把结果放到一个单独目录后开 PR：

```text
annotation_results/v0_1/reviewer_b/
  reviewer_b_review_records.jsonl
  reviewer_b_refined_samples.jsonl
  manual_refined_masks_reviewer_b/
```

这种方式适合十几条、几十条样本的校准批次。维护者可以直接看 diff、检查文件数量，并把结果合并回服务器。

正式批量标注时，不建议频繁把大量 `.npy` mask 文件提交到普通 Git 仓库。更稳妥的做法是审查者打包自己的结果，再通过 GitHub Release 附件、网盘或其他文件传输方式发回维护者：

```bash
tar -czf reviewer_b_annotation_v0_1_YYYYMMDD.tar.gz \
  MultiEEAffordance/processed/annotation_batches/v0_1/reviewer_b_review_records.jsonl \
  MultiEEAffordance/processed/annotation_batches/v0_1/reviewer_b_refined_samples.jsonl \
  MultiEEAffordance/processed/annotation_batches/v0_1/manual_refined_masks_reviewer_b
```

维护者收到后解压到服务器的同名批次目录，再执行合并和 release。这样 GitHub 只负责代码、文档和小型索引文件，大体积点云、候选、人工 mask 不会把仓库变得很重。

## 9. 每日交付清单

每位审查者每天至少同步：

- 已审查数量；
- 遇到的典型错误候选；
- 是否有 `pos=0` 但不确定的样本；
- 自己的 `*_review_records.jsonl`；
- 自己的 `*_refined_samples.jsonl`；
- 自己的 `manual_refined_masks_*` 目录。

如果审查页面出现 500、点云不显示、保存后行数不增加，应立即暂停该样本并记录 `sample_id / task / executor / pilot_id`。

## 10. 常见问题

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

- 不要在审查过程中重跑同名 `v3_candidate_samples_v0_1.jsonl`。
- 不要把 `processed/annotation_batches/` 提交到 Git。
- 不要两个人共用同一个端口和同一个输出路径。
- 不要为了让样本非空而强行补点。
- 不要把自动候选直接当作 GT。
- 不要在没有记录的情况下删除或覆盖 refined mask。

## 12. 当前原始数据说明

当前 v0.1 标注主线使用的核心原始数据是：

```text
/home/lzq/Multi-EE-3DAG/MultiEEAffordance/raw/3d_affordancenet/full-shape.zip
```

它是当前 3D AffordanceNet full-shape 流程的主要来源。转换脚本会从里面读取 `full_shape_train_data.pkl` 或 `full_shape_val_data.pkl`，得到完整物体点云和原始 affordance 弱标签，再展开成我们现在的 `pick_up / open_pull / press_push` 审查样本。

需要注意三点：

1. 对当前 v3 大规模标注主线来说，`full-shape.zip` 足够作为 3D AffordanceNet 样本池的核心原始数据。
2. 它不是整个项目里所有可能数据的合集。`partial.zip`、`rotate.zip`、PartNet-Mobility 等数据如果后续接入，会是额外数据源或补充实验源。
3. 它不包含已经生成好的 VLM 语义计划、候选区域、人工 refined mask 和 reviewed dataset。这些都是后续 pipeline 运行产物，保存在 `processed/`、`manifests/`、`splits/` 等目录中。

因此，如果审查者只是做人工标注，通常不需要下载完整 `full-shape.zip`，也不需要自己重新跑转换和候选生成。维护者应该提前生成候选，并给审查者准备最小数据包。审查者本地只需要能打开点云、候选和保存 refined mask。
