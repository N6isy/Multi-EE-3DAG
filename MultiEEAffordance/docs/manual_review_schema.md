# 人工审查表填写说明

本文件说明 `processed/metadata/manual_review_v0_1.csv` 的填写方式。该表用于人工可视化审查当前 v3 弱标签样本，并为后续自动修正 metadata、删除问题样本、关闭错误执行器通道或标记待精修 mask 做准备。

## 审查表位置

```text
processed/metadata/manual_review_v0_1.csv
```

每一行对应一个 object-task 样本。

## 推荐审查流程

### 方式 A：网页表单审查

推荐使用本地审查网页。它可以在浏览器中查看点云和四通道 mask，并直接把表单结果保存回 `manual_review_v0_1.csv`。

在 VSCode PowerShell 终端运行：

```powershell
cd D:\VSCode\Multi-EE-3DAG
conda activate multieeaffordance

python MultiEEAffordance\tools\serve_review_app.py `
  --dataset-root MultiEEAffordance `
  --host 127.0.0.1 `
  --port 8765 `
  --max-points 4096
```

然后在浏览器打开：

```text
http://127.0.0.1:8765/
```

网页中点击“保存当前样本”或“保存并下一个”后，会自动更新：

```text
processed/metadata/manual_review_v0_1.csv
```

网页右侧表单已经内置中文注解，包括：

- 样本级判断说明；
- 每个字段的填写含义；
- gripper、suction、hook、dexterous_hand 四个执行器的审查重点；
- 下拉选项的英文值和中文解释。

如果服务已经在运行，修改脚本后需要停止服务并重新运行上面的启动命令，然后刷新浏览器页面。

### 方式 B：直接编辑 CSV

如果不启动网页服务，也可以直接编辑 CSV。

1. 打开 HTML 可视化索引：

```text
processed/visualizations/html_v3/index.html
```

2. 在浏览器中打开某个样本页面。
3. 依次查看 `raw`、`gripper`、`suction`、`hook`、`dexterous_hand` 五个视图。
4. 在 `manual_review_v0_1.csv` 中找到同名 `sample_id`。
5. 填写样本级判断和每个执行器通道判断。

## 样本级字段

| 字段 | 是否需要填写 | 说明 |
| --- | --- | --- |
| `sample_id` | 不改 | 样本唯一 ID。 |
| `object_id` | 不改 | 物体 ID。 |
| `object_category` | 不改 | 物体类别。 |
| `task` | 不改 | 当前任务。 |
| `visualization_html_path` | 不改 | 对应 HTML 可视化文件路径。 |
| `point_cloud_path` | 不改 | 点云文件路径。 |
| `multi_channel_mask_path` | 不改 | 四通道 mask 文件路径。 |
| `review_status` | 必填 | 审查状态。 |
| `keep_sample` | 必填 | 是否保留该 object-task 样本。 |
| `quality_after_review` | 建议填 | 审查后的质量等级。 |
| `sample_issue_type` | 有问题时填 | 样本级问题类型。 |
| `sample_notes` | 建议填 | 样本级备注。 |

### `review_status`

推荐取值：

| 取值 | 含义 |
| --- | --- |
| `pending` | 尚未审查。默认值。 |
| `checked` | 已检查，基本可用。 |
| `needs_fix` | 已检查，但需要修正规则或 mask。 |
| `reject` | 样本不适合当前任务，应删除或暂时不用。 |

### `keep_sample`

推荐取值：

| 取值 | 含义 |
| --- | --- |
| `yes` | 保留该样本。 |
| `no` | 删除或暂时排除该样本。 |
| `maybe` | 暂时不确定，需要二次检查。 |

### `quality_after_review`

推荐取值：

| 取值 | 含义 |
| --- | --- |
| `weak` | 仍然只是弱标签，未充分确认。 |
| `checked` | 已人工看过，明显问题较少。 |
| `verified` | 可作为高质量评测或强对比样本。 |

### `sample_issue_type`

推荐取值：

| 取值 | 含义 |
| --- | --- |
| `none` | 没有明显样本级问题。 |
| `task_mismatch` | 物体和任务不匹配，例如普通耳机的 `open_pull`。 |
| `bad_geometry` | 点云形状异常、缺失严重或不易判断。 |
| `ambiguous_object` | 类别或结构不清晰。 |
| `all_masks_bad` | 四个执行器通道都明显不合理。 |
| `needs_partnet` | 仅靠 3D AffordanceNet 不够，需要 PartNet-Mobility 部件补充。 |

## 执行器级字段

每个执行器都有三列：

```text
{executor}_decision
{executor}_issue_type
{executor}_notes
```

其中 `{executor}` 包括：

```text
gripper
suction
hook
dexterous_hand
```

### `{executor}_decision`

推荐取值：

| 取值 | 含义 |
| --- | --- |
| `keep` | 当前通道基本合理，保留。 |
| `disable` | 当前执行器不应可行，应把该通道清零，并将 feasibility 设为 false。 |
| `refine` | 当前通道有一定合理性，但区域需要人工精修或规则修正。 |
| `add_missing` | 当前执行器应该可行，但现有 mask 缺失。 |
| `not_applicable` | 当前任务下该执行器本来就不适用，且当前 mask 为空或可忽略。 |
| `uncertain` | 暂时不确定。 |

### `{executor}_issue_type`

推荐取值：

| 取值 | 含义 |
| --- | --- |
| `none` | 没有明显问题。 |
| `over_label` | 标得过大，例如 suction 覆盖了几乎整个物体。 |
| `under_label` | 标得过小，明显漏掉可操作区域。 |
| `wrong_region` | 标到了错误位置。 |
| `task_mismatch` | 对当前任务不合理。 |
| `executor_mismatch` | 对该执行器不合理，例如 hook 没有孔洞或挂接结构。 |
| `too_noisy` | mask 零散、噪声大。 |
| `missing_positive` | 应该有正样本但当前为空。 |
| `needs_geometry_rule` | 需要法向、曲率、平面检测等几何规则补充。 |
| `needs_part_annotation` | 需要部件标注补充，例如 handle、button、ring、hole。 |

## 填写示例

### 示例 1：样本整体可用，但 suction 过大

适用于类似 `Door + open_pull`，其中 gripper/hook/dexterous_hand 都集中在把手附近，但 suction 覆盖了几乎整块门板。

```text
review_status = needs_fix
keep_sample = yes
quality_after_review = weak
sample_issue_type = none
sample_notes = suction 面板区域过大，后续需要几何平面和接触面积规则筛选

gripper_decision = keep
gripper_issue_type = none

suction_decision = refine
suction_issue_type = over_label
suction_notes = 当前 suction 覆盖几乎整块门板，可能需要限制为平整可吸附区域或人工精修

hook_decision = refine
hook_issue_type = executor_mismatch
hook_notes = 当前 hook 来自 pull/openable，未必是真正孔洞或挂接结构，需要 PartNet-Mobility 结构验证

dexterous_hand_decision = keep
dexterous_hand_issue_type = none
```

### 示例 2：任务不匹配，应删除样本

适用于物体和任务明显不匹配的情况。

```text
review_status = reject
keep_sample = no
quality_after_review = weak
sample_issue_type = task_mismatch
sample_notes = 当前物体没有打开或拉开的语义，不适合作为 open_pull 样本
```

执行器列可以简单填：

```text
gripper_decision = disable
suction_decision = disable
hook_decision = disable
dexterous_hand_decision = disable
```

### 示例 3：pick_up 样本基本可用

适用于把手、杯身、瓶身、刀柄等区域较清晰的样本。

```text
review_status = checked
keep_sample = yes
quality_after_review = checked
sample_issue_type = none
sample_notes = pick_up 弱标签区域与可抓取区域基本一致

gripper_decision = keep
gripper_issue_type = none

suction_decision = not_applicable
suction_issue_type = none

hook_decision = not_applicable
hook_issue_type = none

dexterous_hand_decision = keep
dexterous_hand_issue_type = none
```

### 示例 4：hook 通道需要 PartNet-Mobility 补强

适用于当前 hook mask 来自 3D AffordanceNet 的 `pull/openable`，但可视化中看不出孔洞、内环或挂接结构。

```text
hook_decision = refine
hook_issue_type = needs_part_annotation
hook_notes = 当前 3D AffordanceNet 标签无法确认 hookable 结构，需要 PartNet-Mobility 的 handle hole/ring/gap 标注补强
```

## 当前优先审查对象

建议优先检查：

- `open_pull` 中的 `Bag`、`Bottle`、`TrashCan`；
- `Door + open_pull` 的 `suction` 是否过大；
- 所有 `hook` 非空样本是否真的有可挂接结构；
- `dexterous_hand` 是否只是复用了 gripper 区域，是否过窄或过泛；
- `press_push` 中 suction 和 dexterous_hand 的差异是否合理。

## 后续自动处理计划

等审查表填完后，可以编写 `tools/apply_manual_review.py`，自动执行：

- `keep_sample = no` 的样本从 metadata 中排除；
- `{executor}_decision = disable` 的通道清零；
- `{executor}_decision = refine` 的样本标记为待精修；
- 根据审查结果更新 `quality_flag`、`feasibility` 和 `negative_reason`。
