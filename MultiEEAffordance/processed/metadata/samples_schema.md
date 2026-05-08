# samples.jsonl 格式说明

`processed/metadata/samples.jsonl` 中每一行表示一个“物体-任务”样本。除非写成绝对路径，所有路径都默认相对于数据集根目录 `MultiEEAffordance/` 解析。

## 必填字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `object_id` | string | 物体的稳定 ID，可以来自原始数据集，也可以是人工整理后的本地 ID。 |
| `source_dataset` | string | 数据来源。取值为 `3d_affordancenet`、`partnet_mobility`、`shapenet`、`objaverse` 或 `manual`。 |
| `object_category` | string | 物体类别，例如 `mug`、`drawer`、`cabinet`、`bottle`、`pan`、`button_panel`。 |
| `task` | string | 任务类型。取值为 `pick_up`、`lift_carry`、`open_pull`、`press_push`。 |
| `task_instruction` | string | 输入给模型的自然语言任务指令。 |
| `point_cloud_path` | string | 点云文件路径，通常指向 `points.npy`，数组 shape 应为 `[N, 3]` 或 `[N, 6]`。 |
| `multi_channel_mask_path` | string | 多通道 mask 文件路径，数组 shape 必须为 `[N, 4]`。 |
| `executor_order` | array[string] | 执行器通道顺序，必须是 `["gripper", "suction", "hook", "dexterous_hand"]`。 |
| `feasibility` | object | 每类执行器在当前物体和任务下是否可行。 |
| `label_source` | object | 每类执行器 mask 的标签来源。 |
| `negative_reason` | object | 不可行执行器的负样本原因。可行执行器可填 `null` 或空字符串。 |
| `quality_flag` | string | 标签质量标记。取值为 `weak`、`checked` 或 `verified`。 |
| `split` | string | 数据划分。取值为 `train`、`val`、`test` 或 `contrast_test`。 |

## 可选字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `sample_id` | string | 推荐使用的样本唯一 ID，例如 `3danet_mug_0001_pick_up`。 |
| `candidate_region_path` | string | 生成弱标签时使用的候选区域文件路径。 |
| `part_annotation_path` | string | 原始部件标注文件路径。 |
| `notes` | string | 人工整理备注，例如弱标签来源、需要复查的问题或特殊几何结构。 |

## 执行器字段格式

`feasibility`、`label_source` 和 `negative_reason` 的 key 必须严格等于四类执行器名称，并且顺序语义与 `executor_order` 一致。

```json
{
  "gripper": true,
  "suction": false,
  "hook": false,
  "dexterous_hand": true
}
```

`label_source` 的允许取值如下：

| 取值 | 含义 |
| --- | --- |
| `existing_affordance_mask` | 来自已有 affordance mask，例如 3D AffordanceNet 的原始标签。 |
| `part_annotation` | 来自部件标注，例如 PartNet-Mobility 的 handle、button、door panel 等部件。 |
| `geometry_rule` | 来自几何规则，例如平滑大平面、低曲率区域、孔洞边界等。 |
| `manual_refinement` | 来自人工检查或人工精修。 |
| `mixed` | 多种来源混合生成。 |
| `unavailable` | 当前执行器不可行或没有可用标签来源。 |

## JSONL 示例

```json
{"sample_id":"3danet_mug_0001_pick_up","object_id":"3danet_mug_0001","source_dataset":"3d_affordancenet","object_category":"mug","task":"pick_up","task_instruction":"Pick up the mug from a stable graspable region.","point_cloud_path":"processed/points/3danet_mug_0001.npy","multi_channel_mask_path":"processed/masks/3danet_mug_0001_pick_up.npy","executor_order":["gripper","suction","hook","dexterous_hand"],"feasibility":{"gripper":true,"suction":false,"hook":true,"dexterous_hand":true},"label_source":{"gripper":"existing_affordance_mask","suction":"unavailable","hook":"part_annotation","dexterous_hand":"mixed"},"negative_reason":{"gripper":null,"suction":"no_flat_suction_surface","hook":null,"dexterous_hand":null},"quality_flag":"weak","split":"train","notes":"Initial weak label from handle candidate regions."}
```

## 字段含义补充

### `feasibility`

表示某类执行器在当前物体和当前任务下是否存在合理可操作区域。

例如同一个 `drawer` 执行 `open_pull` 任务时：

- 有把手的抽屉：`gripper`、`hook`、`dexterous_hand` 通常可行；
- 平整无把手抽屉面板：`suction` 可能可行，`hook` 通常不可行；
- 如果没有明确抓握、挂接、吸附或按压区域，则对应执行器应标为 `false`。

### `negative_reason`

当 `feasibility[executor]` 为 `false` 时，必须填写明确原因。推荐使用 `taxonomy.yaml` 中的标准原因，例如：

- `no_graspable_region`
- `no_flat_suction_surface`
- `no_hookable_structure`
- `too_small_or_unstable`
- `high_curvature_or_edge`
- `ordinary_surface_without_operation_meaning`
- `missing_candidate_label`

### `quality_flag`

| 取值 | 含义 |
| --- | --- |
| `weak` | 由已有 mask、部件标注、几何规则或人工表自动生成，尚未严格人工确认。 |
| `checked` | 已经可视化检查过，并修正了明显错误。 |
| `verified` | 已经过人工精修或专家确认，可作为较可靠的评测样本。 |

## 校验规则

- `point_cloud_path` 必须存在，并且能加载为二维 NumPy 数组。
- 点云 shape 必须是 `[N, 3]` 或 `[N, 6]`。
- `[N, 3]` 表示只有 `x, y, z`。
- `[N, 6]` 表示 `x, y, z, nx, ny, nz`。
- `multi_channel_mask_path` 必须存在，并且能加载为二维 NumPy 数组。
- mask shape 必须是 `[N, 4]`。
- mask 的第 0 到第 3 个通道必须分别表示 `gripper`、`suction`、`hook`、`dexterous_hand`。
- `executor_order` 必须严格等于 `["gripper", "suction", "hook", "dexterous_hand"]`。
- 如果 `feasibility[executor]` 为 `false`，则 `negative_reason[executor]` 必须是非空字符串。
- `quality_flag` 和 `split` 必须使用 `taxonomy.yaml` 中定义的取值。

## v0.1 阶段约定

- 当前阶段只做物体级数据，不做完整室内场景级数据。
- 当前阶段不训练模型，只建立数据格式、弱标签生成流程和可视化检查流程。
- 弱标签生成应基于已有 affordance mask、部件标注、几何规则和人工精修表。
- 大模型只能辅助整理部件名称、规则说明和元数据文本，不能作为最终逐点 mask 生成器。
- 灵巧手标签不能过泛，不能把所有可接触表面都标成正样本；它应表示当前任务下适合类人多指手稳定抓握、按压或精细操作的区域。
