# 候选区域文件格式说明

候选区域文件用于给 `generate_weak_masks.py` 提供弱标签来源。它不是最终数据标签，而是从已有 affordance mask、部件标注、几何规则或人工点索引中整理出的中间结果。

推荐将候选区域文件放在：

```text
processed/candidates/
```

## 支持格式

当前支持三类输入：

- `.json`
- `.npz`
- `.npy`

其中 `.json` 最适合人工整理和调试，`.npz` 更适合批量从已有数据集中导出。

## JSON 格式

### 直接指定执行器 mask

如果已经知道某个区域属于哪个执行器，可以直接按执行器写：

```json
{
  "executors": {
    "gripper": {
      "indices": [0, 1, 2, 3]
    },
    "suction": {
      "mask_path": "flat_panel_mask.npy"
    },
    "hook": {
      "indices_path": "handle_hole_indices.npy"
    },
    "dexterous_hand": {
      "mask": [1, 1, 0, 0, 1]
    }
  }
}
```

支持的写法包括：

| 字段 | 含义 |
| --- | --- |
| `indices` | 正样本点索引列表。 |
| `indices_path` | 保存点索引的一维 `.npy` 文件路径。 |
| `mask` | 一维二值 mask，长度为 `N`。 |
| `mask_path` | 保存一维二值 mask 的 `.npy` 文件路径。 |
| `path` | 自动判断是索引还是 mask 的文件路径。 |

JSON 内部路径默认相对于该 JSON 文件所在目录解析。

### 指定语义区域

如果只有部件名或候选区域名，可以写成 `regions`。脚本会根据任务和区域名做保守映射：

```json
{
  "regions": [
    {
      "name": "handle_outer",
      "indices_path": "handle_outer.npy",
      "tasks": ["pick_up", "open_pull"],
      "executors": ["gripper", "dexterous_hand"]
    },
    {
      "name": "inner_hole",
      "indices": [10, 11, 12],
      "tasks": ["pick_up", "lift_carry", "open_pull"],
      "executors": ["hook"]
    },
    {
      "name": "flat_panel",
      "mask_path": "drawer_front_mask.npy",
      "tasks": ["open_pull"]
    }
  ]
}
```

如果 `executors` 缺失，脚本会根据 `name` 和 `task` 推断通道。例如：

- `handle_outer` 在 `pick_up` 下通常映射到 `gripper` 和 `dexterous_hand`；
- `inner_hole` 通常映射到 `hook`；
- `flat_panel` 在 `pick_up`、`lift_carry` 或 `press_push` 下通常映射到 `suction`；
- `button` 或 `switch` 在 `press_push` 下通常映射到 `dexterous_hand`。

这只是弱标签规则，不等价于最终人工确认标签。

## NPZ 格式

`.npz` 中每个 key 可以是执行器名，也可以是语义区域名：

```python
np.savez(
    "example_pick_up.npz",
    gripper=gripper_mask,
    flat_panel=flat_panel_mask,
    inner_hole=inner_hole_indices,
)
```

如果 key 是执行器名：

- `gripper`
- `suction`
- `hook`
- `dexterous_hand`

则直接写入对应通道。

如果 key 是语义区域名，例如 `handle`、`flat_panel`、`button`、`ring`、`hole`，脚本会根据任务做弱映射。

## NPY 格式

`.npy` 适合只输入一个执行器通道或完整四通道 mask。

### 单通道输入

```bash
python tools/generate_weak_masks.py \
  --points processed/points/example.npy \
  --candidate processed/candidates/example_gripper.npy \
  --candidate-executor gripper \
  --task pick_up \
  --output processed/masks/example_pick_up.npy
```

### 四通道输入

如果 `.npy` 已经是 `[N, 4]`，脚本会直接按四通道 mask 读取。

## 人工修正规则

可以额外传入 `--manual-overrides`：

```json
{
  "gripper": {
    "add_indices": [1, 2, 3],
    "remove_indices": [8, 9]
  },
  "suction": {
    "set_mask_path": "checked_suction_mask.npy"
  }
}
```

常用字段：

| 字段 | 含义 |
| --- | --- |
| `set` / `set_mask` / `set_indices` | 直接替换该执行器通道。 |
| `add` / `add_mask` / `add_indices` | 向该执行器通道增加正样本。 |
| `remove` / `remove_mask` / `remove_indices` | 从该执行器通道移除正样本。 |

## 标注建议

- `gripper` 不要标大平面中心，优先标把手外侧、细长柄部、可夹持边缘。
- `suction` 优先标平滑大面积低曲率区域，不标孔洞、边缘、细杆和把手。
- `hook` 优先标内孔、拉环、孔洞边界、提手和可挂接结构。
- `dexterous_hand` 只标当前任务下可稳定抓握、按压或精细操作的区域，不要把所有接触面都标成正样本。
