# 10条 Pilot 人工审查工作清单

更新时间：2026-05-20

## 1. 目标

当前阶段从已完成的 `vlm_pilot_005` 扩展到 10 条 pilot 样本。目标不是形成最终大规模数据集，而是验证：

- v2 候选生成是否能覆盖不同物体、任务和执行器；
- VLM 投票和规则过滤给出的 top-k 候选是否足够进入人工审查；
- 网页审查系统的“候选勾选组合 + 点级精修”是否适合小批量标注；
- 哪些执行器/任务组合仍然容易失败，后续反向改进规则和候选生成器。

## 2. 10条 Pilot 样本

| pilot_id | object_category | task | executor | 审查目的 |
| --- | --- | --- | --- | --- |
| `vlm_pilot_001` | Door | open_pull | suction | 检查 suction 在门板/平整面上的过标收缩，排除把手和边缘 |
| `vlm_pilot_002` | Dishwasher | open_pull | hook | 检查 hook 对把手内孔/可挂接结构的判断 |
| `vlm_pilot_003` | Earphone | pick_up | gripper | 检查 gripper 在耳机上的错区修正，避免不稳定连接处 |
| `vlm_pilot_004` | Earphone | pick_up | hook | 检查 hook 对耳机可挂接结构是否会过度泛化 |
| `vlm_pilot_005` | Bag | lift_carry | hook | 已完成首条审查；用于 bag handle / handle loop 的 hook 补标基准 |
| `vlm_pilot_006` | Bottle | lift_carry | dexterous_hand | 检查 dexterous hand 对瓶身可包覆区域的欠标修正 |
| `vlm_pilot_007` | Mug | pick_up | suction | 检查 suction 是否能保守选择可吸附区域，排除杯把手/高曲率边缘 |
| `vlm_pilot_008` | Keyboard | press_push | dexterous_hand | 检查 press_push 中按键/指尖按压区域的欠标修正 |
| `vlm_pilot_009` | Faucet | open_pull | dexterous_hand | 检查水龙头把手/旋钮类精细操作区域，避免泛化到普通管体 |
| `vlm_pilot_010` | Scissors | pick_up | hook | 检查 hook 对剪刀指环/孔洞边界可挂接区域的漏标补充 |

## 3. 远程服务器运行流程

### 3.1 拉取本地代码更新

在远程服务器仓库根目录执行：

```bash
git pull
```

### 3.2 生成 10 条 pilot 的 v2 候选

推荐先统一跑 10 条，保证 `processed/metadata/v2_candidate_samples_v0_1.jsonl` 包含完整 10 条待审查样本。

```bash
CUDA_VISIBLE_DEVICES=1,2 python MultiEEAffordance/tools/run_v2_pipeline.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml \
  --limit 10 \
  --stages generate,render,select,filter,build \
  --min-selected-votes 8 \
  --allow-empty \
  --overwrite
```

说明：

- `--limit 10` 表示读取 `vlm_pilot_samples_v0_1.csv` 前 10 条；
- `--allow-empty` 是必要的，因为某些执行器/任务组合可能没有被规则过滤接受的候选，但仍需要进入人工审查；
- `--overwrite` 会重建 v2 candidate 样本表，不会删除已经保存的 manual refined mask。

### 3.3 启动人工审查网页

```bash
python MultiEEAffordance/tools/serve_v2_annotation_app.py \
  --dataset-root MultiEEAffordance \
  --samples processed/metadata/v2_candidate_samples_v0_1.jsonl \
  --host 0.0.0.0 \
  --port 8770 \
  --top-k-candidates 8
```

浏览器打开：

```text
http://服务器IP:8770
```

如果页面显示旧逻辑，使用 `Ctrl+F5` 强制刷新。

## 4. 人工审查顺序建议

建议顺序：

```text
vlm_pilot_005 -> vlm_pilot_010 -> vlm_pilot_009 -> vlm_pilot_001 -> vlm_pilot_002 -> vlm_pilot_003 -> vlm_pilot_004 -> vlm_pilot_006 -> vlm_pilot_007 -> vlm_pilot_008
```

理由：

- 先从已熟悉的 `vlm_pilot_005` 验证页面是否正常；
- 再审查 `vlm_pilot_010` 和 `vlm_pilot_009`，因为它们是新增样本，能最快暴露扩展是否成功；
- 然后覆盖 suction、hook、gripper、dexterous_hand 的不同任务组合。

## 5. 每条样本的审查动作

1. 查看 `object_category / task / target_executor` 是否与当前样本一致；
2. 悬停候选卡片，快速查看每个候选落点；
3. 点击候选卡片，锁定某个候选进行细看；
4. 用 checkbox 勾选候选组合，点云会实时显示当前勾选集合；
5. 如果候选组合可作为初稿，进入“只删除”模式删除 false positive；
6. 必要时进入“只添加”模式补充少量漏标点；
7. 设置 `review_status`：
   - `checked`：当前结果可作为 checked 候选；
   - `refine_needed`：候选不足或仍需进一步生成；
   - `reject`：候选完全不可用；
8. 设置 `quality_after_review`：
   - `weak`：仍是弱标签，只能作为参考；
   - `checked`：已完成人工点级检查；
   - `verified`：非常确定，后续可作为高质量样本；
9. 保存 refined mask；
10. 重新加载样本确认保存结果保留。

## 6. 审查时重点记录的问题

审查时如果遇到以下问题，需要在 notes 中记录：

- 候选覆盖不足：目标区域没有被任何候选覆盖；
- 候选过宽：候选包含大量主体、普通边缘或背景点；
- VLM 选择错误：VLM 把语义不相关的候选选为目标；
- 规则过滤过严：合理候选被降为 uncertain 或 rejected；
- 规则过滤过宽：普通边缘/平面被接受为可操作区域；
- 点级编辑困难：误标点太多，单点删除效率低。

这些记录后续会用于改进候选生成器、VLM prompt 和 executor rule filter。
