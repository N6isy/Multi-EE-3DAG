# V3 语义目标-拒绝候选 Pipeline 设计与汇报

更新时间：2026-05-21 +08:00

> 当前状态：历史设计文档。本文记录的是旧的 `ground -> project -> grow` target/reject seed-growth 方案。当前推荐主线已经切换为 `candidate-source partseg` + `part-proposal-backend high_recall`，具体命令见 `v3自研高召回3D候选生成器说明.md` 和 `大规模人工审查与已审查数据集可视化.md`。

> 标注版本说明：当前轻量自动候选 + 人工点级审查版本将冻结为 `annotation/mvp-v0.1`，用于双人协作标注。后续候选生成器改进进入 `dev/high-recall-candidate-v0.2`，不要在同一标注批次中混用不同候选生成版本。

## 1. 当前问题判断

v2 / v2.2 的核心问题不是“候选数量不够多”，而是候选生成逻辑仍然偏几何优先。对于 `Scissors / pick_up / hook` 这类样本，剪刀刃因为细长、高曲率、边界明显，很容易被几何规则当成 hook 候选；但从执行器机制看，hook 需要插入、挂住、形成机械互锁并沿任务方向施力，剪刀刃和刀尖显然不应该作为正例。

当前数据也有天然限制：

| 限制 | 影响 |
| --- | --- |
| 点云通常较稀疏，很多样本约 2048 点 | 小部件可能只有十几个点，不能期待 VLM/SAM2 直接给出精确连续 mask |
| 当前输入主要是点云渲染图，不是真实 RGB 图像 | 开放词汇 grounding 模型容易把稀疏点云的整体轮廓当成目标部件 |
| 没有 mesh 和真实表面拓扑 | 自动方法只能在已有点上生成候选，不能凭空补出不存在的表面点 |
| 任务和执行器强相关 | “能接触”不等于“能被该执行器完成该任务” |

因此，v3 的目标不是直接自动生成 ground truth，而是生成更适合人工审查的高召回、机制一致、带 reject veto 的候选区域。

## 2. v3 总体设计

v3 改为“语义目标/拒绝先行”的流程：

```text
多视角点云渲染
  -> Qwen3-VL 语义规划：target parts + reject parts
  -> Qwen3-VL 对 target/reject 进行粗定位
  -> point-index map 回投到 3D target/reject votes
  -> 只从 target seed 生长候选，reject 区域硬排除
  -> 候选 overlay / review visualization
  -> 构建候选 [N,4] mask
  -> 网页人工勾选组合 + 点级增删精修
```

核心变化：

| v2 思路 | v3 思路 |
| --- | --- |
| 先用几何生成候选，再让 VLM 从候选里投票 | 先让 VLM 明确目标部件和必须拒绝部件，再投回 3D |
| 几何候选可能把细长边界误认为 hook | reject veto 会显式排除 blade / cutting edge / tip 等语义负例 |
| VLM 只能在已有 A/B/C 候选里选 | VLM 先产生 target/reject seed，候选由 seed 生长得到 |
| 规则过滤容易过宽或过窄 | 规则只做 target seed 生长边界和 reject 裁剪，不单独决定正例 |

## 3. 模块说明

| 阶段 | 脚本 | 输出 | 作用 |
| --- | --- | --- | --- |
| 渲染 | `render_vlm_friendly_views.py` | `processed/vlm_semantic_part/renders/.../view_manifest.json` | 生成多视角 VLM-friendly 图和 point-index map |
| 语义规划 | `run_v3_semantic_part_planner.py` | `processed/vlm_candidate_v3/semantic_plans/<pilot_id>/combined_semantic_plan.json` | 明确 target_positive_parts、reject_negative_parts |
| 2D 粗定位 | `run_v3_target_reject_grounding.py` | `processed/vlm_candidate_v3/target_reject_grounding/<pilot_id>/...json` | 对 target/reject 输出粗 box 和 positive points |
| 3D 回投 | `project_v3_grounding_to_3d.py` | `processed/vlm_candidate_v3/projected_3d/<pilot_id>_target_reject_votes.npz` | 将 2D target/reject 映射为 3D 点投票 |
| 候选生长 | `grow_v3_reject_aware_candidates.py` | `processed/vlm_candidate_v3/3d_candidates/<pilot_id>/candidate_manifest.json` | 从 target seed 局部扩张，禁止跨过 reject veto |
| 候选可视化 | `render_candidate_overlays_v2.py` / `visualize_v2_candidates.py` | `processed/vlm_candidate_v3/candidate_overlays` / `review_visualizations` | 给 VLM 和人工审查看候选位置 |
| mask 构建 | `build_v3_candidate_masks.py` | `processed/metadata/v3_candidate_samples_v0_1.jsonl` | 构建待审查的四通道候选 mask |
| 人工审查 | `serve_v2_annotation_app.py` | refined mask / review log | 人工勾选候选组合，并点级删除/添加 |

## 4. 关键机制：Target Seed + Reject Veto

v3 不再把“细长、高曲率、边界、极值区域”直接当成正例，而是分两层处理：

1. `target seed`：VLM 根据任务和执行器语义指出真正应该成为候选的功能部件，例如 scissors 的 handle loops / finger holes。
2. `reject veto`：VLM 同时指出必须排除的部件，例如 scissors 的 blade / cutting edge / blade tip。
3. 候选只能从 target seed 生长。
4. 生长过程中遇到 reject veto 会被裁剪，不能把剪刀刃、刀尖、普通长边界带进正例。

这能解决当前最明显的问题：几何上“像 hook 候选”的区域，不一定满足 hook 的插入、挂住、拉力约束。

## 5. 人工审查要看的内容

人工审查时不应该只看一个已经合成的 mask，而应看：

| 要看什么 | 目的 |
| --- | --- |
| `semantic_plan` 中 target/reject 是否合理 | 检查 VLM 对任务和执行器机制是否理解正确 |
| `target_reject_grounding` 中 box/points 是否落在正确部件 | 检查 2D 粗定位有没有偏到主体或错误部位 |
| `candidate_manifest` 中 `default_selected_candidates` | 查看自动推荐候选组合 |
| `review_visualizations/index.html` | 快速判断每个候选的位置和覆盖范围 |
| 网页审查系统中的候选勾选和点级编辑 | 最终确认候选组合，并删除错误点、补充遗漏点 |

人工最终面对的是“v3 推荐候选 + 可手动切换的候选组合 + 可点级增删的当前 mask”，不是固定的单个自动 mask。

## 6. 历史复现实验命令

如需复现旧 target/reject seed-growth 方案，可以使用下面命令。当前正式标注不推荐使用这条链路，因为它依赖 VLM 粗定位和 target seed，容易在稀疏点云、小部件、空标签样本上产生 `candidate_count=0`。

```bash
CUDA_VISIBLE_DEVICES=1,2 python MultiEEAffordance/tools/run_v3_pipeline.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml \
  --pilot-csv processed/metadata/v3_large_scale_review_queue_v0_1.csv \
  --samples processed/metadata/samples_v3_large_batch_v0_1.jsonl \
  --candidate-source grounding \
  --stages views,plan,ground,project,grow,render,coverage,build \
  --allow-empty \
  --overwrite
```

当前推荐主线如下：

```bash
CUDA_VISIBLE_DEVICES=1,2 python MultiEEAffordance/tools/run_v3_pipeline.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml \
  --pilot-csv processed/metadata/v3_large_scale_review_queue_v0_1.csv \
  --samples processed/metadata/samples_v3_large_batch_v0_1.jsonl \
  --candidate-source partseg \
  --part-proposal-backend high_recall \
  --stages views,plan,part_propose,render,part_select,part_filter,build \
  --proposal-max-candidates 64 \
  --max-candidates 12 \
  --part-top-k 5 \
  --allow-empty \
  --overwrite
```

启动网页人工审查：

```bash
python MultiEEAffordance/tools/serve_v2_annotation_app.py \
  --dataset-root MultiEEAffordance \
  --samples processed/metadata/v3_candidate_samples_v0_1.jsonl \
  --port 8765 \
  --top-k-candidates 8
```

## 7. 调参建议

| 参数 | 作用 | 建议 |
| --- | --- | --- |
| `--expand-hops` | target seed 生长范围 | 候选太稀疏时从 `1` 调到 `2` |
| `--reject-score-threshold` | reject veto 严格程度 | 错误区域混入时降低到 `0.05`，过度裁剪时升到 `0.15` |
| `--target-score-threshold` | target seed 召回程度 | 漏掉目标时降低，误吸主体时提高 |
| `--box-shrink-ratio` | 2D box 回投时收缩比例 | VLM box 过大时可用 `0.05` 到 `0.15` |
| `--max-target-box-area-fraction` | target 粗定位框过大时的保护阈值 | 默认 `0.35`；若 VLM 常框住整物体，可降到 `0.25` |
| `--max-candidates` | overlay 展示候选数量 | 人工审查时可提高到 `18`，VLM 自动阶段保持 `12` 左右 |

补充说明：v3 对 target 大框做了额外保护。如果 Qwen3-VL 对 target 输出的 box 超过图像面积阈值，同时又给出了 positive points，脚本会丢弃这个过大的 target box，仅保留点提示参与回投。这样可以降低“整把剪刀/整个袋子被框进 target”的风险。reject box 不做同样丢弃，因为 blade、door panel 等负例有时确实需要较长或较大的 veto 区域。

## 8. 当前结论

v3 的核心定位是“最终可用的候选生成与人工审查加速框架”，不是完全自动标注器。它利用 VLM 最擅长的语义和机制判断，同时把最终 3D mask 限制在真实点云和人工审查闭环内。

当前落地策略是先把这套框架作为稳定标注工具使用：自动阶段只负责生成和排序候选，人工阶段负责最终 mask。双人协作时使用 `reviewer_a/reviewer_b` 独立样本文件、独立 review log、独立 refined mask 目录；候选生成器本身的下一轮改进不进入 `annotation/mvp-v0.1` 标注分支。

下一步应优先用 `vlm_pilot_010: Scissors / pick_up / hook` 检查：

1. `semantic_plan` 是否明确 target 为 finger holes / handle loops。
2. reject 是否明确包含 blade / cutting edge / blade tip。
3. v3 候选是否避开剪刀刃，只围绕指环或内孔边界生长。
4. 网页中是否能通过勾选候选组合快速得到可精修 mask。
