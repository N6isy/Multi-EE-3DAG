# Multi-EE Affordance Dataset v0.1 人工审查总结

生成时间：2026-05-07 20:13 +08:00

本文档总结 `manual_review_v0_1.csv` 中 61 条样本的人工审查结果，并说明当前弱标签规则最容易出错的位置。

## 1. 审查结果总览

- 输入样本数：61
- 保留样本数：45
- 剔除样本数：16
- 需要后续 refine/add_missing 的通道级记录数：66
- 一致性检查结果：45 条 checked 样本全部通过，points/mask/metadata/split 无错误

当前生成的 cleaned v0.1 是“样本级清洗版本”：已经剔除了明显不适合当前任务的 object-task 样本，并对 `disable` / `not_applicable` 通道做了 mask 清零。对于 `refine` / `add_missing` 通道，当前并没有直接伪造新的逐点区域，而是进入后续精修队列。

## 2. 产出文件

- `processed/metadata/samples_checked_v0_1.jsonl`
- `processed/masks_checked_v0_1/`
- `splits_checked_v0_1/`
- `processed/metadata/refine_queue_v0_1.csv`
- `processed/metadata/rejected_samples_v0_1.csv`
- `processed/metadata/manual_review_apply_summary_v0_1.json`

## 3. 样本级问题

| 问题类型 | 数量 | 说明 |
| --- | ---: | --- |
| none | 45 | 样本级 object-task 关系可保留 |
| task_mismatch | 13 | 任务与物体不匹配，例如桌子、沙发等不适合 `pick_up` |
| ambiguous_object | 3 | 点云形态或类别语义不清，人工难以稳定判断 |

最主要的样本级问题是 `task_mismatch`。这说明后续从 3D AffordanceNet 自动生成 object-task 样本时，需要加入更严格的类别-任务先验过滤，不能只根据已有 affordance 名称直接展开任务。

## 4. 通道级决策统计

| 执行器 | keep | refine | add_missing | disable | not_applicable | 空字段 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gripper | 30 | 9 | 5 | 1 | 1 | 15 |
| suction | 28 | 7 | 9 | 3 | 0 | 14 |
| hook | 25 | 2 | 15 | 3 | 0 | 16 |
| dexterous_hand | 27 | 13 | 5 | 2 | 0 | 14 |

`hook` 的 `add_missing` 最多，说明当前规则严重依赖原始 `pull/openable` 区域，但不能可靠识别“孔洞、拉环、可挂接边界”。`dexterous_hand` 的 `refine` 最多，说明灵巧手规则目前仍偏粗，容易把可接触区域和可稳定操作区域混在一起。

## 5. 规则错误排名

| 排名 | 错误类型 | 数量 | 主要涉及 |
| ---: | --- | ---: | --- |
| 1 | 漏标：弱规则没有生成应有正样本 | 38 | hook、suction、dexterous_hand、gripper |
| 2 | 待精修：当前弱标签需要人工或规则修正 | 11 | dexterous_hand、gripper、suction |
| 3 | 执行器不匹配：物理可行性规则不足 | 9 | suction、gripper、hook、dexterous_hand |
| 4 | 错区：弱规则标到错误部位 | 8 | gripper、dexterous_hand、suction、hook |
| 5 | 过标：弱规则生成区域过大 | 7 | suction、gripper、dexterous_hand |
| 6 | 缺少几何规则：需要法向/曲率/面积约束 | 2 | hook、gripper/dexterous_hand |

最高频错误是“漏标”。这说明当前 v0.1 弱标签规则太依赖已有 affordance mask，当原始数据没有显式标出 hookable / suctionable / dexterous 操作区域时，规则不会主动补充合理区域。

## 6. 分执行器主要问题

### gripper

- `missing_positive`：5
- `wrong_region`：4
- `under_label`：3
- `executor_mismatch`：2

主要问题是把 `grasp/hold` 映射到 gripper 时仍不够区分任务语义。例如 `press_push` 中是否允许通过门把手推门，需要和任务、受力方向一起判断。

### suction

- `missing_positive`：7
- `over_label`：5
- `needs_geometry_rule`：3
- `executor_mismatch`：3

主要问题是 suction 的面积、曲率和平整度规则不足。人工审查中已经暴露出：小吸盘可以吸附某些局部平面，但 v0.1 为了保持标签稳定，应优先标注“大于局部阈值、法向一致、低曲率、非边缘”的区域。

### hook

- `missing_positive`：15
- `needs_geometry_rule`：2
- `wrong_region`：1
- `executor_mismatch`：2

hook 是当前最需要补规则的执行器。它不能只看 `pull/openable`，还需要显式识别孔洞、环、把手内侧边界、提手开口等几何结构。

### dexterous_hand

- `under_label`：8
- `missing_positive`：6
- `wrong_region`：2
- `over_label`：1

灵巧手最大问题是边界定义：不能把所有可接触表面都标为正样本，但也不能漏掉可多指包覆、按压、旋转或稳定精细操作的区域。后续需要按任务分别收紧定义。

## 7. 对后续规则的修正建议

- object-task 过滤：增加类别黑名单/白名单，例如桌子、沙发、床等大件家具默认不进入 `pick_up`。
- suction：加入最小吸盘接触面积、局部平面性、法向一致性、边缘距离约束。
- hook：从 PartNet-Mobility 或后续几何规则中引入孔洞/环/把手内孔检测。
- dexterous_hand：将“可触碰”改为“可稳定包覆、按压、旋转或精细操作”，并按任务单独定义正样本。
- refine 队列优先处理 `hook add_missing`、`dexterous_hand under_label`、`suction over_label/missing_positive`。

## 8. 下一步

先从 `processed/metadata/refine_queue_v0_1.csv` 中挑选 5 到 10 条 `refine` / `add_missing` 样本，建立 VLM 多视角小试验队列。小试验只验证“多视角渲染、2D mask 投影回 3D、跨视角融合”的流程是否可行，不全量生成最终数据。
