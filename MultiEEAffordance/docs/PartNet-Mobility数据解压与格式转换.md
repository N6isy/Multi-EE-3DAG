# PartNet-Mobility 数据解压与格式转换

更新时间：2026-06-01

本文档说明如何在数据存储服务器 `10.24.1.11` 上解压并转换 PartNet-Mobility。转换目标不是直接生成最终 affordance 标签，而是为五任务人工审查和后续训练准备：

```text
标准化物体点云 points.npy
URDF link 级部件候选 parts.npz
可读候选说明 candidate_manifest.json
物体级索引 objects_manifest.jsonl
转换统计 summary.json
```

## 1. 数据规模口径

当前数据集规划为：

```text
3D AffordanceNet：8000+ 目标样本行
PartNet-Mobility：4000+ 目标样本行
总计：约 1.2w 五任务人工审查样本行
```

注意：`4000+` 指后续按任务和执行器展开后的目标样本行，不是 PartNet-Mobility raw object 数量。PartNet-Mobility 原始数据应先按 object 统计，再按任务组合生成审查样本。两种数字不能混用。

转换脚本会输出：

```text
discovered_raw_objects
selected_raw_objects
converted_or_reused_objects
total_part_candidates
total_movable_parts
```

先以 summary 为准确认服务器上 zip 的真实内容，再规划最终五任务下采样数量。

## 2. 当前服务器分工

代码修改仍然先在本地完成，再推送到 GitHub。

数据生产和人工审查统一在数据存储服务器执行：

```text
服务器：10.24.1.11
数据根目录：/home/lzq/data/MultiEEAffordance
```

项目服务器已有原始压缩包：

```text
/home/lzq/Multi-EE-3DAG/MultiEEAffordance/raw/partnet_mobility/partnet-mobility-v0.zip
```

如果 `10.24.1.11` 尚未保存该文件，先从项目服务器复制到数据盘。下面命令在能够读取原始 zip 的机器上执行：

```bash
mkdir -p /home/lzq/data/MultiEEAffordance/raw/partnet_mobility

scp \
  /home/lzq/Multi-EE-3DAG/MultiEEAffordance/raw/partnet_mobility/partnet-mobility-v0.zip \
  lzq@10.24.1.11:/home/lzq/data/MultiEEAffordance/raw/partnet_mobility/
```

如果两个路径本来就在同一台服务器或共享文件系统中，可以直接省略复制步骤。

## 3. 转换脚本做什么

脚本：

```text
MultiEEAffordance/tools/convert_partnet_mobility.py
```

处理流程：

```text
partnet-mobility-v0.zip
  -> 安全解压
  -> 扫描每个数字 object_id 目录
  -> 读取 meta.json 获取类别
  -> 读取 mobility.urdf 获取 link、joint、visual mesh
  -> 按 link 进行 surface sampling
  -> 为小部件保留最小采样预算
  -> 保存 [N,3] 点云
  -> 保存 [K,N] link-level 部件候选
```

其中：

- `N` 是每个物体的点数，默认 `2048`。
- `K` 是该物体保留下来的 URDF link 数量。
- `parts.npz` 是 proposal，不是 `[N,4]` 最终标签。
- 脚本不会猜测 gripper、suction、hook、dexterous_hand 的最终正例。
- 最终 `[N,4]` mask 仍然必须经过五任务规则检查和人工审查。

### 3.1 默认只转换补充类别

PartNet-Mobility 在本项目中的用途是补充 3D AffordanceNet 已有类别，不是重复转换整个数据集。脚本默认只转换以下 21 类：

```text
Box
Bucket
Cabinet
Camera
CoffeeMachine
Dispenser
Kettle
Lighter
Mouse
Oven
Phone
Pliers
Remote
Safe
Stapler
Suitcase
Switch
Toaster
Toilet
WashingMachine
Window
```

正常运行时不需要额外传入 `--categories`。如需临时覆盖默认类别，可以显式传入逗号分隔列表：

```bash
--categories Box,Cabinet,Switch
```

只有在诊断原始数据内容时才使用全类别转换：

```bash
--categories all
```

解压阶段仍然会解压原始 zip。类别过滤发生在转换阶段，因此正式输出目录只会生成上述补充类别的标准点云和候选文件。

## 4. 第一次运行：先解压

登录数据存储服务器：

```bash
ssh lzq@10.24.1.11
cd /home/lzq/Multi-EE-3DAG
```

只执行解压：

```bash
python MultiEEAffordance/tools/convert_partnet_mobility.py \
  --dataset-root /home/lzq/data/MultiEEAffordance \
  --zip raw/partnet_mobility/partnet-mobility-v0.zip \
  --extract-dir raw/partnet_mobility/partnet-mobility-v0 \
  --stages extract
```

解压完成后检查：

```bash
find /home/lzq/data/MultiEEAffordance/raw/partnet_mobility/partnet-mobility-v0 \
  -name mobility.urdf | head
```

如果没有任何输出，不要继续全量转换。先检查 zip 内部目录层级。

## 5. 小批 smoke test：只转换 5 个物体

```bash
python MultiEEAffordance/tools/convert_partnet_mobility.py \
  --dataset-root /home/lzq/data/MultiEEAffordance \
  --zip raw/partnet_mobility/partnet-mobility-v0.zip \
  --extract-dir raw/partnet_mobility/partnet-mobility-v0 \
  --stages convert \
  --max-objects 5 \
  --sample-size 2048 \
  --min-points-per-part 12 \
  --points-dir processed/points/partnet_mobility_v0_supplement_21cat_smoke \
  --candidate-dir processed/candidates/partnet_mobility_v0_supplement_21cat_smoke \
  --manifest manifests/partnet_mobility_v0_supplement_21cat_smoke_objects_manifest.jsonl \
  --summary processed/metadata/partnet_mobility_v0_supplement_21cat_smoke_conversion_summary.json \
  --overwrite
```

检查输出：

```bash
cat /home/lzq/data/MultiEEAffordance/processed/metadata/partnet_mobility_v0_supplement_21cat_smoke_conversion_summary.json

find /home/lzq/data/MultiEEAffordance/processed/points/partnet_mobility_v0_supplement_21cat_smoke \
  -name '*.npy' | head

find /home/lzq/data/MultiEEAffordance/processed/candidates/partnet_mobility_v0_supplement_21cat_smoke \
  -name candidate_manifest.json | head
```

重点检查：

1. `converted_or_reused_objects` 是否等于 `5`。
2. `skipped_objects` 是否为 `0` 或数量很少。
3. `points.npy` 是否为 `[2048,3]`。
4. `parts.npz` 中 `candidate_masks` 是否为 `[K,2048]`。
5. `candidate_manifest.json` 中是否保存 link、joint、`is_movable` 和 `need_review=true`。

## 6. 正式全量转换

小批检查通过后，运行：

```bash
python MultiEEAffordance/tools/convert_partnet_mobility.py \
  --dataset-root /home/lzq/data/MultiEEAffordance \
  --zip raw/partnet_mobility/partnet-mobility-v0.zip \
  --extract-dir raw/partnet_mobility/partnet-mobility-v0 \
  --stages convert \
  --sample-size 2048 \
  --min-points-per-part 12 \
  --points-dir processed/points/partnet_mobility_v0_supplement_21cat \
  --candidate-dir processed/candidates/partnet_mobility_v0_supplement_21cat \
  --manifest manifests/partnet_mobility_v0_supplement_21cat_objects_manifest.jsonl \
  --summary processed/metadata/partnet_mobility_v0_supplement_21cat_conversion_summary.json
```

正式输出：

```text
/home/lzq/data/MultiEEAffordance/processed/points/partnet_mobility_v0_supplement_21cat/
/home/lzq/data/MultiEEAffordance/processed/candidates/partnet_mobility_v0_supplement_21cat/
/home/lzq/data/MultiEEAffordance/manifests/partnet_mobility_v0_supplement_21cat_objects_manifest.jsonl
/home/lzq/data/MultiEEAffordance/processed/metadata/partnet_mobility_v0_supplement_21cat_conversion_summary.json
```

脚本默认支持断点复用。已经存在的 object 输出不会重复生成。确实需要重算时再增加：

```bash
--overwrite
```

## 7. 转换后的下一步

当前转换脚本只完成 raw asset 到标准点云和部件 proposal 的转换。不要把 `parts.npz` 直接改名为最终 mask。PartNet-Mobility 的 link segmentation 是高价值部件 proposal，但它不等价于任务相关、执行器相关的 affordance ground truth。

PartNet-Mobility 不需要再经过 3D AffordanceNet 的旧任务 pipeline。它直接进入五任务人工审查准备流程：

```text
PartNet-Mobility objects_manifest.jsonl
  -> build_partnet_5task_review_samples.py
  -> 生成合理类别-任务组合和四执行器审查行
  -> 生成初始全零 [N,4] mask
  -> package_annotation_batches_from_samples.py
  -> 生成 reviewer_a/reviewer_b 独立批次
  -> 在 10.24.1.11 启动网页审查
  -> 保存人工 refined [N,4] mask
  -> build_reviewed_dataset_release.py
```

## 8. 生成五任务审查样本

运行：

```bash
python MultiEEAffordance/tools/build_partnet_5task_review_samples.py \
  --dataset-root /home/lzq/data/MultiEEAffordance \
  --objects-manifest manifests/partnet_mobility_v0_supplement_21cat_objects_manifest.jsonl \
  --output-samples processed/metadata/partnet_mobility_v0_supplement_21cat_5tasks_review_samples.jsonl \
  --initial-mask-dir processed/masks/partnet_mobility_v0_supplement_21cat_initial_empty \
  --summary-json processed/metadata/partnet_mobility_v0_supplement_21cat_5tasks_review_summary.json \
  --task-policy plausible \
  --overwrite
```

输入：

```text
manifests/partnet_mobility_v0_supplement_21cat_objects_manifest.jsonl
```

输出：

```text
processed/metadata/partnet_mobility_v0_supplement_21cat_5tasks_review_samples.jsonl
processed/metadata/partnet_mobility_v0_supplement_21cat_5tasks_review_summary.json
processed/masks/partnet_mobility_v0_supplement_21cat_initial_empty/*.npy
```

说明：

- `--task-policy plausible` 使用类别级高召回先验，生成合理的 `lift/open/pull/press/push` 组合。
- 每个保留的物体-任务组合都会生成四类执行器审查行。
- 即使某个执行器预计没有有效区域，也保留对应行，由人工确认空标签。
- 初始 mask 是全零 `[N,4]`。网页中显示的 URDF link 候选只是辅助选择区域。
- PartNet-Mobility 已经直接使用五任务，不要再运行 `expand_legacy_tasks_to_5tasks.py`。

如需在正式运行前只检查 5 个物体：

```bash
python MultiEEAffordance/tools/build_partnet_5task_review_samples.py \
  --dataset-root /home/lzq/data/MultiEEAffordance \
  --objects-manifest manifests/partnet_mobility_v0_supplement_21cat_objects_manifest.jsonl \
  --output-samples processed/metadata/partnet_mobility_v0_supplement_21cat_5tasks_smoke.jsonl \
  --initial-mask-dir processed/masks/partnet_mobility_v0_supplement_21cat_initial_empty_smoke \
  --summary-json processed/metadata/partnet_mobility_v0_supplement_21cat_5tasks_smoke_summary.json \
  --task-policy plausible \
  --max-objects 5 \
  --overwrite
```

## 9. 生成双人审查批次

```bash
python MultiEEAffordance/tools/package_annotation_batches_from_samples.py \
  --dataset-root /home/lzq/data/MultiEEAffordance \
  --input processed/metadata/partnet_mobility_v0_supplement_21cat_5tasks_review_samples.jsonl \
  --batch-dir processed/annotation_batches/partnet_mobility_v0_supplement_21cat_5tasks_v0_1 \
  --reviewers reviewer_a,reviewer_b \
  --calibration-objects 5 \
  --archive-format tar.gz \
  --overwrite
```

输出：

```text
processed/annotation_batches/partnet_mobility_v0_supplement_21cat_5tasks_v0_1/
  reviewer_a_samples.jsonl
  reviewer_b_samples.jsonl
  reviewer_a_annotation_package.tar.gz
  reviewer_b_annotation_package.tar.gz
  batch_manifest.json
```

分包器只复制网页审查需要的点云、初始 mask、candidate manifest 和 candidate npz。原始 OBJ mesh 只作为溯源信息，不会加入审查压缩包。

## 10. 启动网页审查

`reviewer_a`：

```bash
python MultiEEAffordance/tools/serve_v2_annotation_app.py \
  --dataset-root /home/lzq/data/MultiEEAffordance \
  --samples processed/annotation_batches/partnet_mobility_v0_supplement_21cat_5tasks_v0_1/reviewer_a_samples.jsonl \
  --review-jsonl processed/annotation_batches/partnet_mobility_v0_supplement_21cat_5tasks_v0_1/reviewer_a_review_records.jsonl \
  --output-mask-root processed/annotation_batches/partnet_mobility_v0_supplement_21cat_5tasks_v0_1/manual_refined_masks_reviewer_a \
  --output-samples processed/annotation_batches/partnet_mobility_v0_supplement_21cat_5tasks_v0_1/reviewer_a_refined_samples.jsonl \
  --host 0.0.0.0 \
  --port 8765 \
  --top-k-candidates 8
```

`reviewer_b` 使用相同命令，将路径中的 `reviewer_a` 替换为 `reviewer_b`，端口改为 `8766`。

网页已经兼容没有 VLM 投票的 URDF link proposal。即使不传 `--candidate-min-selected-votes 0`，仍会展示前 `top-k` 个 PartNet 候选，并优先展示 movable link。

## 11. 生成人工审查 release

两位审查者完成一个批次后运行：

```bash
python MultiEEAffordance/tools/build_reviewed_dataset_release.py \
  --dataset-root /home/lzq/data/MultiEEAffordance \
  --reviewed-samples processed/annotation_batches/partnet_mobility_v0_supplement_21cat_5tasks_v0_1/reviewer_a_refined_samples.jsonl,processed/annotation_batches/partnet_mobility_v0_supplement_21cat_5tasks_v0_1/reviewer_b_refined_samples.jsonl \
  --output-samples processed/metadata/reviewed_partnet_mobility_v0_supplement_21cat_5tasks_v0_1.jsonl \
  --summary-json processed/metadata/reviewed_partnet_mobility_v0_supplement_21cat_5tasks_summary_v0_1.json \
  --output-split-dir splits_reviewed_partnet_mobility_v0_supplement_21cat_5tasks_v0_1 \
  --include-tasks lift,open,pull,press,push \
  --overwrite
```

保存网页标注时，系统会根据人工最终正例同步更新当前执行器的：

```text
feasibility
label_source
negative_reason
```

人工加点后写入 `manual_refinement`；人工确认空标签后写入 `confirmed_empty_by_human_review`。
