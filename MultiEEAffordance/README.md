# Multi-EE Affordance Dataset v0.1 构建说明

本目录用于构建一个物体级、多执行器、多标签 3D affordance 数据集原型。

当前任务定义为：

```text
P, q -> [M_gripper, M_suction, M_hook, M_dexterous_hand]
```

其中 `P` 是单个物体点云，`q` 是任务指令，输出 mask 的 shape 为 `[N, 4]`。

四个 mask 通道顺序固定为：

```text
0: gripper
1: suction
2: hook
3: dexterous_hand
```

## 当前目录结构

```text
MultiEEAffordance/
  taxonomy.yaml
  README.md
  docs/
    候选区域格式说明.md
    3D-AffordanceNet转换报告.md
    可视化检查报告.md
    人工审查表字段说明.md
    v0.1人工审查总结.md
    VLM多视角试验设计.md
    VLM试验运行手册.md
    Qwen3-VL+SAM2远程运行说明.md
    Qwen3-VL+SAM2候选标注流程图.md
    Qwen3-VL+SAM2候选标注与标注规范汇报稿.md
    本地Codex与远程服务器工作流.md
    项目进度日志.md
    选题思路跟进报告.md
    异构末端执行器标注规范.md
    VLM语义部件引导标注Pipeline.md
    VLM候选选择Pipeline_v2设计与汇报.md
  manifests/
    sample_manifest_template.jsonl
  raw/
    3d_affordancenet/
    partnet_mobility/
  processed/
    candidates/
    points/
    masks/
    visualizations/
    metadata/
      samples.jsonl
      samples_schema.md
      contrast_samples_v0_1.jsonl
  splits/
    train.txt
    val.txt
    test.txt
    contrast_test.txt
  tools/
    preprocess_pointcloud.py
    generate_weak_masks.py
    visualize_masks.py
    build_samples_jsonl.py
    check_dataset.py
    generate_3d_candidate_regions.py
    render_candidate_overlays_v2.py
    run_vlm_candidate_selection_v2.py
    filter_candidates_by_executor_rules.py
    build_v2_candidate_masks.py
    visualize_v2_candidates.py
```

## 推荐构建流程

### 1. 放入原始数据

将原始物体级数据放入：

```text
raw/3d_affordancenet/
raw/partnet_mobility/
```

第一阶段只处理物体级点云，不处理完整室内场景。

### 2. 预处理点云

将原始点云统一转换为 `processed/points/*.npy`：

```bash
python tools/preprocess_pointcloud.py \
  --input raw/3d_affordancenet/example.npy \
  --output processed/points/example.npy \
  --normalize unit_sphere
```

输出点云 shape 可以是 `[N, 3]` 或 `[N, 6]`。

- `[N, 3]` 表示 `x, y, z`
- `[N, 6]` 表示 `x, y, z, nx, ny, nz`

如果原始数据没有法向，先保留 `[N, 3]` 即可。

### 3. 准备候选区域

候选区域可以来自：

- 3D AffordanceNet 的已有 affordance mask；
- PartNet-Mobility 的部件标注；
- 人工整理的点索引；
- 简单几何规则产生的候选区域。

候选区域建议放入：

```text
processed/candidates/
```

具体格式见：

```text
docs/候选区域格式说明.md
```

### 4. 生成四通道弱标签 mask

```bash
python tools/generate_weak_masks.py \
  --points processed/points/example.npy \
  --candidate processed/candidates/example_pick_up.json \
  --task pick_up \
  --output processed/masks/example_pick_up.npy
```

输出 mask shape 必须是 `[N, 4]`。

### 5. 构建 samples.jsonl

先填写：

```text
manifests/sample_manifest_template.jsonl
```

再生成 metadata：

```bash
python tools/build_samples_jsonl.py \
  --dataset-root . \
  --manifest manifests/sample_manifest_template.jsonl \
  --output processed/metadata/samples.jsonl \
  --write-splits
```

### 6. 可视化检查

```bash
python tools/visualize_masks.py \
  --points processed/points/example.npy \
  --masks processed/masks/example_pick_up.npy \
  --channel all \
  --backend matplotlib \
  --output processed/visualizations/example_pick_up.png
```

也可以只看单个通道：

```bash
python tools/visualize_masks.py \
  --points processed/points/example.npy \
  --masks processed/masks/example_pick_up.npy \
  --channel hook
```

### 7. 检查数据集一致性

```bash
python tools/check_dataset.py --dataset-root .
```

该脚本会检查：

- metadata 字段是否完整；
- 点云文件是否存在；
- mask 文件是否存在；
- 点云 shape 是否为 `[N, 3]` 或 `[N, 6]`；
- mask shape 是否为 `[N, 4]`；
- 点数 `N` 是否一致；
- 不可行执行器是否填写了 `negative_reason`。

## v0.1 阶段原则

- 不训练模型；
- 不构建场景级 pipeline；
- 不用大模型直接生成逐点 mask；
- 优先把数据格式、弱标签规则、可视化检查和人工精修闭环跑通；
- 先做 30 到 50 组强对比样本，再扩展类别和规模。

## 后续真实数据接入建议

拿到真实数据后，推荐每个物体按下面顺序处理：

1. 预处理点云，得到 `processed/points/{object_id}.npy`。
2. 整理候选区域，得到 `processed/candidates/{sample_id}.json` 或 `.npz`。
3. 生成四通道 mask，得到 `processed/masks/{sample_id}.npy`。
4. 在 manifest 中登记该 object-task 样本。
5. 运行 `build_samples_jsonl.py` 生成 metadata 和 split。
6. 运行 `visualize_masks.py` 做人工检查。
7. 运行 `check_dataset.py` 做一致性检查。

## 3D AffordanceNet 接入

如果已经下载 `rotate.zip`、`full-shape.zip`、`partial.zip` 到：

```text
raw/3d_affordancenet/
```

可以先从 full-shape validation split 构建一个平衡弱标签原型：

```bash
python tools/convert_3d_affordancenet.py \
  --dataset-root . \
  --source-split val \
  --target-split val \
  --tasks all \
  --max-per-category 2 \
  --points-dir processed/points/3d_affordancenet_full_shape_val_balanced_v3 \
  --candidate-dir processed/candidates/3d_affordancenet_full_shape_val_balanced_v3 \
  --mask-dir processed/masks/3d_affordancenet_full_shape_val_balanced_v3 \
  --manifest manifests/3d_affordancenet_full_shape_val_balanced_v3_manifest.jsonl \
  --summary processed/metadata/3d_affordancenet_full_shape_val_balanced_v3_summary.json \
  --skip-all-negative \
  --overwrite
```

然后生成正式 metadata：

```bash
python tools/build_samples_jsonl.py \
  --dataset-root . \
  --manifest manifests/3d_affordancenet_full_shape_val_balanced_v3_manifest.jsonl \
  --output processed/metadata/samples.jsonl \
  --write-splits \
  --strict-files
```

本轮转换记录见：

```text
docs/3D-AffordanceNet转换报告.md
```

## HTML 可视化检查

如果当前环境暂时无法安装 `open3d` 或 `matplotlib`，可以使用无外部依赖的 HTML 可视化导出：

```bash
python tools/export_mask_html.py \
  --dataset-root . \
  --samples processed/metadata/samples.jsonl \
  --output-dir processed/visualizations/html_v3 \
  --limit 100 \
  --max-points 2048 \
  --write-index
```

打开索引页：

```text
processed/visualizations/html_v3/index.html
```

本轮可视化环境记录见：

```text
docs/可视化检查报告.md
```

## 网页人工审查

推荐使用本地审查网页直接填写审查结果，并自动保存到 CSV：

```powershell
cd D:\VSCode\Multi-EE-3DAG
conda activate multieeaffordance

python MultiEEAffordance\tools\serve_review_app.py `
  --dataset-root MultiEEAffordance `
  --host 127.0.0.1 `
  --port 8765 `
  --max-points 4096
```

打开：

```text
http://127.0.0.1:8765/
```

填写结果会保存到：

```text
processed/metadata/manual_review_v0_1.csv
```
