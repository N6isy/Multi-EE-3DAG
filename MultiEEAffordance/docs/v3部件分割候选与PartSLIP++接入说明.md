# v3 部件分割候选与 PartSLIP++ 接入说明

更新时间：2026-05-26

> 当前状态：本文档记录 PartSLIP++ 外部 adapter 的历史接入方式。当前推荐主线已经切换为 `v3自研高召回3D候选生成器说明.md` 中的 `high_recall` backend，不再要求配置 PartSLIP++。

## 1. 当前主线

v3 正式链路不再包含自然化点云表面渲染。本文记录的 PartSLIP++ 路径只作为可选外部 adapter 保留。当前默认候选区域划分已经改为：

```text
原始点云 / 弱标签 / 自研 high_recall 3D 候选生成器
  -> 3D part candidates [K, N]
  -> VLM 看候选 overlay，只选择候选 ID
  -> 执行器规则过滤
  -> 人工审查
  -> [N,4] mask
```

关键原则：

- `N` 始终是原始点云点数。
- PartSLIP++ 输出只作为候选区域来源，不直接成为 ground truth。
- VLM 不输出点坐标或 box，只判断候选区域是否适合当前 `task + executor`。
- 人工审查仍是最终确认环节。

## 2. 已实现的接入方式

正式候选入口：

```bash
python MultiEEAffordance/tools/propose_v3_part_candidates.py \
  --dataset-root MultiEEAffordance \
  --pilot-csv processed/metadata/v3_test_review_queue_v0_1.csv \
  --samples processed/metadata/samples_v3_large_batch_v0_1.jsonl \
  --backend partslippp \
  --partslippp-root external/partslippp/outputs \
  --overwrite
```

当前支持两个 backend：

| backend | 状态 | 用途 |
| --- | --- | --- |
| `geometry` | 可运行 fallback | 仅用原始点云几何、弱标签和通用候选 family 生成候选 |
| `partslippp` | 已接入 adapter | 读取外部 PartSLIP++ 分割结果，转换为 v3 `[K,N]` 候选 |

PartSLIP++ adapter 支持 `.npz` 和 `.json` 两种输入格式。

## 3. 服务器路径与输出放置约定

服务器端 PartSLIP++ 固定放置路径：

```text
/home/lzq/Multi-EE-3DAG/MultiEEAffordance/external/partslippp
```

建议把该目录理解成 PartSLIP++ 的独立工作区：

```text
/home/lzq/Multi-EE-3DAG/MultiEEAffordance/external/partslippp/
  data/        # PartSLIP++ 官方数据或转换后的输入，可选
  model/       # PartSLIP++ checkpoint
  outputs/     # 给 Multi-EE v3 pipeline 读取的标准化输出
  GLIP/        # 官方 submodule
  partition/   # cut-pursuit 等官方代码
  src/         # 官方代码
```

Multi-EE v3 默认读取 PartSLIP++ 预测结果的根目录是：

```text
external/partslippp/outputs/
```

推荐每条审查行按 `pilot_id` 放：

```text
external/partslippp/outputs/<pilot_id>/segments.npz
```

例如：

```text
external/partslippp/outputs/v3_review_000061/segments.npz
```

也可以显式指定路径模板：

```bash
--partslippp-path "external/partslippp/outputs/{pilot_id}/segments.npz"
```

可用模板变量：

```text
{pilot_id}
{sample_id}
{task}
{executor}
{object_category}
```

## 4. Adapter 支持的文件格式

### 4.1 推荐 npz：语义/实例标签

```python
np.savez_compressed(
    "segments.npz",
    semantic_seg=semantic_seg,      # shape [N], -1 表示非部件或背景
    instance_seg=instance_seg,      # 可选，shape [N], -1 表示无实例
    label_names={1: "handle", 2: "blade", 3: "button"}
)
```

如果没有实例标签，只提供 `semantic_seg` 也可以。adapter 会把每个非负语义标签转换为一个候选；如果同时有 `instance_seg`，会按 `(semantic, instance)` 拆成多个候选。

### 4.2 推荐 npz：直接候选 mask

```python
np.savez_compressed(
    "segments.npz",
    candidate_masks=candidate_masks,        # shape [K,N] 或 [N,K]
    candidate_names=np.array(["handle", "button", "panel"]),
    scores=np.array([0.92, 0.81, 0.74])     # 可选
)
```

### 4.3 子采样输出

如果 PartSLIP++ 只对部分点输出标签或 mask，必须额外保存原始点索引：

```python
np.savez_compressed(
    "segments.npz",
    semantic_seg=subset_semantic_seg,   # shape [M]
    point_indices=point_indices,        # shape [M], 指向原始 [N] 点云
    label_names={1: "handle"}
)
```

adapter 会把 `[M]` 子采样结果恢复为 `[N]` 原始点云 mask。

### 4.4 JSON 格式

```json
{
  "parts": [
    {
      "name": "handle",
      "indices": [0, 12, 31, 48],
      "score": 0.91
    },
    {
      "name": "button",
      "indices": [100, 101, 102]
    }
  ]
}
```

也支持 JSON 中提供 `semantic_seg / instance_seg / label_names` 或 `candidate_masks / candidate_names`。

## 5. 服务器端完整操作流程

下面所有命令都按服务器路径 `/home/lzq/Multi-EE-3DAG/MultiEEAffordance/external/partslippp` 编写。

### 5.1 设置路径变量

```bash
export MULTIEE_ROOT=/home/lzq/Multi-EE-3DAG
export DATASET_ROOT=$MULTIEE_ROOT/MultiEEAffordance
export PARTSLIPPP_HOME=$DATASET_ROOT/external/partslippp
export PARTSLIPPP_OUTPUT_ROOT=$PARTSLIPPP_HOME/outputs
```

### 5.2 下载 PartSLIP++ 官方仓库

如果目录不存在：

```bash
mkdir -p $DATASET_ROOT/external
git clone https://github.com/zyc00/PartSLIP2.git $PARTSLIPPP_HOME
cd $PARTSLIPPP_HOME
```

如果之前已经 clone 过：

```bash
cd $PARTSLIPPP_HOME
git pull
```

创建输出目录：

```bash
mkdir -p $PARTSLIPPP_OUTPUT_ROOT
```

### 5.3 创建 PartSLIP++ 专用环境

官方仓库提供 `environment.yml`：

```bash
cd $PARTSLIPPP_HOME
conda env create -f environment.yml -p /home/lzq/conda_envs/partslippp
conda activate /home/lzq/conda_envs/partslippp
```

不要把 PartSLIP++ 装进 `multiee3dag` 或 `multiee3dag_torch26` 环境。PartSLIP++ 依赖 PyTorch3D、GLIP、cut-pursuit、Segment Anything，和当前 v3 pipeline 的 Torch/VLM 依赖混在一起风险很高。

### 5.4 安装 PyTorch3D

```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

### 5.5 初始化并安装 GLIP

```bash
cd $PARTSLIPPP_HOME
git submodule update --init
cd GLIP
python setup.py build develop --user
cd ..
```

### 5.6 编译 cut-pursuit

先确认 Python 版本：

```bash
python -V
```

然后编译：

```bash
export CONDAENV=/home/lzq/conda_envs/partslippp
export PYVER=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

cd $PARTSLIPPP_HOME/partition/cut-pursuit
rm -rf build
mkdir -p build
cd build
cmake .. \
  -DPYTHON_LIBRARY=$CONDAENV/lib/libpython${PYVER}.so \
  -DPYTHON_INCLUDE_DIR=$CONDAENV/include/python${PYVER} \
  -DBOOST_INCLUDEDIR=$CONDAENV/include \
  -DEIGEN3_INCLUDE_DIR=$CONDAENV/include/eigen3
make
```

### 5.7 安装 Segment Anything

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 5.8 下载官方数据和 checkpoint

官方 README 要求：

```text
data/   放 PartNet-Ensembled 数据
model/  放 few-shot checkpoints
```

你需要从官方 Hugging Face 链接下载：

- PartNet-Ensembled dataset；
- PartSLIP++ few-shot checkpoints。

这部分是外部模型资产，不放进当前 Multi-EE-3DAG 仓库。

建议放置为：

```text
/home/lzq/Multi-EE-3DAG/MultiEEAffordance/external/partslippp/data/
/home/lzq/Multi-EE-3DAG/MultiEEAffordance/external/partslippp/model/
```

### 5.9 准备我们的输入点云

当前 v3 样本的点云路径在：

```text
/home/lzq/Multi-EE-3DAG/MultiEEAffordance/processed/metadata/samples_v3_large_batch_v0_1.jsonl
```

每条样本里有：

```text
sample_id
point_cloud_path
object_category
task
target_executor
```

对应点云通常是：

```text
processed/points/3d_affordancenet_full_shape_val_batch_v3/<object_id>.npy
```

PartSLIP++ 官方脚本默认面向 PartNet-Ensembled。用于我们的 3D AffordanceNet 点云时，你有两种方式：

1. 改 PartSLIP++ 的数据读取脚本，让它读取上述 `.npy` 点云。
2. 先把 `.npy` 转成 PartSLIP++ 期望的数据组织，再跑官方 `gen_sp.py` 和 `run_partslip++.py`。

无论采用哪种方式，最终只要把预测结果保存成第 4 节的 `segments.npz/json` 即可接入我们 v3 pipeline。

### 5.10 运行 PartSLIP++ 并导出标准输出

官方流程是：

```bash
cd $PARTSLIPPP_HOME
conda activate /home/lzq/conda_envs/partslippp
python gen_sp.py
python run_partslip++.py
```

如果你改了数据读取逻辑，请确保每个 `pilot_id` 最终能得到：

```text
/home/lzq/Multi-EE-3DAG/MultiEEAffordance/external/partslippp/outputs/<pilot_id>/segments.npz
```

最小可用输出是：

```python
semantic_seg.shape == [N]
label_names = {label_id: "part name"}
```

## 6. 用 PartSLIP++ 候选跑 v3 小批测试

### 6.1 先生成小批队列

```bash
python MultiEEAffordance/tools/build_large_scale_review_queue.py \
  --dataset-root MultiEEAffordance \
  --samples processed/metadata/samples_v3_large_batch_v0_1.jsonl \
  --output-csv processed/metadata/v3_test_review_queue_v0_1.csv \
  --summary-json processed/metadata/v3_test_review_queue_summary_v0_1.json \
  --include-tasks pick_up,open_pull,press_push \
  --exclude-tasks lift_carry \
  --executor-scope all \
  --quality-scope all \
  --empty-policy review \
  --common-sense-filter \
  --limit 24 \
  --limit-strategy round_robin_category_task_executor \
  --overwrite
```

### 6.2 只跑 PartSLIP++ 候选和 overlay

```bash
python MultiEEAffordance/tools/run_v3_pipeline.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml \
  --pilot-csv processed/metadata/v3_test_review_queue_v0_1.csv \
  --samples processed/metadata/samples_v3_large_batch_v0_1.jsonl \
  --include-tasks pick_up,open_pull,press_push \
  --exclude-tasks lift_carry \
  --candidate-source partseg \
  --part-proposal-backend partslippp \
  --partslippp-root external/partslippp/outputs \
  --stages views,part_propose,render \
  --limit 6 \
  --allow-empty \
  --overwrite
```

如果暂时有些样本还没有 PartSLIP++ 输出，只想先看 pipeline 是否能走通，可以临时使用：

```bash
--partslippp-fallback geometry
```

正式评估 PartSLIP++ 效果时不要加 fallback，否则你可能看不出哪些样本其实没有 PartSLIP++ 结果。

### 6.3 小批完整 VLM 选择

```bash
CUDA_VISIBLE_DEVICES=1,2 python MultiEEAffordance/tools/run_v3_pipeline.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml \
  --pilot-csv processed/metadata/v3_test_review_queue_v0_1.csv \
  --samples processed/metadata/samples_v3_large_batch_v0_1.jsonl \
  --include-tasks pick_up,open_pull,press_push \
  --exclude-tasks lift_carry \
  --candidate-source partseg \
  --part-proposal-backend partslippp \
  --partslippp-root external/partslippp/outputs \
  --stages views,plan,part_propose,render,part_select,part_filter,build \
  --limit 6 \
  --allow-empty \
  --overwrite
```

检查：

```text
processed/vlm_candidate_v3/3d_candidates/<pilot_id>/candidate_manifest.json
processed/vlm_candidate_v3/candidate_overlays/<pilot_id>/
processed/vlm_candidate_v3/vlm_selection/<pilot_id>/combined_selection.json
processed/vlm_candidate_v3/rule_filter/<pilot_id>/combined_rule_filter.json
processed/metadata/v3_candidate_samples_v0_1.jsonl
```

规则层对 PartSLIP++ 有一条额外保护：如果某个候选来自 PartSLIP++，并且被 VLM 选中，会获得额外加权，使它更容易进入默认候选；但过大主体区域、明显不适合当前执行器的候选仍会被规则和人工审查拦住。

## 7. 人工审查

```bash
python MultiEEAffordance/tools/serve_v2_annotation_app.py \
  --dataset-root MultiEEAffordance \
  --samples processed/metadata/v3_candidate_samples_v0_1.jsonl \
  --review-jsonl processed/metadata/v3_point_level_review_records.jsonl \
  --output-mask-root processed/vlm_candidate_v3/manual_refined_masks \
  --output-samples processed/metadata/v3_manual_refined_samples_v0_1.jsonl \
  --port 8765 \
  --max-points 0 \
  --top-k-candidates 8
```

浏览器打开：

```text
http://127.0.0.1:8765
```

人工检查重点：

- 候选是否是完整语义部件，而不是几何碎片；
- VLM 是否能根据 `task + executor` 选中正确部件；
- 规则过滤是否把明显错误区域压下去；
- 人工是否只需要选择候选组合和少量点级修正。

## 8. 全量运行

小批效果稳定后，再生成不带 `--limit` 的全量队列：

```bash
python MultiEEAffordance/tools/build_large_scale_review_queue.py \
  --dataset-root MultiEEAffordance \
  --samples processed/metadata/samples_v3_large_batch_v0_1.jsonl \
  --output-csv processed/metadata/v3_full_review_queue_v0_1.csv \
  --summary-json processed/metadata/v3_full_review_queue_summary_v0_1.json \
  --include-tasks pick_up,open_pull,press_push \
  --exclude-tasks lift_carry \
  --executor-scope all \
  --quality-scope all \
  --empty-policy review \
  --common-sense-filter \
  --limit-strategy round_robin_category_task_executor \
  --overwrite
```

然后运行：

```bash
CUDA_VISIBLE_DEVICES=1,2 python MultiEEAffordance/tools/run_v3_pipeline.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml \
  --pilot-csv processed/metadata/v3_full_review_queue_v0_1.csv \
  --samples processed/metadata/samples_v3_large_batch_v0_1.jsonl \
  --include-tasks pick_up,open_pull,press_push \
  --exclude-tasks lift_carry \
  --candidate-source partseg \
  --part-proposal-backend partslippp \
  --partslippp-root external/partslippp/outputs \
  --stages views,plan,part_propose,render,part_select,part_filter,build \
  --allow-empty \
  --overwrite
```

## 9. 常见问题

### PartSLIP++ 没有输出怎么办

默认会报错。真实评估时这是正确行为，因为它能提醒你哪些样本缺失外部预测。

调试 pipeline 时可以加：

```bash
--partslippp-fallback geometry
```

### PartSLIP++ 输出点数和原始点云不一致怎么办

如果是子采样，请保存 `point_indices`。如果是重新采样且没有原始索引，就不能安全接入，因为 adapter 无法知道预测标签对应原始点云中的哪个点。

### PartSLIP++ 输出太碎怎么办

先检查 `candidate_manifest.json` 中每个候选的 `point_count` 和 `point_fraction`。必要时可以调：

```bash
--min-points 8
--max-candidates 12
--max-candidate-fraction 0.5
```

### PartSLIP++ 输出覆盖整个主体怎么办

大主体候选会被 `--max-candidate-fraction` 过滤。对于 hook/gripper 等需要细部件的执行器，后续 `part_filter` 还会按执行器规则保守筛选。

## 10. v0.2 hybrid 接入方式

当前不建议把 PartSLIP++ 作为唯一候选生成器。v0.2 研发分支采用 hybrid 路由：

- 类别在 `configs/partslippp_category_map.json` 中有映射时，读取 PartSLIP++ 输出作为 primary candidates。
- 始终保留 `high_recall` supplement，补充小部件、局部可见部件、细长结构和弱标签先验。
- 类别未映射、输出缺失或格式不匹配时，自动 fallback 到 `high_recall`，不会中断整批 pipeline。
- `candidate_manifest.json` 会记录 `partslippp_status` 和 `fallback_reason`，用于排查候选为空到底是模型未覆盖、输出缺失，还是高召回生成器也失败。

推荐命令：

```bash
python MultiEEAffordance/tools/run_v3_pipeline.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml \
  --pilot-csv processed/metadata/v3_test_review_queue_v0_2.csv \
  --samples processed/metadata/samples_v3_large_batch_v0_1.jsonl \
  --include-tasks pick_up,open_pull,press_push \
  --exclude-tasks lift_carry \
  --candidate-source partseg \
  --part-proposal-backend hybrid_partslippp_high_recall \
  --partslippp-root external/partslippp/outputs \
  --partslippp-category-map configs/partslippp_category_map.json \
  --stages views,part_propose,render \
  --limit 6 \
  --proposal-max-candidates 64 \
  --max-candidates 24 \
  --part-top-k 5 \
  --allow-empty \
  --overwrite
```

外部 PartSLIP++ 输出仍应放在独立环境中生成；Multi-EE 主环境只读取 normalized `.npz` / `.json` 候选，不在主 pipeline 内强行安装或调用 PartSLIP++。
