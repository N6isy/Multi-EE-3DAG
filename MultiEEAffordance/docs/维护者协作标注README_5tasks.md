# 维护者协作标注 README

更新时间：2026-05-29

本文档只面向维护者。维护者负责准备数据、生成自动候选、拆分标注批次、发给两位审查者、收回结果、合并并发布已审查数据集。

审查者不需要理解整条 v3 pipeline，也不需要运行 VLM 或 GPU 推理。审查者只需要按照 `双人协作标注README.md` 本地打开网页做点级审查。维护者要保证他们拿到的数据包可以直接运行。

当前协作身份统一使用：

- `reviewer_a`
- `reviewer_b`

不要把真实姓名写进路径名、文件名或脚本参数。这样合并和排查时更稳定。

## 0.1 当前存储约定

从本轮开始，维护者生成“网页人工审查所需输入文件”时，推荐把中间候选和待审查 samples 写到数据存储盘：

```text
/home/lzq/data/MultiEEAffordance/
```

这个目录不是代码仓库，而是一个可打包的数据镜像目录。它里面保存的路径结构尽量和项目里的 `MultiEEAffordance/` 一致，例如：

```text
/home/lzq/data/MultiEEAffordance/processed/vlm_candidate_v3/3d_candidates/
/home/lzq/data/MultiEEAffordance/processed/vlm_candidate_v3/fused_masks/
/home/lzq/data/MultiEEAffordance/processed/metadata/v3_candidate_samples_v0_1.jsonl
```

这样做的好处是：

1. 服务器中间文件不挤占项目仓库目录。
2. `v3_candidate_samples_v0_1.jsonl` 里写入的是 `processed/...` 这种相对路径，审查者把数据包解压到本地项目下也能读取。
3. 审查者的人工输出仍然写回当前项目下的 `processed/annotation_batches/v0_1/`，不要写进 `/home/lzq/data`。

服务器 IP 记录为：

```text
10.24.1.11
```

如果维护者要把审查输入包发给审查者，打包时从 `/home/lzq/data/MultiEEAffordance/` 复制所需的 `processed/...` 文件，审查者解压到自己仓库的 `MultiEEAffordance/` 目录下即可。


## 0.2 当前 train / val 双线运行记录与防覆盖规则

本节记录 2026-05-28 当前正在使用的两条数据生成线。核心原则是：

```text
val 线继续使用 val_batch_v3 / vlm_candidate_v3 默认目录；
train 线全部使用 train_100pcat / vlm_candidate_v3_train_100pcat 专用目录；
两条线的 manifest、samples、review queue、candidate、fused mask、summary、split-dir 均不得共名。
```

不要为了省事让 train 复用 val 的这些路径：

```text
processed/vlm_candidate_v3/
processed/metadata/v3_candidate_samples_v0_1.jsonl
processed/metadata/v3_candidate_summary_v0_1.json
splits_v3_candidates/
```

如果 train 和 val 共用这些默认路径，再加 `--overwrite`，会覆盖或污染正在生成的 val 数据。

### 0.2.1 当前 val 数据线

val 线用于验证流程和小批人工审查。当前 val 的主版本名是：

```text
val_batch_v3
```

#### 阶段 A：3D AffordanceNet val -> 项目 manifest

输入：

```text
MultiEEAffordance/raw/3d_affordancenet/full-shape.zip
zip 内部 entry: full_shape_val_data.pkl
```

命令：

```bash
python MultiEEAffordance/tools/convert_3d_affordancenet.py \
  --dataset-root MultiEEAffordance \
  --zip raw/3d_affordancenet/full-shape.zip \
  --source-split val \
  --target-split val \
  --tasks pick_up,open_pull,press_push \
  --max-per-category 20 \
  --points-dir processed/points/3d_affordancenet_full_shape_val_batch_v3 \
  --candidate-dir processed/candidates/3d_affordancenet_full_shape_val_batch_v3 \
  --mask-dir processed/masks/3d_affordancenet_full_shape_val_batch_v3 \
  --manifest manifests/3d_affordancenet_full_shape_val_batch_v3_manifest.jsonl \
  --summary processed/metadata/3d_affordancenet_full_shape_val_batch_v3_summary.json \
  --overwrite
```

输出：

```text
MultiEEAffordance/processed/points/3d_affordancenet_full_shape_val_batch_v3/
MultiEEAffordance/processed/candidates/3d_affordancenet_full_shape_val_batch_v3/
MultiEEAffordance/processed/masks/3d_affordancenet_full_shape_val_batch_v3/
MultiEEAffordance/manifests/3d_affordancenet_full_shape_val_batch_v3_manifest.jsonl
MultiEEAffordance/processed/metadata/3d_affordancenet_full_shape_val_batch_v3_summary.json
```

#### 阶段 B：val manifest -> samples JSONL

输入：

```text
MultiEEAffordance/manifests/3d_affordancenet_full_shape_val_batch_v3_manifest.jsonl
```

命令：

```bash
python MultiEEAffordance/tools/build_samples_jsonl.py \
  --dataset-root MultiEEAffordance \
  --manifest manifests/3d_affordancenet_full_shape_val_batch_v3_manifest.jsonl \
  --output processed/metadata/samples_v3_large_batch_v0_1.jsonl \
  --default-split val \
  --default-quality weak \
  --write-splits \
  --split-dir splits_v3_large_batch
```

输出：

```text
MultiEEAffordance/processed/metadata/samples_v3_large_batch_v0_1.jsonl
MultiEEAffordance/splits_v3_large_batch/val.txt
```

#### 阶段 C：val samples -> review queue

输入：

```text
MultiEEAffordance/processed/metadata/samples_v3_large_batch_v0_1.jsonl
```

当前推荐输出写到数据盘，避免后续打包时再从项目目录搬运：

```bash
python MultiEEAffordance/tools/build_large_scale_review_queue.py \
  --dataset-root MultiEEAffordance \
  --samples processed/metadata/samples_v3_large_batch_v0_1.jsonl \
  --output-csv /home/lzq/data/MultiEEAffordance/processed/metadata/v3_large_scale_review_queue_v0_1.csv \
  --summary-json /home/lzq/data/MultiEEAffordance/processed/metadata/v3_large_scale_review_queue_summary_v0_1.json \
  --include-tasks pick_up,open_pull,press_push \
  --exclude-tasks lift_carry \
  --executor-scope all \
  --quality-scope all \
  --empty-policy review \
  --common-sense-filter \
  --limit-strategy round_robin_category_task_executor \
  --overwrite
```

输出：

```text
/home/lzq/data/MultiEEAffordance/processed/metadata/v3_large_scale_review_queue_v0_1.csv
/home/lzq/data/MultiEEAffordance/processed/metadata/v3_large_scale_review_queue_summary_v0_1.json
```

如果只是 smoke test，可以临时加：

```bash
--limit 300
```

正式生成候选时不要误留小 `--limit`，否则后续候选只覆盖被限制的那一小批。

#### 阶段 D：val review queue -> 规则候选与待审查 samples

输入：

```text
/home/lzq/data/MultiEEAffordance/processed/metadata/v3_large_scale_review_queue_v0_1.csv
MultiEEAffordance/processed/metadata/samples_v3_large_batch_v0_1.jsonl
```

命令：

```bash
python MultiEEAffordance/tools/run_v3_pipeline.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml \
  --pilot-csv /home/lzq/data/MultiEEAffordance/processed/metadata/v3_large_scale_review_queue_v0_1.csv \
  --samples processed/metadata/samples_v3_large_batch_v0_1.jsonl \
  --include-tasks pick_up,open_pull,press_push \
  --exclude-tasks lift_carry \
  --include-decisions all \
  --candidate-source partseg \
  --part-proposal-backend high_recall \
  --stages part_propose,build \
  --data-storage-root /home/lzq/data \
  --proposal-max-candidates 64 \
  --max-candidates 12 \
  --part-top-k 5 \
  --allow-empty \
  --overwrite
```

主要输出：

```text
/home/lzq/data/MultiEEAffordance/processed/vlm_candidate_v3/3d_candidates/
/home/lzq/data/MultiEEAffordance/processed/vlm_candidate_v3/fused_masks/
/home/lzq/data/MultiEEAffordance/processed/vlm_candidate_v3/pipeline_runs/latest_run_manifest.json
/home/lzq/data/MultiEEAffordance/processed/metadata/v3_candidate_samples_v0_1.jsonl
/home/lzq/data/MultiEEAffordance/processed/metadata/v3_candidate_summary_v0_1.json
/home/lzq/data/MultiEEAffordance/splits_v3_candidates/
```

val 线的默认输出名保留不变，是为了继续兼容前面已有标注拆包脚本。但 train 线必须显式改名，不能复用这些默认路径。

### 0.2.2 当前 train 数据线

train 线用于扩大数据规模。当前 train 的主版本名是：

```text
train_100pcat
```

其中 `100pcat` 表示：

```text
convert_3d_affordancenet.py 阶段使用 --max-per-category 100
```

注意：`--max-per-category` 只属于 `convert_3d_affordancenet.py`，不是 `build_large_scale_review_queue.py` 或 `run_v3_pipeline.py` 的参数。

#### 阶段 A：3D AffordanceNet train -> 项目 manifest

输入：

```text
MultiEEAffordance/raw/3d_affordancenet/full-shape.zip
zip 内部 entry: full_shape_train_data.pkl
```

命令：

```bash
python MultiEEAffordance/tools/convert_3d_affordancenet.py \
  --dataset-root MultiEEAffordance \
  --zip raw/3d_affordancenet/full-shape.zip \
  --source-split train \
  --target-split train \
  --tasks pick_up,open_pull,press_push \
  --max-per-category 100 \
  --points-dir processed/points/3d_affordancenet_full_shape_train_100pcat_v3 \
  --candidate-dir processed/candidates/3d_affordancenet_full_shape_train_100pcat_v3 \
  --mask-dir processed/masks/3d_affordancenet_full_shape_train_100pcat_v3 \
  --manifest manifests/3d_affordancenet_full_shape_train_100pcat_v3_manifest.jsonl \
  --summary processed/metadata/3d_affordancenet_full_shape_train_100pcat_v3_summary.json \
  --overwrite
```

输出：

```text
MultiEEAffordance/processed/points/3d_affordancenet_full_shape_train_100pcat_v3/
MultiEEAffordance/processed/candidates/3d_affordancenet_full_shape_train_100pcat_v3/
MultiEEAffordance/processed/masks/3d_affordancenet_full_shape_train_100pcat_v3/
MultiEEAffordance/manifests/3d_affordancenet_full_shape_train_100pcat_v3_manifest.jsonl
MultiEEAffordance/processed/metadata/3d_affordancenet_full_shape_train_100pcat_v3_summary.json
```

#### 阶段 B：train manifest -> samples JSONL

输入：

```text
MultiEEAffordance/manifests/3d_affordancenet_full_shape_train_100pcat_v3_manifest.jsonl
```

命令：

```bash
python MultiEEAffordance/tools/build_samples_jsonl.py \
  --dataset-root MultiEEAffordance \
  --manifest manifests/3d_affordancenet_full_shape_train_100pcat_v3_manifest.jsonl \
  --output processed/metadata/samples_v3_large_train_100pcat_v0_1.jsonl \
  --default-split train \
  --default-quality weak \
  --write-splits \
  --split-dir splits_v3_large_train_100pcat
```

输出：

```text
MultiEEAffordance/processed/metadata/samples_v3_large_train_100pcat_v0_1.jsonl
MultiEEAffordance/splits_v3_large_train_100pcat/train.txt
```

#### 阶段 C：train samples -> review queue

输入：

```text
MultiEEAffordance/processed/metadata/samples_v3_large_train_100pcat_v0_1.jsonl
```

命令：

```bash
python MultiEEAffordance/tools/build_large_scale_review_queue.py \
  --dataset-root MultiEEAffordance \
  --samples processed/metadata/samples_v3_large_train_100pcat_v0_1.jsonl \
  --output-csv /home/lzq/data/MultiEEAffordance/processed/metadata/v3_large_scale_review_queue_train_100pcat_v0_1.csv \
  --summary-json /home/lzq/data/MultiEEAffordance/processed/metadata/v3_large_scale_review_queue_train_100pcat_summary_v0_1.json \
  --include-tasks pick_up,open_pull,press_push \
  --exclude-tasks lift_carry \
  --executor-scope all \
  --quality-scope all \
  --empty-policy review \
  --common-sense-filter \
  --limit-strategy round_robin_category_task_executor \
  --overwrite
```

输出：

```text
/home/lzq/data/MultiEEAffordance/processed/metadata/v3_large_scale_review_queue_train_100pcat_v0_1.csv
/home/lzq/data/MultiEEAffordance/processed/metadata/v3_large_scale_review_queue_train_100pcat_summary_v0_1.json
```

按前面 `50pcat -> 5996 rows` 的实际比例估算，`100pcat` 的 review queue 约为 1.2 万行。若目标是 1.5 万行左右，后续可改成 `--max-per-category 125`，但当前记录以正在运行的 `100pcat` 为准。

#### 阶段 D：train review queue -> 规则候选与待审查 samples

输入：

```text
/home/lzq/data/MultiEEAffordance/processed/metadata/v3_large_scale_review_queue_train_100pcat_v0_1.csv
MultiEEAffordance/processed/metadata/samples_v3_large_train_100pcat_v0_1.jsonl
```

当前正在使用的 train 参数如下：

```bash
python MultiEEAffordance/tools/run_v3_pipeline.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml \
  --pilot-csv /home/lzq/data/MultiEEAffordance/processed/metadata/v3_large_scale_review_queue_train_100pcat_v0_1.csv \
  --samples processed/metadata/samples_v3_large_train_100pcat_v0_1.jsonl \
  --include-tasks pick_up,open_pull,press_push \
  --exclude-tasks lift_carry \
  --include-decisions all \
  --candidate-source partseg \
  --part-proposal-backend high_recall \
  --stages part_propose,build \
  --data-storage-root /home/lzq/data \
  --v3-output-root /home/lzq/data/MultiEEAffordance/processed/vlm_candidate_v3_train_100pcat \
  --renders-root /home/lzq/data/MultiEEAffordance/processed/vlm_semantic_part/renders_train_100pcat \
  --review-output-samples /home/lzq/data/MultiEEAffordance/processed/metadata/v3_candidate_samples_train_100pcat_v0_1.jsonl \
  --review-summary-json /home/lzq/data/MultiEEAffordance/processed/metadata/v3_candidate_summary_train_100pcat_v0_1.json \
  --review-split-dir /home/lzq/data/MultiEEAffordance/splits_v3_candidates_train_100pcat \
  --run-manifest /home/lzq/data/MultiEEAffordance/processed/vlm_candidate_v3_train_100pcat/pipeline_runs/latest_run_manifest.json \
  --proposal-max-candidates 16 \
  --max-candidates 6 \
  --part-top-k 5 \
  --allow-empty \
  --overwrite
```

主要输出：

```text
/home/lzq/data/MultiEEAffordance/processed/vlm_candidate_v3_train_100pcat/3d_candidates/
/home/lzq/data/MultiEEAffordance/processed/vlm_candidate_v3_train_100pcat/fused_masks/
/home/lzq/data/MultiEEAffordance/processed/vlm_candidate_v3_train_100pcat/pipeline_runs/latest_run_manifest.json
/home/lzq/data/MultiEEAffordance/processed/vlm_semantic_part/renders_train_100pcat/
/home/lzq/data/MultiEEAffordance/processed/metadata/v3_candidate_samples_train_100pcat_v0_1.jsonl
/home/lzq/data/MultiEEAffordance/processed/metadata/v3_candidate_summary_train_100pcat_v0_1.json
/home/lzq/data/MultiEEAffordance/splits_v3_candidates_train_100pcat/
```

当前 train 线为了控制候选文件数量和运行成本，把候选参数收紧为：

```text
proposal-max-candidates = 16
max-candidates = 6
part-top-k = 5
```

含义：

- `proposal-max-candidates=16`：每条 review 记录最多先生成 16 个高召回候选区域；
- `max-candidates=6`：后续构建和可视化最多保留或展示 6 个候选；
- `part-top-k=5`：每组 part proposal 内保留 top 5。

### 0.2.3 train / val 防覆盖检查清单

在同时跑 val 和 train 前，必须检查下面路径是否分开：

| 阶段 | val 路径 | train 路径 |
| --- | --- | --- |
| convert points | `processed/points/3d_affordancenet_full_shape_val_batch_v3/` | `processed/points/3d_affordancenet_full_shape_train_100pcat_v3/` |
| convert candidates | `processed/candidates/3d_affordancenet_full_shape_val_batch_v3/` | `processed/candidates/3d_affordancenet_full_shape_train_100pcat_v3/` |
| convert masks | `processed/masks/3d_affordancenet_full_shape_val_batch_v3/` | `processed/masks/3d_affordancenet_full_shape_train_100pcat_v3/` |
| manifest | `manifests/3d_affordancenet_full_shape_val_batch_v3_manifest.jsonl` | `manifests/3d_affordancenet_full_shape_train_100pcat_v3_manifest.jsonl` |
| samples | `processed/metadata/samples_v3_large_batch_v0_1.jsonl` | `processed/metadata/samples_v3_large_train_100pcat_v0_1.jsonl` |
| review queue csv | `/home/lzq/data/MultiEEAffordance/processed/metadata/v3_large_scale_review_queue_v0_1.csv` | `/home/lzq/data/MultiEEAffordance/processed/metadata/v3_large_scale_review_queue_train_100pcat_v0_1.csv` |
| review queue summary | `/home/lzq/data/MultiEEAffordance/processed/metadata/v3_large_scale_review_queue_summary_v0_1.json` | `/home/lzq/data/MultiEEAffordance/processed/metadata/v3_large_scale_review_queue_train_100pcat_summary_v0_1.json` |
| v3 candidate root | `/home/lzq/data/MultiEEAffordance/processed/vlm_candidate_v3/` | `/home/lzq/data/MultiEEAffordance/processed/vlm_candidate_v3_train_100pcat/` |
| candidate samples | `/home/lzq/data/MultiEEAffordance/processed/metadata/v3_candidate_samples_v0_1.jsonl` | `/home/lzq/data/MultiEEAffordance/processed/metadata/v3_candidate_samples_train_100pcat_v0_1.jsonl` |
| candidate summary | `/home/lzq/data/MultiEEAffordance/processed/metadata/v3_candidate_summary_v0_1.json` | `/home/lzq/data/MultiEEAffordance/processed/metadata/v3_candidate_summary_train_100pcat_v0_1.json` |
| split dir | `/home/lzq/data/MultiEEAffordance/splits_v3_candidates/` | `/home/lzq/data/MultiEEAffordance/splits_v3_candidates_train_100pcat/` |
| run manifest | `/home/lzq/data/MultiEEAffordance/processed/vlm_candidate_v3/pipeline_runs/latest_run_manifest.json` | `/home/lzq/data/MultiEEAffordance/processed/vlm_candidate_v3_train_100pcat/pipeline_runs/latest_run_manifest.json` |

只要表中 train 和 val 路径不同，就可以保留 `--overwrite`。如果发现任意路径共名，必须先改路径再运行。

## 0.3 本地反向隧道与数据传输记录

当前远程服务器记录为：

```text
服务器 IP：10.24.1.11
服务器项目根目录：/home/lzq/Multi-EE-3DAG
服务器数据镜像根目录：/home/lzq/data/MultiEEAffordance
```

数据传输原则：

```text
服务器负责生成 /home/lzq/data/MultiEEAffordance 下的数据镜像；
本地通过 SSH/隧道从服务器拉取需要的数据包；
不要把本地 Windows 绝对路径写进 samples JSONL；
不要让审查者直接依赖 /home/lzq/data 这个服务器路径。
```

当前 README 里暂不写死本地反向隧道端口，因为端口可能随本地网络和隧道会话变化。维护者实际使用时，把下面模板中的 `<LOCAL_TUNNEL_PORT>` 替换为当前本地反向隧道映射端口。

如果服务器可以直接访问，可以从本地执行：

```bash
rsync -avP lzq@10.24.1.11:/home/lzq/data/MultiEEAffordance/processed/metadata/ \
  ./MultiEEAffordance/processed/metadata/
```

如果当前通过本地反向隧道访问服务器，则从本地执行：

```bash
rsync -avP -e "ssh -p <LOCAL_TUNNEL_PORT>" \
  lzq@127.0.0.1:/home/lzq/data/MultiEEAffordance/processed/metadata/ \
  ./MultiEEAffordance/processed/metadata/
```

拉取完整 reviewer 数据包时，推荐先在服务器打包，再从本地拉取 tar.gz：

```bash
rsync -avP -e "ssh -p <LOCAL_TUNNEL_PORT>" \
  lzq@127.0.0.1:/home/lzq/data/MultiEEAffordance/processed/annotation_batches/v0_1/packages/reviewer_b_annotation_package_v0_1.tar.gz \
  ./MultiEEAffordance/processed/annotation_batches/v0_1/packages/
```

如果从本地回传审查结果包到服务器：

```bash
rsync -avP -e "ssh -p <LOCAL_TUNNEL_PORT>" \
  ./MultiEEAffordance/processed/annotation_batches/v0_1/returned/reviewer_x_result_YYYYMMDD.tar.gz \
  lzq@127.0.0.1:/home/lzq/data/MultiEEAffordance/processed/annotation_batches/v0_1/returned/
```

维护者每次正式传输前应记录：

```text
传输日期
本地机器
隧道方式：direct / reverse-ssh / tailscale / other
本地映射端口
服务器目标路径
传输文件名
文件大小
sha256sum
```

建议在本地和服务器分别执行：

```bash
sha256sum reviewer_x_result_YYYYMMDD.tar.gz
```

两边 hash 一致后，再进入合并步骤。

## 0.4 任务设定变更与候选复用原则

当前稳定任务集为：

```text
pick_up
open_pull
press_push
```

`lift_carry` 暂时排除，原因是它与 `pick_up` 在当前 object-level 几何标注中边界过近，容易让审查者把两者标成几乎相同的 mask。

如果后续要按 3D AffordanceNet 原始 affordance 类别扩展任务，不能直接把 affordance 名称机械等价为任务名。更稳妥的做法是分成两层：

```text
底层 affordance primitive：grasp / wrap_grasp / lift / support / layable / pull / openable / pushable / press 等
上层 Multi-EE task：pick_up / open_pull / press_push / 后续新增 task
```

可考虑的任务扩展方向：

| 候选任务 | 可利用的 3D AffordanceNet affordance primitive | 是否建议当前加入 | 原因 |
| --- | --- | --- | --- |
| `grasp_hold` / `grasp` | `grasp`, `wrap_grasp` | 暂不建议单独加入 | 与 `pick_up` 高度重叠，除非任务定义为“不要求拿起，只要求稳定抓住” |
| `support_place` / `place_on` | `support`, `layable` | 可以作为后续扩展 | 对 suction、dexterous_hand、gripper 的差异可能明显，但需要重新定义任务方向 |
| `open_pull` | `pull`, `openable` | 已加入 | 是当前 articulated / handle 类任务主线 |
| `press_push` | `press`, `pushable` | 已加入 | 是按钮、开关、面板类任务主线 |
| `lift_carry` | `lift`, `grasp`, `wrap_grasp` | 暂缓 | 与 `pick_up` 边界不清，除非把“持续承重搬运”写成更严格任务 |
| `hang_hook` / `hook_pull` | 原始数据通常没有显式 hook primitive | 后续可以做 contrast_test | 需要 PartNet-Mobility 或人工规则补孔洞/拉环/把手内侧，不适合只靠 3D AffordanceNet 映射 |

是否需要重跑候选，按下面原则判断：

| 改动类型 | 是否需要重跑 convert | 是否需要重跑 review queue | 是否需要重跑 run_v3_pipeline | 说明 |
| --- | --- | --- | --- | --- |
| 只改任务说明文字，不改 task 名称、不改样本集合 | 否 | 否 | 否 | 更新 README / guideline 即可 |
| 只调整哪些 object-task 进入人工审查，例如打开或关闭 `--common-sense-filter` | 否 | 是 | 对新增进入队列的行需要跑 | 已生成候选的旧行可复用 |
| 新增任务名，例如 `support_place` | 是 | 是 | 是 | 需要生成新的 object-task sample 和 mask |
| 修改 `convert_3d_affordancenet.py` 中任务到 affordance 的弱标签映射 | 是 | 是 | 是 | mask 和 feasibility 都会变化 |
| 只新增执行器通道 | 是 | 是 | 是 | `[N,4]` schema 变化，属于破坏性变更 |
| 只修改 `part_propose` 的候选数量参数 | 否 | 否 | 是 | 可只重跑 part_propose/build，输入 samples 不变 |
| 只修改 build 阶段如何合并候选或是否允许空样本 | 否 | 否 | 只重跑 build | 可以复用已有 `3d_candidates/` |
| 只改人工审查标准 | 否 | 否 | 否 | 不重跑自动候选，但需要在人工端重新审查受影响样本 |

如果当前 `run_v3_pipeline.py` 已经在跑 train 或 val，不建议中途改任务 taxonomy。正确做法是：

```text
先让当前三任务版本完整跑完并保存；
把新任务作为 v0.2_task_expansion 分支或新版本目录；
对新增任务只重跑新增任务对应的 convert -> samples -> review_queue -> pipeline；
不要覆盖当前 v0.1 三任务数据。
```

已有候选能否复用，取决于 sample_id 是否不变：

```text
同一个 object_id + 同一个 task + 同一个 executor：
  可以复用已有 candidate_manifest 和 candidates.npz。

同一个 object_id 但 task 改了：
  不建议直接复用最终样本；
  最多把已有 part proposal 当作几何候选底座，重新跑 build 和人工审查。

新增 task：
  必须生成新 sample_id，例如 xxx_support_place；
  必须重新生成或至少重新绑定候选。
```

当前建议：

```text
短期不要立刻改动三任务主线；
先完成 train_100pcat 和 val_batch_v3 的候选生成；
再用一个小分支尝试从 3D AffordanceNet affordance primitive 中新增 1 个任务，例如 support_place；
新增任务先做 20~50 个样本的 pilot，不要直接全量重跑。
```

## 0. 维护者到底要做什么

维护者的工作可以理解成 6 件事：

1. 准备原始数据。
2. 把原始数据转换成项目内部样本格式。
3. 生成一批需要人工审查的样本队列。
4. 跑轻量自动候选，生成网页能读取的候选 mask。
5. 把队列拆成 `reviewer_a` 和 `reviewer_b` 两份本地数据包。
6. 收回两份结果，检查、合并、生成已审查数据集。

推荐流程是：

```text
raw 3D AffordanceNet
  -> convert_3d_affordancenet.py
  -> build_samples_jsonl.py
  -> build_large_scale_review_queue.py
  -> run_v3_pipeline.py
  -> split reviewer packages
  -> reviewers local annotation
  -> collect reviewer result packages
  -> merge reviewed release
```

## 1. 当前原始数据在哪里

服务器上当前原始数据已经解压在：

```text
/home/lzq/Multi-EE-3DAG/MultiEEAffordance/raw/3d_affordancenet/full_shape_train_data.pkl
/home/lzq/Multi-EE-3DAG/MultiEEAffordance/raw/3d_affordancenet/full_shape_val_data.pkl
```

原始 zip 也建议保留：

```text
/home/lzq/Multi-EE-3DAG/MultiEEAffordance/raw/3d_affordancenet/full-shape.zip
```

这里需要注意一个实现细节：

当前 `MultiEEAffordance/tools/convert_3d_affordancenet.py` 是直接读取 `full-shape.zip`，然后从 zip 内部找：

```text
full_shape_train_data.pkl
full_shape_val_data.pkl
```

所以即使 pkl 文件已经解压出来，当前转换命令仍然应该使用：

```bash
--zip raw/3d_affordancenet/full-shape.zip
```

不要直接把 `--zip` 改成：

```text
raw/3d_affordancenet/full_shape_val_data.pkl
```

否则当前脚本会把 pkl 当 zip 打开，导致报错。

如果后面只保留了解压后的 pkl、没有保留 `full-shape.zip`，有两种选择：

1. 给 `convert_3d_affordancenet.py` 增加 `--pkl` 参数，让它直接读取 pkl。
2. 重新打包一个满足当前脚本预期的 zip。

临时重新打包命令如下：

```bash
cd /home/lzq/Multi-EE-3DAG/MultiEEAffordance/raw/3d_affordancenet
zip -j full-shape.zip full_shape_train_data.pkl full_shape_val_data.pkl
```

当前建议保留 zip，不急着改 converter。

## 2. 这些 pkl 是不是当前需要的全部数据

对当前这条 3D AffordanceNet full-shape 标注流程来说，需要的数据主要就是：

```text
full_shape_train_data.pkl
full_shape_val_data.pkl
```

其中：

- `train` 是训练 split。
- `val` 是验证 split。
- 当前人工审查批次一般先从 `val` 开始做，因为体量更可控，适合先验证流程。

但这不等于整个项目所有可能用到的数据都只有这两个文件。项目里还有其他来源或未来可能接入的数据，例如 PartNet-Mobility。它们不属于当前 3D AffordanceNet full-shape 批次的主输入。

当前维护者可以先把目标明确为：

```text
先基于 3D AffordanceNet full-shape 的 val split，生成一版稳定可审查数据集。
```

流程稳定后，再扩大到 train split 或其他数据源。

## 3. 维护者环境要求

维护者需要两类环境。

第一类是普通 Python 环境，用于数据转换、队列生成、拆包、合并：

```text
Python
numpy
项目脚本
```

第二类是带 GPU 和 VLM 依赖的环境，用于跑自动候选：

```text
Qwen3-VL
torch
CUDA
configs/qwen3vl_sam2_pilot.yaml 中配置的模型路径
```

审查者本地不需要第二类环境。

维护者在服务器上操作时，建议固定进入项目根目录：

```bash
cd /home/lzq/Multi-EE-3DAG
```

后续所有命令都默认从这个目录执行。

## 4. 推荐目录结构

当前建议把人工标注批次统一放在：

```text
MultiEEAffordance/processed/annotation_batches/v0_1/
```

推荐结构：

```text
processed/annotation_batches/v0_1/
  batch_manifest.json
  batch_summary.json
  reviewer_a_samples.jsonl
  reviewer_b_samples.jsonl
  reviewer_a_review_records.jsonl
  reviewer_b_review_records.jsonl
  reviewer_a_refined_samples.jsonl
  reviewer_b_refined_samples.jsonl
  manual_refined_masks_reviewer_a/
  manual_refined_masks_reviewer_b/
  packages/
    reviewer_a_annotation_package_v0_1.tar.gz
    reviewer_b_annotation_package_v0_1.tar.gz
  returned/
    reviewer_a_result_YYYYMMDD.tar.gz
    reviewer_b_result_YYYYMMDD.tar.gz
  merged/
    merged_review_records.jsonl
    merged_refined_samples.jsonl
    conflict_samples.jsonl
    reviewed_dataset_summary.json
```

原则：

- Git 管代码和文档。
- `raw/`、`processed/`、`.npy`、`.npz`、大压缩包不要直接提交到普通 Git。
- 给审查者的数据包和审查者回传的结果包都放到批次目录，方便以后复现。

## 5. 第一步：把原始数据转换成项目样本

先用 `val` split 做一批。下面命令会从 `full-shape.zip` 读取 `full_shape_val_data.pkl`，并把点云、弱 mask、manifest 写到项目内部目录。

```bash
python MultiEEAffordance/tools/convert_3d_affordancenet.py \
  --dataset-root MultiEEAffordance \
  --zip raw/3d_affordancenet/full-shape.zip \
  --source-split val \
  --target-split val \
  --tasks pick_up,open_pull,press_push \
  --max-per-category 20 \
  --points-dir processed/points/3d_affordancenet_full_shape_val_batch_v3 \
  --candidate-dir processed/candidates/3d_affordancenet_full_shape_val_batch_v3 \
  --mask-dir processed/masks/3d_affordancenet_full_shape_val_batch_v3 \
  --manifest manifests/3d_affordancenet_full_shape_val_batch_v3_manifest.jsonl \
  --summary processed/metadata/3d_affordancenet_full_shape_val_batch_v3_summary.json \
  --overwrite
```

你应该重点看输出里的这些字段：

```text
objects_loaded
objects_converted
samples_written
tasks
categories
feasible_counts_by_task
```

含义：

- `objects_loaded`：原始 pkl 里读到了多少个物体。
- `objects_converted`：实际转换了多少个物体。
- `samples_written`：写出了多少个物体-任务样本。
- `tasks`：每个任务各写出多少条。
- `categories`：每个物体类别各写出多少个物体。
- `feasible_counts_by_task`：弱标签阶段认为哪些任务和执行器可能有正例。

如果 `samples_written` 明显为 0，通常是路径、split 或任务过滤参数错了。

## 6. 第二步：生成统一 samples JSONL

转换后的 manifest 还不是审查系统直接使用的最终样本表。需要再构建 `samples_v3_large_batch_v0_1.jsonl`。

```bash
python MultiEEAffordance/tools/build_samples_jsonl.py \
  --dataset-root MultiEEAffordance \
  --manifest manifests/3d_affordancenet_full_shape_val_batch_v3_manifest.jsonl \
  --output processed/metadata/samples_v3_large_batch_v0_1.jsonl \
  --default-split val \
  --default-quality weak \
  --write-splits \
  --split-dir splits_v3_large_batch
```

输出里要看：

```text
samples_written
splits
```

正常情况下，`samples_written` 应该和上一步的 `samples_written` 一致。

## 7. 第三步：生成人工审查队列

这一步决定哪些样本进入这次人工审查批次。

当前任务策略：

- 保留：`pick_up,open_pull,press_push`
- 删除：`lift_carry`

原因是 `lift_carry` 和 `pick_up` 在当前定义里过于接近，先不作为独立任务推进。

示例命令：

```bash
python MultiEEAffordance/tools/build_large_scale_review_queue.py \
  --dataset-root MultiEEAffordance \
  --samples processed/metadata/samples_v3_large_batch_v0_1.jsonl \
  --output-csv processed/metadata/v3_large_scale_review_queue_v0_1.csv \
  --summary-json processed/metadata/v3_large_scale_review_queue_summary_v0_1.json \
  --include-tasks pick_up,open_pull,press_push \
  --exclude-tasks lift_carry \
  --executor-scope all \
  --quality-scope all \
  --empty-policy review \
  --common-sense-filter \
  --limit-strategy round_robin_category_task_executor \
  --limit 300 \
  --overwrite
```

这里几个参数要理解清楚：

- `--executor-scope all`：同一个物体和任务下，不只保留原始弱标签中有正例的执行器，也保留其他执行器组合。
- `--empty-policy review`：即使某个任务-执行器组合没有候选区域，也保留为待审查空样本。这样才能体现“这个执行器不适合这个任务”。
- `--common-sense-filter`：先过滤明显不合理的物体-任务组合，例如某些物体根本不存在可 open_pull 的结构。
- `--limit 300`：这次先抽 300 条做批次。全量跑时可以去掉或调大。

输出里要看：

```text
rows
counts_by_task
counts_by_executor
counts_by_decision
counts_by_category
skipped
```

如果 `counts_by_category` 只出现很少类别，说明队列抽样策略或 limit 可能偏向了前几个类别，需要调整 `--limit-strategy` 或增大 `--limit`。

## 8. 第四步：先小批 smoke test

当前协作标注版先走“纯规则候选 + 人工审查”，不让 VLM 决定候选是否可用。先用 3 到 5 条样本确认 pipeline 能跑完。

```bash
python MultiEEAffordance/tools/run_v3_pipeline.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml \
  --pilot-csv processed/metadata/v3_large_scale_review_queue_v0_1.csv \
  --samples processed/metadata/samples_v3_large_batch_v0_1.jsonl \
  --include-tasks pick_up,open_pull,press_push \
  --exclude-tasks lift_carry \
  --include-decisions all \
  --candidate-source partseg \
  --part-proposal-backend high_recall \
  --stages part_propose,build \
  --limit 5 \
  --data-storage-root /home/lzq/data \
  --proposal-max-candidates 64 \
  --max-candidates 12 \
  --part-top-k 5 \
  --allow-empty \
  --overwrite
```

检查重点：

1. `part_propose` 是否能产生候选，而不是大量 `candidate_count=0`。
2. `build` 是否能生成待审查 samples。
3. 空样本是否能被保留下来，而不是让 pipeline 崩溃。
4. `/home/lzq/data/MultiEEAffordance/processed/vlm_candidate_v3/3d_candidates/` 是否有候选 manifest 和 npz。
5. `/home/lzq/data/MultiEEAffordance/processed/metadata/v3_candidate_samples_v0_1.jsonl` 是否生成。

如果小批都不稳定，不要继续全量跑。

## 9. 第五步：正式生成自动候选

小批通过后，再跑完整队列。

```bash
python MultiEEAffordance/tools/run_v3_pipeline.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml \
  --pilot-csv processed/metadata/v3_large_scale_review_queue_v0_1.csv \
  --samples processed/metadata/samples_v3_large_batch_v0_1.jsonl \
  --include-tasks pick_up,open_pull,press_push \
  --exclude-tasks lift_carry \
  --include-decisions all \
  --candidate-source partseg \
  --part-proposal-backend high_recall \
  --stages part_propose,build \
  --data-storage-root /home/lzq/data \
  --proposal-max-candidates 64 \
  --max-candidates 12 \
  --part-top-k 5 \
  --allow-empty \
  --overwrite
```

维护者要记录这次运行的：

```text
运行日期
git commit hash
命令完整文本
输入 samples 路径
输入 review queue 路径
输出 candidate samples 路径
错误样本列表
```

建议写到：

```text
processed/annotation_batches/v0_1/batch_manifest.json
```

## 10. 第六步：检查候选质量

自动候选不是最终标注。维护者至少要抽查几类样本：

```text
Bag / pick_up / gripper
Faucet / open_pull / dexterous_hand
Scissors / pick_up / hook
Door / open_pull / suction
Keyboard / press_push / dexterous_hand
Mug / pick_up / suction
```

检查时重点看：

- 自动候选有没有完全空掉。
- 自动候选有没有把大面积主体都选成正例。
- 小部件是否保留下来了，例如 button、handle、ring、knob、switch。
- 对 hook 来说，blade、普通边缘、长杆尖端不应被默认认为可 hook。
- 对 suction 来说，候选应更偏向平整可吸附面。
- 对 dexterous_hand 来说，候选可以比 gripper 更宽，但不能把完全无任务意义的主体都选进去。

如果自动候选质量一般，也可以继续进入人工审查；但要保证网页里有足够候选可选，并且人工可以方便增删点。

## 11. 第七步：展开五任务 samples 并划分审查批次

**当前主流程以本节为准。** 旧版文档里曾经使用 `v0_1_5tasks` 和 `package_reviewer_b_only_annotation_batch_progress.py`，现在统一改为 `v0_2_5tasks` 和 `package_annotation_batches_from_samples.py`。旧任务候选仍然保留，只作为五任务人工审查 proposal。

旧任务到五任务的展开关系：

```text
pick_up    -> lift
open_pull  -> open + pull
press_push -> press + push
lift_carry -> lift
```

先把旧任务候选 samples 展开成五任务 samples：

```bash
python MultiEEAffordance/tools/expand_legacy_tasks_to_5tasks.py \
  --input /home/lzq/data/MultiEEAffordance/processed/metadata/v3_candidate_samples_v0_1.jsonl \
  --output /home/lzq/data/MultiEEAffordance/processed/metadata/v3_candidate_samples_v0_2_5tasks.jsonl \
  --summary-json /home/lzq/data/MultiEEAffordance/processed/metadata/v3_candidate_samples_v0_2_5tasks_summary.json \
  --overwrite
```

展开脚本只改 metadata，不重新生成点云、候选或 mask。输出必须包含 `row_key`、`task_display`、`source_task`、`source_sample_id`、`task_taxonomy_version`、`task_split_source`；其中 `task_taxonomy_version` 应为 `v0_2_5tasks`，`task_split_source` 应为 `legacy_task_expansion`。

然后按 object 分组拆分 reviewer 批次并打包依赖：

```bash
python MultiEEAffordance/tools/package_annotation_batches_from_samples.py \
  --dataset-root /home/lzq/data/MultiEEAffordance \
  --input processed/metadata/v3_candidate_samples_v0_2_5tasks.jsonl \
  --batch-dir processed/annotation_batches/v0_2_5tasks \
  --reviewers reviewer_a,reviewer_b \
  --calibration-objects 0 \
  --archive-format tar.gz \
  --dry-run \
  --overwrite
```

dry-run 重点检查 `input_validation`、`missing_references` 和两位 reviewer 的样本数量。确认无关键依赖缺失后再正式打包：

```bash
python MultiEEAffordance/tools/package_annotation_batches_from_samples.py \
  --dataset-root /home/lzq/data/MultiEEAffordance \
  --input processed/metadata/v3_candidate_samples_v0_2_5tasks.jsonl \
  --batch-dir processed/annotation_batches/v0_2_5tasks \
  --reviewers reviewer_a,reviewer_b \
  --calibration-objects 0 \
  --archive-format tar.gz \
  --overwrite
```

正式输出：

```text
/home/lzq/data/MultiEEAffordance/processed/annotation_batches/v0_2_5tasks/reviewer_a_samples.jsonl
/home/lzq/data/MultiEEAffordance/processed/annotation_batches/v0_2_5tasks/reviewer_b_samples.jsonl
/home/lzq/data/MultiEEAffordance/processed/annotation_batches/v0_2_5tasks/reviewer_a_annotation_package.tar.gz
/home/lzq/data/MultiEEAffordance/processed/annotation_batches/v0_2_5tasks/reviewer_b_annotation_package.tar.gz
/home/lzq/data/MultiEEAffordance/processed/annotation_batches/v0_2_5tasks/batch_manifest.json
```

如果维护者本人就是 `reviewer_a`，可以直接使用 `reviewer_a_samples.jsonl` 启动网页；另一位审查者使用自己的压缩包。两人的 samples 不应重叠，除非维护者显式设置了 `--calibration-objects` 做一致性校准。

### 11.0 历史说明

下面保留的旧说明仅用于追溯早期 `v0_1_5tasks` 打包方式。新批次不要再按旧脚本执行。

## 11.old 历史：划分审查样本并只打包 reviewer_b 本地数据包

当前采用 **v0.1_5tasks** 标注批次。这里的五任务文件不是重新跑 `run_v3_pipeline.py` 生成的，而是由旧三任务候选结果展开得到：

```text
pick_up    -> lift
open_pull  -> open + pull
press_push -> press + push
```

因此，当前用于人工审查划分的主输入文件是：

```text
/home/lzq/data/MultiEEAffordance/processed/metadata/v3_candidate_samples_val_5tasks_from_v0_1.jsonl
```

不要再用旧三任务文件作为正式标注批次输入：

```text
/home/lzq/data/MultiEEAffordance/processed/metadata/v3_candidate_samples_v0_1.jsonl
```

当前协作分工是：

```text
reviewer_a：维护者本人 / 有服务器数据权限，不需要本地数据压缩包。
reviewer_b：外部审查者 / 不登录服务器，需要完整本地数据压缩包。
```

所以本步骤只需要：

```text
1. 按 object 分组拆分 reviewer_a_samples.jsonl 和 reviewer_b_samples.jsonl；
2. 为 reviewer_b 收集点云、mask、candidate manifest、candidate npz 等依赖文件；
3. 只生成 reviewer_b_annotation_package.tar.gz；
4. 不生成 reviewer_a 压缩包。
```

### 11.1 数据都在哪里

候选生成阶段使用了：

```bash
--data-storage-root /home/lzq/data
```

因此，网页审查需要的大部分中间文件都在数据存储目录：

```text
/home/lzq/data/MultiEEAffordance/
```

典型位置包括：

```text
/home/lzq/data/MultiEEAffordance/processed/metadata/v3_candidate_samples_v0_1.jsonl
/home/lzq/data/MultiEEAffordance/processed/metadata/v3_candidate_samples_val_5tasks_from_v0_1.jsonl
/home/lzq/data/MultiEEAffordance/processed/vlm_candidate_v3/3d_candidates/
/home/lzq/data/MultiEEAffordance/processed/vlm_candidate_v3/fused_masks/
/home/lzq/data/MultiEEAffordance/processed/vlm_candidate_v3/pipeline_runs/
```

所以划分和打包脚本应该直接在数据存储服务器上运行。不要先把整个 `/home/lzq/data/MultiEEAffordance` 拷到本地再打包，也不要用 VSCode 文件树去解压大包；候选文件很多，VSCode 远程文件树容易卡住。

### 11.2 使用的打包脚本

当前使用脚本：

```text
MultiEEAffordance/tools/package_reviewer_b_only_annotation_batch_progress.py
```

这个脚本会完成：

```text
读取 v3_candidate_samples_val_5tasks_from_v0_1.jsonl
  -> 按 object_id / sample_id 分组
  -> 写 reviewer_a_samples.jsonl
  -> 写 reviewer_b_samples.jsonl
  -> 只扫描 reviewer_b 的样本依赖
  -> 复制 reviewer_b 需要的点云、mask、candidate manifest、candidate npz
  -> 重写数据包内部路径为相对路径
  -> 只生成 reviewer_b_annotation_package.tar.gz
  -> 写 batch_manifest.json
```

进度条含义：

```text
read xxx.jsonl                    读取输入 samples
package reviewers                 处理 reviewer_a / reviewer_b
write reviewer_x samples          写入分工后的 samples
scan sample rows                  扫描样本里的路径字段
resolve reviewer_b dependencies   递归解析候选 manifest、npz、mask 等依赖
copy reviewer_b files             复制 reviewer_b 需要的文件
archive reviewer_b_annotation_package.tar.gz  压缩打包
```

如果服务器终端显示进度条乱码，可以加：

```bash
--no-progress
```

### 11.3 先 dry-run 检查依赖

正式打包前，先 dry-run。dry-run 会划分样本并扫描依赖，但不真正复制和压缩大文件，适合先检查路径是否完整。

```bash
cd /home/lzq/Multi-EE-3DAG

python MultiEEAffordance/tools/package_reviewer_b_only_annotation_batch_progress.py   --dataset-root /home/lzq/data/MultiEEAffordance   --input processed/metadata/v3_candidate_samples_val_5tasks_from_v0_1.jsonl   --batch-dir processed/annotation_batches/v0_1_5tasks   --reviewers reviewer_a,reviewer_b   --package-reviewers reviewer_b   --calibration-objects 0   --archive-format tar.gz   --dry-run   --overwrite
```

检查输出和 `batch_manifest.json`，重点看：

```text
rows_total
object_groups_total
reviewer_a.rows
reviewer_b.rows
reviewer_b.dependencies_found
reviewer_b.missing_references
```

如果 `missing_references` 很多，说明样本索引里某些路径没有找到，不能直接发包。先排查：

```bash
cat /home/lzq/data/MultiEEAffordance/processed/annotation_batches/v0_1_5tasks/batch_manifest.json
```

### 11.4 正式划分并只打包 reviewer_b

确认 dry-run 没有明显缺失后，正式执行：

```bash
cd /home/lzq/Multi-EE-3DAG

python MultiEEAffordance/tools/package_reviewer_b_only_annotation_batch_progress.py   --dataset-root /home/lzq/data/MultiEEAffordance   --input processed/metadata/v3_candidate_samples_val_5tasks_from_v0_1.jsonl   --batch-dir processed/annotation_batches/v0_1_5tasks   --reviewers reviewer_a,reviewer_b   --package-reviewers reviewer_b   --calibration-objects 0   --archive-format tar.gz   --overwrite
```

预期输出目录：

```text
/home/lzq/data/MultiEEAffordance/processed/annotation_batches/v0_1_5tasks/
```

预期文件：

```text
reviewer_a_samples.jsonl
reviewer_b_samples.jsonl
reviewer_b_annotation_package.tar.gz
batch_manifest.json
```

注意：正常情况下不会生成：

```text
reviewer_a_annotation_package.tar.gz
```

因为 `reviewer_a` 直接使用服务器数据。

### 11.5 检查打包结果

打包完成后检查文件是否存在：

```bash
ls -lh /home/lzq/data/MultiEEAffordance/processed/annotation_batches/v0_1_5tasks/
```

检查压缩包内容，不要直接解压：

```bash
tar -tzf /home/lzq/data/MultiEEAffordance/processed/annotation_batches/v0_1_5tasks/reviewer_b_annotation_package.tar.gz | head -50
```

正常应该看到类似：

```text
README_reviewer_b.md
package_manifest_reviewer_b.json
MultiEEAffordance/processed/annotation_batches/v0_1_5tasks/reviewer_b_samples.jsonl
MultiEEAffordance/processed/points/...
MultiEEAffordance/processed/masks/...
MultiEEAffordance/processed/vlm_candidate_v3/3d_candidates/...
MultiEEAffordance/processed/vlm_candidate_v3/fused_masks/...
```

检查包大小：

```bash
du -h /home/lzq/data/MultiEEAffordance/processed/annotation_batches/v0_1_5tasks/reviewer_b_annotation_package.tar.gz
```

如果压缩包很大，不要用 VSCode 文件树打开或解压；用终端 `tar`。

## 12. 第八步：reviewer_a 在服务器上怎么标注

`reviewer_a` 有服务器数据，不需要压缩包。直接在服务器上启动网页即可。

```bash
cd /home/lzq/Multi-EE-3DAG

python MultiEEAffordance/tools/serve_v2_annotation_app.py   --dataset-root /home/lzq/data/MultiEEAffordance   --samples processed/annotation_batches/v0_1_5tasks/reviewer_a_samples.jsonl   --review-jsonl processed/annotation_batches/v0_1_5tasks/reviewer_a_review_records.jsonl   --output-mask-root processed/annotation_batches/v0_1_5tasks/manual_refined_masks_reviewer_a   --output-samples processed/annotation_batches/v0_1_5tasks/reviewer_a_refined_samples.jsonl   --port 8765   --top-k-candidates 8
```

如果在本地浏览器访问服务器网页，需要按当前服务器连接方式做端口转发或反向隧道。不要把 `reviewer_a` 的数据包再打包给自己。

## 13. 第九步：reviewer_b 本地怎么解压和标注

维护者只需要把这个压缩包发给 `reviewer_b`：

```text
/home/lzq/data/MultiEEAffordance/processed/annotation_batches/v0_1_5tasks/reviewer_b_annotation_package.tar.gz
```

`reviewer_b` 本地先准备代码仓库，例如：

```text
/path/to/Multi-EE-3DAG/
```

然后用终端解压压缩包。不要用 VSCode 文件树或系统图形界面解压大包。

```bash
tar -xzf reviewer_b_annotation_package.tar.gz -C /path/to/Multi-EE-3DAG
```

解压后应出现：

```text
/path/to/Multi-EE-3DAG/MultiEEAffordance/processed/annotation_batches/v0_1_5tasks/reviewer_b_samples.jsonl
```

然后在本地启动标注网页：

```bash
cd /path/to/Multi-EE-3DAG

python MultiEEAffordance/tools/serve_v2_annotation_app.py   --dataset-root MultiEEAffordance   --samples processed/annotation_batches/v0_1_5tasks/reviewer_b_samples.jsonl   --review-jsonl processed/annotation_batches/v0_1_5tasks/reviewer_b_review_records.jsonl   --output-mask-root processed/annotation_batches/v0_1_5tasks/manual_refined_masks_reviewer_b   --output-samples processed/annotation_batches/v0_1_5tasks/reviewer_b_refined_samples.jsonl   --port 8765   --top-k-candidates 8
```

两个人如果在不同机器上标注，都可以使用 `8765`。只有同一台机器同时开两个网页时，才需要改成不同端口，例如 `8765` 和 `8766`。

## 14. 第十步：当前数据包流程常见检查

### 14.1 为什么文件树里只有 `reviewer_samples.jsonl`

当前正式流程应该输出：

```text
reviewer_a_samples.jsonl
reviewer_b_samples.jsonl
reviewer_b_annotation_package.tar.gz
batch_manifest.json
```

如果只看到：

```text
reviewer_samples.jsonl
```

通常说明运行的不是当前 `reviewer_b only` 打包脚本，或者看到的是某个旧包 / 旧 staging 目录。请重新检查执行命令和输出目录。

### 14.2 为什么 VSCode 解压一直卡住

候选数据包里包含大量小文件，例如点云、mask、candidate manifest、candidate npz。VSCode 远程文件树展开或解压这类目录很慢，甚至看起来像卡死。

正确做法是：

```bash
tar -tzf reviewer_b_annotation_package.tar.gz | head
```

先查看包内容；确实要解压时使用：

```bash
tar -xzf reviewer_b_annotation_package.tar.gz -C /path/to/Multi-EE-3DAG
```

### 14.3 如何确认 reviewer_b 包是完整的

在服务器上检查：

```bash
cat /home/lzq/data/MultiEEAffordance/processed/annotation_batches/v0_1_5tasks/batch_manifest.json

tar -tzf /home/lzq/data/MultiEEAffordance/processed/annotation_batches/v0_1_5tasks/reviewer_b_annotation_package.tar.gz | head -50
```

重点确认：

```text
missing_references 为空或数量很少且原因明确
压缩包中有 reviewer_b_samples.jsonl
压缩包中有 processed/vlm_candidate_v3/3d_candidates/
压缩包中有 processed/vlm_candidate_v3/fused_masks/
压缩包中有 processed/points/ 或对应点云文件
压缩包中有 processed/masks/ 或对应 mask 文件
```

### 14.4 reviewer_b 不需要登录服务器

只要压缩包完整，`reviewer_b` 不需要登录服务器。其本地只需要：

```text
1. Multi-EE-3DAG 代码仓库；
2. reviewer_b_annotation_package.tar.gz；
3. 能运行 serve_v2_annotation_app.py 的 Python 环境。
```

解压数据包后，网页读取的都是本地 `MultiEEAffordance/processed/...` 下的相对路径。

## 15. 第十一步：审查者每天要交付什么

维护者应要求每天回传这些文件：

```text
processed/annotation_batches/v0_1/reviewer_x_review_records.jsonl
processed/annotation_batches/v0_1/reviewer_x_refined_samples.jsonl
processed/annotation_batches/v0_1/manual_refined_masks_reviewer_x/
```

其中：

- `review_records.jsonl`：每次点击保存都会追加一条审查记录。
- `refined_samples.jsonl`：当前已经完成审查的样本索引。
- `manual_refined_masks_reviewer_x/`：真正的人工修正后 mask。

如果只回传 JSONL、不回传 mask 目录，结果是不完整的。

## 16. 第十二步：维护者收包后检查

收到结果包后，先不要直接覆盖已有目录。建议放到：

```text
processed/annotation_batches/v0_1/returned/
```

解压到临时目录后检查：

```text
review_records 是否存在
refined_samples 是否存在
manual_refined_masks 是否存在
mask 文件数量是否大于 0
JSONL 是否每行都是合法 JSON
样本数是否和审查者当天报告一致
```

可以用下面命令快速看行数：

```bash
wc -l processed/annotation_batches/v0_1/reviewer_a_review_records.jsonl
wc -l processed/annotation_batches/v0_1/reviewer_a_refined_samples.jsonl
find processed/annotation_batches/v0_1/manual_refined_masks_reviewer_a -name "*.npy" | wc -l
```

Windows 本地检查可用 PowerShell：

```powershell
(Get-Content MultiEEAffordance\processed\annotation_batches\v0_1\reviewer_a_review_records.jsonl).Count
(Get-Content MultiEEAffordance\processed\annotation_batches\v0_1\reviewer_a_refined_samples.jsonl).Count
(Get-ChildItem MultiEEAffordance\processed\annotation_batches\v0_1\manual_refined_masks_reviewer_a -Filter *.npy -Recurse).Count
```

## 17. 第十三步：处理两人冲突

如果同一个 `sample_key` 被两个人都标了，不要自动覆盖。

冲突样本应写入：

```text
processed/annotation_batches/v0_1/merged/conflict_samples.jsonl
```

冲突处理原则：

- 如果两人都标记为空，并且理由一致，可以合并为空。
- 如果一人为空、一人有正例，需要复审。
- 如果两人都有正例但区域差异很大，需要复审。
- 如果两人差异很小，可以保留更干净、更符合标注规范的一版。

## 18. 第十四步：生成已审查数据集 release

合并后建议输出：

```text
processed/annotation_batches/v0_1/merged/merged_refined_samples.jsonl
processed/annotation_batches/v0_1/merged/reviewed_dataset_summary.json
processed/reviewed_dataset/v0_1/
```

已审查数据集至少要包含：

- 原始点云路径。
- `[N,4]` refined mask 路径。
- object category。
- task。
- executor。
- review status。
- reviewer id。
- review decision。
- 是否为空样本。
- 如果为空，空样本原因。

发布前必须抽查：

```text
pos=0 样本
正例点数异常大的样本
正例点数异常小的样本
hook 选到 blade/普通边缘的样本
suction 选到非平面/边缘的样本
open_pull 选到主体而不是把手/开关的样本
```

## 19. 每日维护者检查清单

每天建议记录一次：

```text
日期
reviewer_a 完成多少条
reviewer_b 完成多少条
今天新增 refined mask 数量
今天 pos=0 数量
今天发现的可疑样本
今天是否有网页或路径报错
是否已备份当天结果包
```

建议写到：

```text
processed/annotation_batches/v0_1/daily_logs/YYYYMMDD.md
```

## 20. 常见问题

### 20.1 已经解压 pkl 了，为什么还要用 zip

因为当前 converter 的实现就是读 zip 内部的 pkl entry。解压出来的 pkl 可以留作备用，但当前命令不要直接指向 pkl。

### 20.2 审查者网页打开后点云不显示

优先检查：

1. 数据包是否解压在仓库根目录下。
2. `--dataset-root MultiEEAffordance` 是否正确。
3. `reviewer_x_samples.jsonl` 中引用的 `.npy/.npz` 文件是否真实存在。
4. 是否只发了 samples JSONL，没有发 points/masks/candidates。

### 20.3 为什么有些样本 pos=0

这是允许的。对于某个物体、任务、执行器组合，如果没有合适区域，应该保留空样本。它能表达“这个执行器不适合这个任务或物体”。

但 `pos=0` 需要人工确认，不应该全靠自动结果。

### 20.4 自动候选很差，还能标吗

可以，但效率会下降。当前版本的定位是轻量自动候选 + 人工点级审查。自动候选只是辅助，不是最终答案。

如果某一类样本自动候选持续很差，维护者应记录下来，放到后续 `dev/high-recall-candidate-v0.2` 研发分支改进，不要在稳定标注分支里频繁大改。

### 20.5 什么东西不能提交到 Git

不要提交：

```text
raw/3d_affordancenet/full-shape.zip
raw/3d_affordancenet/full_shape_train_data.pkl
raw/3d_affordancenet/full_shape_val_data.pkl
processed/points/
processed/masks/
processed/candidates/
processed/vlm_candidate_v3/
processed/annotation_batches/
external/
*.npy
*.npz
*.pkl
*.tar.gz
```

这些都属于大文件或生成结果。GitHub 仓库只放代码、文档、少量小型配置文件。

## 21. 推荐维护节奏

第一阶段：10 条样本校准。

```text
两人各标 5 条，另外 5 条重叠校准。
```

第二阶段：50 条样本试运行。

```text
检查两人标注速度、网页稳定性、pos=0 比例、候选质量。
```

第三阶段：300 条样本正式小批。

```text
每天收包，每 50 或 100 条生成一次 summary。
```

第四阶段：扩大到更多类别和 train split。

```text
只有当前流程稳定后再扩大，不要在自动候选还频繁崩溃时直接全量跑。
```

## 22. 维护者最终要交付什么

一个完整批次结束后，维护者应能交付：

```text
1. 本批次输入样本列表
2. reviewer_a / reviewer_b 原始审查记录
3. 合并后的 refined samples
4. 合并后的 refined masks
5. conflict samples
6. reviewed dataset summary
7. 本批次运行命令和 git commit hash
8. 已知问题列表
```

这样后续无论是继续标注、训练模型，还是写论文实验部分，都能追溯这批数据是怎么来的。
