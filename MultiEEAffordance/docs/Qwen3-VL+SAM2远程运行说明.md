# Qwen3-VL-8B-Instruct + SAM2 远程服务器运行说明

更新时间：2026-05-14 +08:00

本文档对应当前项目的长期工作流：本地使用 Codex 修改代码，经 GitHub 同步到远程服务器；远程服务器不能使用 Codex，只负责按文档手动运行真实大模型推理。

## 1. 服务器资源判断

当前远程服务器有 4 张 24GB 显存 GPU。建议不要使用 GPU0，因为截图中 GPU0 已有一个 Python 进程占用约 1908 MiB。pilot 阶段建议使用：

```bash
CUDA_VISIBLE_DEVICES=1,2
```

这样脚本内部看到的 `cuda:0` 实际对应物理 GPU1，`cuda:1` 实际对应物理 GPU2。

## 2. 推荐环境

```bash
conda create -n multiee_vlm python=3.11 -y
conda activate multiee_vlm

python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r MultiEEAffordance/requirements-vlm.txt
```

说明：

- Qwen3-VL 官方模型卡建议使用最新源码版 `transformers`。
- 服务器驱动显示 CUDA Version 12.0，通常可以运行 PyTorch cu121 wheel；如果远程报 CUDA/driver 不兼容，再改为服务器管理员推荐的 PyTorch 版本。

## 3. 安装 SAM2

```bash
cd /path/to/Multi-EE-3DAG
mkdir -p external
cd external
git clone https://github.com/facebookresearch/sam2.git
cd sam2
python -m pip install -e .
```

下载 SAM2.1 checkpoints：

```bash
cd checkpoints
./download_ckpts.sh
cd /path/to/Multi-EE-3DAG
```

当前默认配置使用：

```text
external/sam2/checkpoints/sam2.1_hiera_large.pt
configs/sam2.1/sam2.1_hiera_l.yaml
```

如果显存或速度有压力，可以把配置改成 `sam2.1_hiera_small.pt` 和 `sam2.1_hiera_s.yaml`。

## 4. 服务器无法访问 Hugging Face 时的模型准备

如果运行时报：

```text
[Errno 101] Network is unreachable
Can't load the configuration of 'Qwen/Qwen3-VL-8B-Instruct'
```

说明服务器当前不能联网访问 Hugging Face，脚本无法自动下载 Qwen3-VL。此时需要在一台能联网的机器上先下载模型，再拷贝到服务器。

在可联网机器上下载：

```bash
python -m pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct \
  --local-dir Qwen3-VL-8B-Instruct
```

把整个 `Qwen3-VL-8B-Instruct/` 目录传到服务器项目下，例如：

```text
/mnt/ssd/lzq/Multi-EE-3DAG/MultiEEAffordance/external/models/Qwen3-VL-8B-Instruct/
```

该目录下必须能看到：

```text
config.json
model*.safetensors
tokenizer / processor 相关文件
```

然后修改 `MultiEEAffordance/configs/qwen3vl_sam2_pilot.yaml`：

```yaml
qwen3vl:
  model_id: Qwen/Qwen3-VL-8B-Instruct
  model_path: external/models/Qwen3-VL-8B-Instruct
  local_files_only: true
```

也可以用服务器上的绝对路径：

```yaml
qwen3vl:
  model_path: /mnt/ssd/lzq/models/Qwen3-VL-8B-Instruct
  local_files_only: true
```

可选：运行前设置离线环境变量，避免 `transformers` 尝试联网：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

注意：`model_path` 可以是相对路径；相对路径会基于 `--dataset-root MultiEEAffordance` 解析。

## 5. 从 GitHub 拉取项目

远程服务器上：

```bash
cd /path/to
git clone https://github.com/<你的用户名>/Multi-EE-3DAG.git
cd Multi-EE-3DAG
```

之后每次本地 Codex 改完并 push 后，远程执行：

```bash
cd /path/to/Multi-EE-3DAG
git pull
```

注意：raw 数据、点云、mask、模型权重不建议提交 GitHub。远程服务器需要另外准备这些大文件，保持目录结构与本地一致。

## 6. 运行 Qwen3-VL + SAM2 pilot

先重新生成 VLM 友好的 yaw 视角渲染。当前推荐先只验证 `vlm_pilot_005`，也就是 `Bag / lift_carry / hook`，因为 bag handle 是 hook 的典型正例。

```bash
python MultiEEAffordance/tools/render_vlm_pilot_views.py \
  --dataset-root MultiEEAffordance \
  --pilot-id vlm_pilot_005 \
  --image-size 768 \
  --point-size 1 \
  --visual-point-size 4 \
  --overwrite
```

这会生成 8 个 yaw 视角：

```text
yaw000_elev20
yaw045_elev20
yaw090_elev20
yaw135_elev20
yaw180_elev20
yaw225_elev20
yaw270_elev20
yaw315_elev20
```

确认 pilot 渲染图已经存在：

```bash
ls MultiEEAffordance/processed/vlm_pilot/renders
```

不加载模型，只检查配置、pilot 表和渲染图路径：

```bash
python MultiEEAffordance/tools/run_qwen3vl_sam2_pilot.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml \
  --validate-only
```

如果这里报出类似 `D:\VSCode\...front_render.png` 的路径错误，说明旧 `view_manifest.json` 仍然保存了本地 Windows 绝对路径。先运行：

```bash
python MultiEEAffordance/tools/normalize_render_manifests.py \
  --dataset-root MultiEEAffordance
```

再重新运行 `--validate-only`。新版本的 `render_multiview.py` 会默认写入相对路径，读取脚本也会自动兼容旧 Windows 绝对路径。

先运行 `vlm_pilot_005` 做 smoke test：

```bash
CUDA_VISIBLE_DEVICES=1,2 python MultiEEAffordance/tools/run_qwen3vl_sam2_pilot.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml \
  --pilot-id vlm_pilot_005 \
  --overwrite
```

检查是否产生非空 2D mask：

```bash
python - <<'PY'
import json
p = "MultiEEAffordance/processed/vlm_pilot/qwen3vl_sam2_responses/run_summary.json"
d = json.load(open(p, "r", encoding="utf-8"))
for row in d["rows"]:
    print(row["pilot_id"], row["sample_id"], row["executor"])
    for view in row["views"]:
        print(" ", view["view"], "positive_pixels=", view["positive_pixels"], "feasible=", view["feasible"])
PY
```

同时查看 prompt overlay，确认 Qwen3-VL 给出的 box/point 是否落在合理位置：

```bash
find MultiEEAffordance/processed/vlm_pilot/qwen3vl_sam2_responses/vlm_pilot_005 \
  -name "*_prompt_overlay.png"
```

如果 `vlm_pilot_005` 能产生非空 mask，再生成完整 8 条 pilot 的新视角渲染：

```bash
python MultiEEAffordance/tools/render_vlm_pilot_views.py \
  --dataset-root MultiEEAffordance \
  --image-size 768 \
  --point-size 1 \
  --visual-point-size 4 \
  --overwrite
```

然后跑完整 8 条 pilot：

```bash
CUDA_VISIBLE_DEVICES=1,2 python MultiEEAffordance/tools/run_qwen3vl_sam2_pilot.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml \
  --overwrite
```

输出位置：

- `processed/vlm_pilot/qwen3vl_sam2_responses/`
- `processed/vlm_pilot/vlm_2d_masks/{sample_id}/{executor}/{view}.npy`
- `processed/vlm_pilot/vlm_2d_masks/{sample_id}/{executor}/{view}.png`

## 7. 回投与融合

如果只验证 `vlm_pilot_005`：

```bash
python MultiEEAffordance/tools/build_vlm_pilot_candidates.py \
  --dataset-root MultiEEAffordance \
  --pilot-id vlm_pilot_005 \
  --score-threshold 0.45 \
  --min-visible 1 \
  --overwrite
```

如果完整 8 条都已经生成了非空候选，再全量融合：

```bash
python MultiEEAffordance/tools/build_vlm_pilot_candidates.py \
  --dataset-root MultiEEAffordance \
  --score-threshold 0.45 \
  --min-visible 1 \
  --overwrite
```

输出位置：

- `processed/vlm_pilot/projected/`
- `processed/vlm_pilot/fused_masks/`
- `processed/metadata/vlm_pilot_candidate_samples_v0_1.jsonl`
- `processed/metadata/vlm_pilot_candidate_summary_v0_1.json`

## 8. 回到本地复核

远程生成候选结果后，把以下文件/目录同步回本地：

- `processed/vlm_pilot/vlm_2d_masks/`
- `processed/vlm_pilot/qwen3vl_sam2_responses/`
- `processed/vlm_pilot/projected/`
- `processed/vlm_pilot/fused_masks/`
- `processed/metadata/vlm_pilot_candidate_samples_v0_1.jsonl`
- `processed/metadata/vlm_pilot_candidate_summary_v0_1.json`

本地启动网页复核：

```powershell
python MultiEEAffordance\tools\serve_review_app.py `
  --dataset-root MultiEEAffordance `
  --samples processed\metadata\vlm_pilot_candidate_samples_v0_1.jsonl `
  --review-csv processed\metadata\vlm_pilot_review_v0_1.csv `
  --host 127.0.0.1 `
  --port 8766 `
  --max-points 4096
```

浏览器打开：

```text
http://127.0.0.1:8766/
```

## 9. hook 几何候选路线

如果 hook 通道在源数据集中没有现成 mask，尤其是 `Bag / lift_carry / hook` 这类样本，不建议继续让 Qwen3-VL 直接输出像素坐标。当前新增一条更稳的路线：

```text
多视角 point-index map
  -> 几何规则生成 3D hook 候选 A/B/C
  -> 渲染候选 overlay
  -> Qwen3-VL 只选择候选标签，不输出坐标
  -> 选中候选写回 [N,4] mask 的 hook 通道
  -> 本地网页人工复核
```

先以 `vlm_pilot_005` 为单样本闭环：

```bash
python MultiEEAffordance/tools/render_vlm_pilot_views.py \
  --dataset-root MultiEEAffordance \
  --pilot-id vlm_pilot_005 \
  --image-size 768 \
  --point-size 1 \
  --visual-point-size 4 \
  --overwrite
```

生成 hook 几何候选：

```bash
python MultiEEAffordance/tools/generate_hook_candidates.py \
  --dataset-root MultiEEAffordance \
  --pilot-id vlm_pilot_005 \
  --overwrite
```

渲染带候选标签的 overlay 图：

```bash
python MultiEEAffordance/tools/render_hook_candidate_overlay.py \
  --dataset-root MultiEEAffordance \
  --pilot-id vlm_pilot_005 \
  --overwrite
```

运行 Qwen3-VL 候选选择。注意这里不需要 SAM2，因为 Qwen 只需要从 A/B/C 中选择：

```bash
CUDA_VISIBLE_DEVICES=1,2 python MultiEEAffordance/tools/run_qwen3vl_candidate_selector.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml \
  --pilot-id vlm_pilot_005 \
  --overwrite
```

把选中的候选写回四通道 mask：

```bash
python MultiEEAffordance/tools/build_hook_candidate_mask.py \
  --dataset-root MultiEEAffordance \
  --pilot-id vlm_pilot_005 \
  --overwrite
```

如果只是调试候选是否合理，可以手动指定候选，例如先把 A 写回 hook 通道：

```bash
python MultiEEAffordance/tools/build_hook_candidate_mask.py \
  --dataset-root MultiEEAffordance \
  --pilot-id vlm_pilot_005 \
  --selected-candidates A \
  --overwrite
```

检查候选样本格式：

```bash
python MultiEEAffordance/tools/check_dataset.py \
  --dataset-root MultiEEAffordance \
  --samples processed/metadata/hook_candidate_samples_v0_1.jsonl \
  --split-dir splits_hook_candidates
```

本地复核该候选样本：

```powershell
python MultiEEAffordance\tools\serve_review_app.py `
  --dataset-root MultiEEAffordance `
  --samples processed\metadata\hook_candidate_samples_v0_1.jsonl `
  --review-csv processed\metadata\hook_candidate_review_v0_1.csv `
  --host 127.0.0.1 `
  --port 8767 `
  --max-points 4096
```

这条路线的定位是：几何规则负责提出候选，Qwen3-VL 负责候选级语义筛选，人工审查负责最终确认。候选 mask 不能直接当作 ground truth。

## 10. 常见问题

### Qwen3-VL 显存不足

- 先只跑 `--limit 1`。
- 使用 `CUDA_VISIBLE_DEVICES=1,2,3`。
- 把 `configs/qwen3vl_sam2_pilot.yaml` 中 `qwen3vl.device_map` 保持为 `auto`。
- 降低渲染图分辨率，例如重新生成 `--image-size 384` 或更低。

### SAM2 checkpoint 找不到

检查：

```bash
cd ~/Multi-EE-3DAG
ls MultiEEAffordance/external/sam2/checkpoints/sam2.1_hiera_large.pt
```

`sam2.checkpoint` 支持相对路径和绝对路径。相对路径会基于 `--dataset-root MultiEEAffordance` 解析，所以默认配置：

```yaml
sam2:
  checkpoint: external/sam2/checkpoints/sam2.1_hiera_large.pt
```

实际对应：

```text
~/Multi-EE-3DAG/MultiEEAffordance/external/sam2/checkpoints/sam2.1_hiera_large.pt
```

如果你还没拉取支持该解析逻辑的新代码，可以临时把配置改成绝对路径：

```yaml
sam2:
  checkpoint: /home/lzq/Multi-EE-3DAG/MultiEEAffordance/external/sam2/checkpoints/sam2.1_hiera_large.pt
```

### 输出 JSON 解析失败

Qwen3-VL 偶尔会输出解释性文字。当前脚本会尽量从回复中提取 JSON；如果仍失败，可以降低 `max_new_tokens` 或把对应 pilot 的 prompt response 保存下来人工检查。
