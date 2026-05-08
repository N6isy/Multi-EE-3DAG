# Qwen3-VL-8B-Instruct + SAM2 远程服务器运行说明

更新时间：2026-05-08 +08:00

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

## 4. 从 GitHub 拉取项目

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

## 5. 运行 Qwen3-VL + SAM2 pilot

先确认 pilot 渲染图已经存在：

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

运行 1 条样本做 smoke test：

```bash
CUDA_VISIBLE_DEVICES=1,2 python MultiEEAffordance/tools/run_qwen3vl_sam2_pilot.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml \
  --limit 1 \
  --overwrite
```

确认输出正常后跑完整 8 条 pilot：

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

## 6. 回投与融合

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

## 7. 回到本地复核

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

## 8. 常见问题

### Qwen3-VL 显存不足

- 先只跑 `--limit 1`。
- 使用 `CUDA_VISIBLE_DEVICES=1,2,3`。
- 把 `configs/qwen3vl_sam2_pilot.yaml` 中 `qwen3vl.device_map` 保持为 `auto`。
- 降低渲染图分辨率，例如重新生成 `--image-size 384` 或更低。

### SAM2 checkpoint 找不到

检查：

```bash
ls external/sam2/checkpoints
```

如果路径不同，修改 `configs/qwen3vl_sam2_pilot.yaml` 中的 `sam2.checkpoint`。

### 输出 JSON 解析失败

Qwen3-VL 偶尔会输出解释性文字。当前脚本会尽量从回复中提取 JSON；如果仍失败，可以降低 `max_new_tokens` 或把对应 pilot 的 prompt response 保存下来人工检查。
