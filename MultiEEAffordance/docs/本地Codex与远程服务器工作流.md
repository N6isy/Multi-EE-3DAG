# 本地 Codex 开发 + GitHub 同步 + 远程服务器运行工作流

记录时间：2026-05-08 +08:00

## 1. 重要背景

当前项目采用以下协作方式：

- 本地 Windows / VSCode / Codex：负责改代码、整理数据格式、写脚本、写文档、做轻量检查。
- GitHub 仓库：作为本地与远程服务器之间的同步通道。
- 远程服务器：负责实际运行重依赖和高算力任务，例如 Qwen3-VL、SAM2、VLM pilot、批量回投融合和较大规模数据处理。

远程服务器无法使用 Codex，因此后续所有代码和文档都必须默认支持“用户在远程服务器手动执行命令”的方式。

## 2. 对后续代码的要求

- 不把 `D:\VSCode\...`、`/home/xxx/...` 这类机器相关绝对路径写进可提交文件。
- 所有脚本必须支持 `--dataset-root`、`--config`、`--output-dir` 等命令行参数。
- 需要 GPU/大模型的脚本应提供清晰的命令行入口和环境检查报错。
- 远程服务器运行前，应能通过 GitHub 拉取最新代码后直接按文档执行。
- 大文件、模型权重、原始数据压缩包不应默认提交到 GitHub。
- manifest、metadata、config 中的资源路径优先写成相对 `MultiEEAffordance/` 的路径。

## 3. 推荐工作流

### 本地

```powershell
cd D:\VSCode\Multi-EE-3DAG

git status
git add MultiEEAffordance\tools MultiEEAffordance\docs MultiEEAffordance\configs MultiEEAffordance\requirements-vlm.txt
git commit -m "Add Qwen3-VL SAM2 pilot pipeline"
git push
```

如果需要提交已归一化的 pilot render manifest，也可以额外添加：

```powershell
git add MultiEEAffordance\processed\vlm_pilot\renders\*\view_manifest.json
```

### 远程服务器

```bash
cd /path/to/Multi-EE-3DAG
git pull

conda activate multiee_vlm

python MultiEEAffordance/tools/run_qwen3vl_sam2_pilot.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml \
  --validate-only
```

如果 `--validate-only` 报出本地 Windows 绝对路径，先运行：

```bash
python MultiEEAffordance/tools/normalize_render_manifests.py \
  --dataset-root MultiEEAffordance
```

然后再运行 smoke test：

```bash
CUDA_VISIBLE_DEVICES=1,2 python MultiEEAffordance/tools/run_qwen3vl_sam2_pilot.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml \
  --limit 1 \
  --overwrite
```

## 4. 路径约束

- 新生成的 `view_manifest.json` 必须保存相对路径，例如 `processed/vlm_pilot/renders/.../front_render.png`。
- 旧的 Windows 绝对路径可以通过 `tools/normalize_render_manifests.py` 批量改成相对路径。
- 读取脚本已经兼容旧 manifest：如果发现 `D:\...\MultiEEAffordance\processed\...`，会自动映射到当前 `--dataset-root`。

## 5. 后续设计原则

1. 本地能做语法检查和小规模 dry-run。
2. 远程能直接按文档运行真实模型。
3. 运行结果能回传或同步到本地继续审查。
4. 不依赖 Codex 在远程服务器上存在。
