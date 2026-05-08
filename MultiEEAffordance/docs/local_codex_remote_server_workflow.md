# 本地 Codex 开发 + 远程服务器运行工作流

记录时间：2026-05-08 +08:00

## 1. 重要背景

当前项目采用以下协作方式：

- 本地 Windows / VSCode / Codex：负责改代码、整理数据格式、写脚本、写文档、做轻量检查。
- GitHub 仓库：作为本地与远程服务器之间的同步通道。
- 远程服务器：负责实际运行重依赖和高算力任务，例如 Qwen3-VL、SAM2、VLM pilot、批量回投融合、大规模数据处理。

远程服务器无法使用 Codex，因此后续所有代码和文档都必须默认支持“用户在远程服务器手动执行命令”的方式。

## 2. 对后续代码的要求

- 不把 `D:\VSCode\...` 这类本地绝对路径写死进脚本。
- 所有脚本必须支持 `--dataset-root`、`--config`、`--output-dir` 等命令行参数。
- 需要 GPU/大模型的脚本应提供清晰的命令行入口和环境检查报错。
- 远程服务器运行前，应能通过 GitHub 拉取最新代码后直接按文档执行。
- 大文件、模型权重、原始数据压缩包不应默认提交到 GitHub。
- 输出目录应清晰区分：
  - 本地轻量中间产物
  - 远程服务器生成结果
  - 人工审查结果
  - 可提交的 metadata/config/docs

## 3. 推荐工作流

### 本地

```powershell
cd D:\VSCode\Multi-EE-3DAG

# Codex 修改代码、文档、配置

git status
git add MultiEEAffordance\tools MultiEEAffordance\docs MultiEEAffordance\*.yaml
git commit -m "Add VLM pilot pipeline"
git push
```

### 远程服务器

```bash
cd /path/to/Multi-EE-3DAG
git pull

conda activate multiee_vlm

python MultiEEAffordance/tools/run_qwen3vl_sam2_pilot.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml
```

## 4. 后续设计原则

后续如果要新增 Qwen3-VL + SAM2、批量渲染、批量回投、网页复核或数据清洗脚本，都应优先满足：

1. 本地能做语法检查和小规模 dry-run。
2. 远程能直接按文档运行真实模型。
3. 运行结果能回传或同步到本地继续审查。
4. 不依赖 Codex 在远程服务器上存在。

## 5. 当前下一步

- 本地继续补齐 `Qwen3-VL + SAM2` 脚本、配置和远程运行文档。
- 用户将代码推送到 GitHub。
- 远程服务器拉取仓库后运行真实 VLM/SAM2 pipeline。
- 远程产出的候选 3D mask 再回到本地网页审查工具中复核。
