# VLM Pilot 运行说明

更新时间：2026-05-07 20:51 +08:00

本说明用于把 8 条 pilot 样本从“多视角图像”推进到“候选 3D mask + 网页复核”。

## 1. 当前状态

- 已有 8 条通道级 pilot 样本：`processed/metadata/vlm_pilot_samples_v0_1.csv`
- 已有 7 个唯一 object-task 样本的多视角渲染：`processed/vlm_pilot/renders/`
- 已有 OpenAI VLM 调用脚本：`tools/run_openai_vlm_pilot.py`
- 已有批量回投/融合脚本：`tools/build_vlm_pilot_candidates.py`

注意：VLM 输出只能作为候选区域，不能直接作为最终 verified 标签。

## 2. 运行 VLM 生成 2D mask

在 VSCode PowerShell 中执行：

```powershell
cd D:\VSCode\Multi-EE-3DAG
conda activate multieeaffordance

python MultiEEAffordance\tools\run_openai_vlm_pilot.py `
  --dataset-root MultiEEAffordance `
  --model gpt-4o-mini `
  --overwrite
```

如果你的 OpenAI key 没有在当前终端环境中生效，先设置：

```powershell
$env:OPENAI_API_KEY="你的 API key"
```

VLM 输出位置：

- `processed/vlm_pilot/vlm_responses/{pilot_id}/response.json`
- `processed/vlm_pilot/vlm_2d_masks/{sample_id}/{executor}/{view}.npy`
- `processed/vlm_pilot/vlm_2d_masks/{sample_id}/{executor}/{view}.png`

## 3. 回投并融合成候选 3D mask

VLM 2D mask 生成后执行：

```powershell
python MultiEEAffordance\tools\build_vlm_pilot_candidates.py `
  --dataset-root MultiEEAffordance `
  --score-threshold 0.45 `
  --min-visible 1 `
  --overwrite
```

输出位置：

- `processed/vlm_pilot/projected/`
- `processed/vlm_pilot/fused_masks/`
- `processed/metadata/vlm_pilot_candidate_samples_v0_1.jsonl`
- `processed/metadata/vlm_pilot_candidate_summary_v0_1.json`
- `splits_vlm_pilot_candidates/`

## 4. 启动网页复核

建议使用单独端口，避免覆盖当前 61 条人工审查页面：

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

## 5. 复核时重点看什么

- VLM 是否把普通接触面误标为正样本。
- hook 是否真的标到了孔洞、环、把手内侧或可挂接边界。
- suction 是否避开边缘、把手和高曲率区域。
- dexterous_hand 是否只标稳定包覆/按压/精细操作区域，而不是整片物体表面。
- gripper 是否标到可两侧夹持的区域，而不是不可稳定夹取的位置。

## 6. 本轮 Codex 运行说明

Codex 当前进程检测到了 `OPENAI_API_KEY` 变量名和 OpenAI SDK，但外部网络调用审批连续超时，因此没有在 Codex 内部完成真实 VLM API 调用。为避免污染数据，没有生成伪 VLM mask。请在你的 VSCode 终端中执行第 2 节命令继续真实 VLM 步骤。
