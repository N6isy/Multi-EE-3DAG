# Multi-EE Affordance Dataset v0.1 进度日志

本文档用于记录数据集构建过程中的实质性进度。后续每完成一个重要阶段，都需要追加记录：完成日期时间、完成内容、产出文件、发现的问题和下一步计划。

## 2026-05-06 22:47 +08:00

### 完成内容

- 创建 `MultiEEAffordance/` 数据集原型目录结构。
- 建立原始数据、处理后数据、metadata、split 和工具脚本目录。
- 固定 v0.1 的四执行器通道顺序：
  - `gripper`
  - `suction`
  - `hook`
  - `dexterous_hand`

### 产出文件

- `taxonomy.yaml`
- `processed/metadata/samples_schema.md`
- `processed/metadata/samples.jsonl`
- `splits/train.txt`
- `splits/val.txt`
- `splits/test.txt`
- `splits/contrast_test.txt`

### 下一步

- 实现弱标签生成、可视化和数据检查脚本。

## 2026-05-06 23:10 +08:00

### 完成内容

- 实现点云、mask、metadata 的最小闭环工具。
- 将 `samples_schema.md` 改为中文说明。
- 补充 40 组强对比样本候选表。

### 产出文件

- `tools/generate_weak_masks.py`
- `tools/visualize_masks.py`
- `tools/check_dataset.py`
- `processed/metadata/contrast_samples_v0_1.jsonl`

### 下一步

- 补充真实数据接入流程和 3D AffordanceNet 转换脚本。

## 2026-05-07 10:57 +08:00

### 完成内容

- 识别 `raw/3d_affordancenet/` 下的三个压缩包：
  - `rotate.zip`
  - `full-shape.zip`
  - `partial.zip`
- 直接读取 `full-shape.zip` 中的 validation pkl，不全量解压大文件。
- 编写 3D AffordanceNet full-shape 转换脚本。
- 多轮修正弱映射规则，形成 v3 数据原型。
- v3 当前保留 46 个物体、61 条 object-task 样本。

### 产出文件

- `tools/convert_3d_affordancenet.py`
- `manifests/3d_affordancenet_full_shape_val_balanced_v3_manifest.jsonl`
- `processed/metadata/samples.jsonl`
- `processed/metadata/3d_affordancenet_full_shape_val_balanced_v3_summary.json`
- `docs/3d_affordancenet_conversion_report.md`

### 发现问题

- 3D AffordanceNet 原始 affordance 不是按执行器定义的，hook 和 suction 的弱标签需要保守处理。
- `open_pull` 不能从普通 `grasp` 直接推断。
- `pick_up/lift_carry` 不应对大件家具样本过度泛化。

### 下一步

- 做可视化检查和人工审查，找出弱标签规则错误。

## 2026-05-07 11:35 +08:00

### 完成内容

- 创建并验证 `multieeaffordance` Conda 可视化环境。
- 修正 matplotlib 导出 PNG 时的后端问题。
- 生成 HTML 和 PNG 可视化。

### 产出文件

- `tools/export_mask_html.py`
- `processed/visualizations/html_v3/index.html`
- `processed/visualizations/png_v3/door_open_pull.png`
- `processed/visualizations/png_v3/door_press_push.png`
- `processed/visualizations/png_v3/dishwasher_open_pull.png`
- `processed/visualizations/png_v3/earphone_pick_up.png`
- `docs/visualization_report.md`

### 下一步

- 建立人工审查表和网页审查工具。

## 2026-05-07 14:42 +08:00

### 完成内容

- 建立本地网页审查工具，支持在浏览器中查看单个样本并直接填写人工审查表单。
- 增加中文字段注解，说明样本级、执行器级、质量等级和备注如何填写。
- 根据需求回退掉原始 AffordanceNet candidate channel 显示，只保留 raw point cloud 与四执行器 mask 通道。

### 产出文件

- `tools/serve_review_app.py`
- `processed/metadata/manual_review_v0_1.csv`
- `docs/manual_review_schema.md`

### 下一步

- 完成 61 条样本人工审查，并根据审查结果生成 cleaned v0.1。

## 2026-05-07 19:49 +08:00

### 完成内容

- 61 条样本的人工审查已完成。
- 明确 `quality_after_review` 的判断口径：
  - `weak`：只能作为弱标签候选，仍需修正。
  - `checked`：人工看过且可作为当前版本训练/分析使用。
  - `verified`：需要更高可信度，通常要有多视角确认或精修结果支撑。
- 明确当前 suction/hook 的物理边界标准，避免把小吸盘或 hook 规则泛化到所有可接触表面。

### 产出文件

- `processed/metadata/manual_review_v0_1.csv`

### 下一步

- 将人工审查结果应用到 metadata、mask 和 splits，生成 checked/cleaned v0.1。

## 2026-05-07 20:13 +08:00

### 完成内容

- 编写并运行 `apply_manual_review.py`。
- 将 61 条人工审查结果应用到数据集中：
  - 保留 45 条样本。
  - 剔除 16 条样本。
  - 生成 66 条通道级 refine 队列。
- 对 `disable` / `not_applicable` 执行器通道做 mask 清零。
- 运行 `check_dataset.py`，确认 checked v0.1 的 45 条样本无 points/mask/metadata/split 错误。

### 产出文件

- `tools/apply_manual_review.py`
- `processed/metadata/samples_checked_v0_1.jsonl`
- `processed/masks_checked_v0_1/`
- `splits_checked_v0_1/`
- `processed/metadata/refine_queue_v0_1.csv`
- `processed/metadata/rejected_samples_v0_1.csv`
- `processed/metadata/manual_review_apply_summary_v0_1.json`

### 发现问题

- 当前 cleaned v0.1 只是样本级清洗和不可行通道清零；`refine` / `add_missing` 的逐点区域还没有真正补画。

### 下一步

- 生成人工审查总结文档，统计哪些弱规则错得最多。

## 2026-05-07 20:16 +08:00

### 完成内容

- 生成 `review_summary_v0_1.md`。
- 统计出最高频错误：
  - 漏标最多，共 38 次。
  - hook 的 `missing_positive` 最突出，共 15 次。
  - dexterous_hand 的 `under_label` 较多，共 8 次。
  - suction 同时存在 `missing_positive` 与 `over_label`，说明平面面积、边缘、曲率规则不足。

### 产出文件

- `docs/review_summary_v0_1.md`

### 下一步

- 设计 VLM 多视角辅助精修 pilot，先验证小样本流程，不全量跑。

## 2026-05-07 20:18 +08:00

### 完成内容

- 编写 VLM 多视角小试验设计文档。
- 编写三段脚本骨架：
  - 多视角渲染
  - 2D mask 回投 3D
  - 多视角投票融合
- 固定 VLM 提示词模板和四执行器定义。
- 修正投影脚本的点数推断逻辑：优先读取 `view_manifest.json` 中的 `num_points`，避免遮挡导致点数被低估。

### 产出文件

- `docs/vlm_multiview_pilot_design.md`
- `tools/render_multiview.py`
- `tools/project_2d_masks_to_3d.py`
- `tools/fuse_multiview_masks.py`
- `vlm_prompt_templates.yaml`

### 下一步

- 从 refine 队列中挑选 5 到 10 条 pilot 样本，并生成多视角渲染。

## 2026-05-07 20:20 +08:00

### 完成内容

- 从 `refine_queue_v0_1.csv` 中挑选 8 条通道级 VLM pilot 样本。
- 覆盖执行器：
  - suction：2 条
  - hook：3 条
  - gripper：1 条
  - dexterous_hand：2 条
- 覆盖问题类型：
  - `over_label`
  - `needs_geometry_rule`
  - `wrong_region`
  - `missing_positive`
  - `under_label`
- 为 7 个唯一 object-task 样本生成 6 视角渲染，共 42 张 PNG，并保存 point-index/depth map。
- 修正 `render_multiview.py`：当 matplotlib 不可用时，使用标准库 PNG fallback，保证仍可导出图像。

### 产出文件

- `processed/metadata/vlm_pilot_samples_v0_1.csv`
- `processed/vlm_pilot/renders/*/view_manifest.json`
- `processed/vlm_pilot/renders/*/*_render.png`
- `processed/vlm_pilot/renders/*/*_point_index.npy`
- `processed/vlm_pilot/renders/*/*_depth.npy`

### 下一步

- 对 pilot 渲染图运行或手动模拟 VLM 2D 分割，得到每个视角的二值 mask。
- 用 `project_2d_masks_to_3d.py` 回投到点云。
- 用 `fuse_multiview_masks.py` 生成候选 3D mask。
- 将候选 mask 放回网页可视化工具中做人工复核。

## 2026-05-07 20:51 +08:00

### 完成内容

- 新增 OpenAI VLM pilot 调用脚本，用于把 6 视角渲染图转换为 per-view 2D mask。
- 新增批量候选构建脚本，用于把 VLM 2D mask 回投到 3D 点云、融合目标执行器通道，并生成网页复核用 samples JSONL。
- 新增 VLM pilot 运行说明文档，明确 VSCode 终端中的执行命令和输出路径。
- 尝试在 Codex 当前进程中调用 OpenAI API 进行 1 条 pilot 验证，但外部网络审批连续超时；为避免污染数据，没有生成伪 VLM mask。

### 产出文件

- `tools/run_openai_vlm_pilot.py`
- `tools/build_vlm_pilot_candidates.py`
- `docs/vlm_pilot_runbook.md`
- 更新 `docs/vlm_multiview_pilot_design.md`

### 发现问题

- 当前 Codex shell 不等同于用户 VSCode 中激活的 `multieeaffordance` 环境。
- Codex 进程检测到 OpenAI SDK 和 `OPENAI_API_KEY` 变量名，但网络调用审批未通过，因此真实 VLM 调用需要用户在 VSCode 终端中执行。

### 下一步

- 用户在 VSCode 终端运行 `run_openai_vlm_pilot.py` 生成真实 VLM 2D masks。
- 运行 `build_vlm_pilot_candidates.py` 生成候选 3D masks。
- 用 `serve_review_app.py` 加载 `vlm_pilot_candidate_samples_v0_1.jsonl`，对候选 mask 进行网页复核。

## 2026-05-08 +08:00

### 完成内容

- 明确项目后续采用“本地 Codex 改代码 + GitHub 同步 + 远程服务器实际运行”的工作流。
- 记录远程服务器无法使用 Codex，因此所有重模型运行脚本必须能由用户在远程服务器手动执行。
- 将该约束写入独立工作流文档，作为后续开发和部署的长期约束。

### 产出文件

- `docs/local_codex_remote_server_workflow.md`

### 下一步

- 后续补 Qwen3-VL + SAM2 pipeline 时，优先写远程服务器可执行的命令、配置文件和环境检查逻辑。
- 本地只做代码修改、轻量 dry-run、文档维护和网页审查，不默认承担大模型实际推理。

## 2026-05-08 10:41 +08:00

### 完成内容

- 根据远程服务器 4 张 24GB GPU 的资源情况，确定 Qwen3-VL-8B-Instruct + SAM2 的 pilot 运行策略。
- 新增远程服务器可执行的 `Qwen3-VL + SAM2` 脚本：
  - Qwen3-VL 逐视角读取渲染图。
  - Qwen3-VL 输出 box / positive point / negative point。
  - SAM2 根据这些 prompt 生成每个视角的 2D mask。
  - 输出目录兼容已有 `build_vlm_pilot_candidates.py`。
- 新增远程运行配置，默认使用 `Qwen/Qwen3-VL-8B-Instruct` 和 SAM2.1 large。
- 新增远程安装与运行说明文档，明确本地 GitHub 同步、远程 pull、远程运行、结果回传复核的流程。
- 修正 `local_codex_remote_server_workflow.md` 中旧的配置文件名。
- 增加 `--validate-only` 模式，用于远程服务器在不加载 Qwen3-VL/SAM2 的情况下检查配置、pilot 表和多视角渲染文件。

### 产出文件

- `configs/qwen3vl_sam2_pilot.yaml`
- `tools/run_qwen3vl_sam2_pilot.py`
- `docs/qwen3vl_sam2_remote_setup.md`
- `requirements-vlm.txt`
- 更新 `docs/vlm_multiview_pilot_design.md`
- 更新 `docs/local_codex_remote_server_workflow.md`

### 本地轻量检查

- `python -m py_compile MultiEEAffordance/tools/run_qwen3vl_sam2_pilot.py` 通过。
- `--validate-only --limit 1` 通过，已检查 1 条 pilot 的 6 个视角渲染文件。

### 远程运行建议

- 优先使用 `CUDA_VISIBLE_DEVICES=1,2`，避免占用截图中已有 Python 进程的 GPU0。
- 先运行 `--limit 1 --overwrite` 做 smoke test，再跑完整 8 条 pilot。
- 如果 Qwen3-VL 显存不足，再扩展到 `CUDA_VISIBLE_DEVICES=1,2,3` 或降低渲染图分辨率。

### 下一步

- 将本地代码提交并推送到 GitHub。
- 在远程服务器 `git pull` 后按 `docs/qwen3vl_sam2_remote_setup.md` 安装环境并运行 1 条 pilot smoke test。
- smoke test 成功后运行完整 8 条 pilot，并回投融合生成候选 3D mask。

## 2026-05-08 路径兼容修复

### 完成内容

- 修复远程服务器运行 `run_qwen3vl_sam2_pilot.py --validate-only` 时读取到本地 Windows 绝对路径的问题。
- 新增通用路径工具 `path_utils.py`，支持相对路径、当前系统绝对路径和旧 manifest 中的 Windows 绝对路径。
- 修改 `render_multiview.py`，后续新生成的 `view_manifest.json` 默认写相对路径。
- 修改 Qwen3-VL+SAM2、OpenAI VLM、2D 回投、候选融合脚本，使其读取旧 manifest 时自动做跨机器路径映射。
- 新增 `normalize_render_manifests.py`，可批量把已有 render manifest 改成相对路径。
- 已将当前 7 个 pilot render manifest 从 `D:\VSCode\...` 改成 `processed/...` 相对路径。

### 产出文件

- `tools/path_utils.py`
- `tools/normalize_render_manifests.py`
- 更新 `tools/render_multiview.py`
- 更新 `tools/run_qwen3vl_sam2_pilot.py`
- 更新 `tools/run_openai_vlm_pilot.py`
- 更新 `tools/project_2d_masks_to_3d.py`
- 更新 `tools/build_vlm_pilot_candidates.py`
- 更新 `docs/qwen3vl_sam2_remote_setup.md`
- 重写 `docs/local_codex_remote_server_workflow.md`，明确路径约束和 GitHub/远程运行流程。

### 本地检查

- `run_qwen3vl_sam2_pilot.py --validate-only --limit 1` 通过。
- `normalize_render_manifests.py --dry-run` 显示当前 7 个 manifest 已无待修改路径。

### 下一步

- 本地将修复提交并推送到 GitHub。
- 远程服务器 `git pull` 后重新运行 `--validate-only`。
