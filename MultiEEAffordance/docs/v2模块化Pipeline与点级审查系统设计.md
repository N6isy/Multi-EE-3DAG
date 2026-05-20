# v2 模块化 Pipeline 与点级人工审查系统设计

更新时间：2026-05-19 +08:00

## 1. 目标

当前项目已经从“单脚本生成候选”进入“候选生成 + VLM 选择 + 规则过滤 + 人工精修”的阶段。下一步目标是把流程模块化，并提供一个可以多人协作的网页审查系统。

系统目标不是让 VLM 自动生成最终真值，而是：

```text
自动模块提出候选 -> VLM/规则筛选 -> 人工在网页中快速确认和点级精修 -> 输出 checked/refined mask
```

## 2. 模块化流程

| 阶段 | 脚本 | 输入 | 输出 | 角色 |
| --- | --- | --- | --- | --- |
| 3D 候选生成 | `generate_3d_candidate_regions.py` | 点云、已有弱标签、多视角 point-index map | `candidate_manifest.json`、`candidate_masks.npz` | 高召回提出 A/B/C... 候选 |
| 候选可视化 | `render_candidate_overlays_v2.py` | 候选区域、多视角渲染 | overlay / selector panel | 给 VLM 选择使用 |
| VLM 候选选择 | `run_vlm_candidate_selection_v2.py` | overlay、语义部件计划、候选列表 | `combined_selection.json` | VLM 从候选编号中投票选择 |
| 规则过滤 | `filter_candidates_by_executor_rules.py` | VLM selection、候选几何属性 | `rule_filter.json` | 保守过滤，不允许未被 VLM 支持的候选自动进正例 |
| 候选 mask 构建 | `build_v2_candidate_masks.py` | rule filter 或人工指定候选 | `v2_candidate_samples_v0_1.jsonl`、候选 `[N,4]` mask | 构建待审查候选样本 |
| 候选审查可视化 | `visualize_v2_candidates.py` | 候选区域和选中候选 | `index.html`、单候选图、已选候选图 | 辅助人工判断候选质量 |
| 点级人工精修 | `serve_v2_annotation_app.py` | 待审查样本、候选 mask | refined mask、审查记录 JSONL | 人工增删点并保存新 mask |

## 3. 一键式 Pipeline 管理

新增脚本：

```text
tools/run_v2_pipeline.py
```

它把前面的模块串起来，方便在远程服务器复现实验。

示例：跑完整 v2 流程。

```bash
python MultiEEAffordance/tools/run_v2_pipeline.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml \
  --pilot-id vlm_pilot_005 \
  --min-selected-votes 8 \
  --selected-candidates A \
  --overwrite
```

示例：只跑候选生成、overlay 和 VLM 选择。

```bash
python MultiEEAffordance/tools/run_v2_pipeline.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml \
  --pilot-id vlm_pilot_005 \
  --stages generate,render,select \
  --overwrite
```

每次运行会写出：

```text
processed/vlm_candidate_v2/pipeline_runs/latest_run_manifest.json
```

用于记录各阶段命令、返回码和运行状态。

## 4. 点级人工审查网页 MVP

新增脚本：

```text
tools/serve_v2_annotation_app.py
```

它读取：

```text
processed/metadata/v2_candidate_samples_v0_1.jsonl
```

并在浏览器中显示候选 mask 对应的目标执行器通道。系统同时会读取该样本的 `candidate_manifest` 和 `rule_filter`，把 VLM/规则筛选后的 top-k 候选显示成可勾选菜单。审查者可以：

- 查看点云和目标通道正样本；
- 查看 top-k 候选区域的名称、类型、VLM 票数、规则分数和状态；
- 勾选一个或多个候选区域，例如 `A`、`A+E` 或 `A+E+I`；
- 点击“应用勾选候选”，将候选组合合并成当前待编辑 mask；
- 旋转、缩放点云；
- 切换编辑模式：查看、添加、删除、切换；
- 点击点云中的点，删除误标点或补充漏标点；
- 填写审查人、审查状态、质量等级和备注；
- 保存新的 refined mask。

运行：

```bash
python MultiEEAffordance/tools/serve_v2_annotation_app.py \
  --dataset-root MultiEEAffordance \
  --samples processed/metadata/v2_candidate_samples_v0_1.jsonl \
  --host 0.0.0.0 \
  --port 8770 \
  --top-k-candidates 8
```

本地访问：

```text
http://127.0.0.1:8770
```

如果是在远程服务器上运行，可通过浏览器访问：

```text
http://服务器IP:8770
```

前提是服务器安全组/防火墙允许该端口访问。

## 5. 点级审查输出

保存后会生成：

| 输出 | 默认路径 | 含义 |
| --- | --- | --- |
| refined mask | `processed/vlm_candidate_v2/manual_refined_masks/` | 人工增删点后的 `[N,4]` mask |
| refined samples | `processed/metadata/v2_manual_refined_samples_v0_1.jsonl` | 指向 refined mask 的样本 metadata |
| review records | `processed/metadata/v2_point_level_review_records.jsonl` | 每次人工保存的审查日志 |

审查日志会记录：

- sample_id；
- executor；
- reviewer；
- selected_candidate_ids；
- review_status；
- review_decision；
- positive_points_before / after；
- added_points；
- removed_points；
- notes；
- output_mask_path。

这样后续可以追溯每个本科生审查者做了哪些修改。

## 6. 多人协作上线建议

当前 `serve_v2_annotation_app.py` 是 MVP，适合局域网、实验室服务器或小规模本科生协作。真正公开上线前，需要升级为生产系统。

### 6.1 不建议直接公网裸奔

不要把无登录、无权限控制的标注系统直接暴露到公网。原因：

- 任意人都能修改 mask；
- 无法追踪审查者身份；
- 数据可能被恶意覆盖；
- 服务器端文件路径和数据资产可能泄露。

### 6.2 推荐部署方式

| 层级 | 建议 |
| --- | --- |
| 访问方式 | 优先校园网 / VPN / 实验室内网；如果公网访问，需要域名 + HTTPS |
| 反向代理 | Nginx |
| 进程管理 | systemd / supervisor / tmux 临时运行 |
| 用户身份 | 最低限度使用用户名字段；正式版应加登录认证 |
| 数据写入 | 每次保存都写 append-only 审查日志，避免覆盖不可追溯 |
| 任务分配 | 按 sample_id 分配给审查者，避免多人同时编辑同一条样本 |
| 备份 | 每日备份 refined masks 和 review records |

### 6.3 后续生产版应增加

- 登录账号与角色：管理员、审查者、复核者；
- 样本领取/锁定机制；
- 双人复核和冲突解决；
- 点级编辑历史回放；
- 按类别/任务/执行器统计审查进度；
- WebGL/Three.js 更高质量点云交互；
- 数据库后端：SQLite 起步，后续可迁移 PostgreSQL；
- 导出 checked / verified 数据集版本。

## 7. 当前阶段推荐工作流

以 `vlm_pilot_005` 为例：

```bash
# 1. 跑 v2 自动候选流程
python MultiEEAffordance/tools/run_v2_pipeline.py \
  --dataset-root MultiEEAffordance \
  --config configs/qwen3vl_sam2_pilot.yaml \
  --pilot-id vlm_pilot_005 \
  --stages generate,render,select,filter \
  --min-selected-votes 8 \
  --overwrite

# 2. 人工保守选择候选 A 生成待审查 mask
python MultiEEAffordance/tools/build_v2_candidate_masks.py \
  --dataset-root MultiEEAffordance \
  --pilot-id vlm_pilot_005 \
  --selected-candidates A \
  --overwrite

# 3. 启动点级审查系统
python MultiEEAffordance/tools/serve_v2_annotation_app.py \
  --dataset-root MultiEEAffordance \
  --samples processed/metadata/v2_candidate_samples_v0_1.jsonl \
  --host 0.0.0.0 \
  --port 8770
```

人工审查时：

1. 删除 A 中落到无关区域的 false positive 点；
2. 如果 A 覆盖不足，在网页候选菜单中勾选 E/I 等候选，点击“应用勾选候选”生成组合 mask；
3. 对组合 mask 继续点级删除/补点；
4. 保存 refined mask；
5. 后续由管理员汇总 `v2_manual_refined_samples_v0_1.jsonl`，再进入二次复核。

## 8. 当前限制

- MVP 使用 Python 标准库 HTTP server，不是高并发生产服务；
- 当前点级编辑以点为单位，适合 2048 点规模数据；
- 暂未实现账号登录、样本锁定和多人冲突检测；
- 候选组合已经支持勾选，但暂未实现候选级 reject 原因结构化记录；
- 暂未实现框选/套索选择，只支持点击单点增删；
- refined mask 仍需要二次质量检查，不能直接等同 verified 数据。
