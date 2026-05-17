# AGENT.md

本文件记录本项目中 Codex 与用户协作时必须遵守的长期约定。即使对话上下文被压缩，重新进入项目后也应优先阅读本文件，再继续工作。

## 项目背景

- 项目主题：面向异构末端执行器的多标签 3D 可供性接地研究。
- 当前阶段：构建物体级 Multi-EE Affordance Dataset v0.1 原型，不训练模型。
- 核心输出：给定点云 `P` 和任务 `q`，输出 `[M_gripper, M_suction, M_hook, M_dexterous_hand]` 四通道 mask。
- 当前重要工作流：本地 Codex 修改代码和文档，经 GitHub 同步到远程服务器；远程服务器只负责运行真实模型和大规模脚本。

## 本地与远程工作流

- 本地路径通常为：`D:\VSCode\Multi-EE-3DAG`。
- 远程路径通常为：`~/Multi-EE-3DAG`。
- 代码、配置和文档应写成跨平台相对路径，不要硬编码 Windows 绝对路径。
- 服务器运行命令应以 `--dataset-root MultiEEAffordance` 为入口。
- 远程服务器不能使用 Codex，因此需要在本地完成代码修改、文档更新和 GitHub 推送，再由服务器 `git pull` 后运行。

## 文档记录规则

- `MultiEEAffordance/docs/项目进度日志.md` 只记录实质性研究进展，例如：
  - 数据集结构、metadata、taxonomy、标注规范的确定；
  - pipeline 设计变化；
  - pilot 样本闭环跑通；
  - VLM/SAM2/grounding 路线的有效或失败结论；
  - 人工审查、候选 mask、回投融合等阶段性成果。
- 不要把普通报错、依赖安装、环境兼容性、traceback、路径小修记录到项目进度日志。
- 环境报错和兼容性问题统一记录到：
  - `MultiEEAffordance/docs/报错问题记录.md`
- 如果某个报错引发了研究路线变化，项目进度日志只记录路线变化和结论；详细报错仍放入报错问题记录。

## Git 规则

- 不要默认使用 `git add .`。
- 提交前先查看：
  - `git status --short --untracked-files=normal`
  - `git diff --stat`
- 优先精确 `git add` 本次任务相关文件，避免把数据、中间结果、临时文件或用户未确认文件提交。
- 可以在用户明确要求时执行 `git add`、`git commit`、`git push`。
- `git push` 需要本地 GitHub remote 和认证已经配置好；如果认证弹窗或 token 缺失，需要用户在本机处理。
- 不要提交大模型权重、数据压缩包、`processed/` 中的大体量中间产物，除非用户明确要求。

## 报错处理规则

- 服务器报错时，先判断属于哪一类：
  - 代码 bug；
  - 路径迁移问题；
  - 依赖缺失；
  - 模型权重缺失；
  - transformers / remote code 兼容性；
  - 数据质量或稀疏点云导致的 pipeline 失败。
- 对环境和兼容性问题，优先记录到 `docs/报错问题记录.md`。
- 如果报错只影响运行环境，不要把它包装成项目进展。

## 当前 VLM Pipeline 立场

- Qwen3-VL 更适合作为语义部件规划器，而不是直接输出稀疏点云渲染图上的精确 box / point。
- Florence-2 / GroundingDINO 负责根据部件文本做 2D grounding。
- SAM2 负责从 box 或 point prompt 生成 2D mask。
- `point_index_map` 负责从 2D mask 回投到真实 3D 点云。
- 所有 VLM/SAM2 输出都只是 candidate proposal，不能直接作为 ground truth，必须经过规则检查和人工审查。

## 回复风格

- 给用户的阶段汇报要简短，说明“已经做了什么、接下来做什么”。
- 中文文档默认使用中文撰写。
- 技术解释要直接服务于当前数据构建目标，不要偏向训练模型或完整场景级 pipeline。
