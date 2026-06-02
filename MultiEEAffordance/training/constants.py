"""Training-only constants.

The training pipeline intentionally accepts only the five independent tasks.
Legacy coarse tasks remain supported by annotation preparation tools, but they
must be expanded and manually reviewed before they reach this package.
"""

from __future__ import annotations


TASK_TAXONOMY_VERSION = "v0_2_5tasks"

TASKS = ["lift", "open", "pull", "press", "push"]
TASK_TO_INDEX = {task: index for index, task in enumerate(TASKS)}

EXECUTORS = ["gripper", "suction", "hook", "dexterous_hand"]
EXECUTOR_TO_INDEX = {executor: index for index, executor in enumerate(EXECUTORS)}


def require_five_task(task: str) -> str:
    task = str(task or "").strip()
    if task not in TASK_TO_INDEX:
        raise ValueError(
            f"Training accepts only independent five-task labels {TASKS}; got {task!r}. "
            "Expand legacy proposals and complete human review first."
        )
    return task


def require_executor(executor: str) -> str:
    executor = str(executor or "").strip()
    if executor not in EXECUTOR_TO_INDEX:
        raise ValueError(f"Unknown executor {executor!r}; expected one of {EXECUTORS}.")
    return executor

