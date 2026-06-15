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

SOURCE_ASSET_KEYS = (
    "source_asset_id",
    "asset_id",
    "cad_asset_id",
    "model_id",
    "source_model_id",
    "partnet_model_id",
    "shape_id",
)


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


def infer_source_dataset(row: dict) -> str:
    """Return a stable dataset name for split/audit metadata."""

    value = str(row.get("source_dataset") or "").strip()
    if value:
        return value
    object_id = str(row.get("object_id") or row.get("sample_id") or "").strip()
    if object_id.startswith("3danet_full_") or object_id.startswith("3danet_"):
        return "3d_affordancenet"
    if object_id.startswith("partnet_") or object_id.startswith("partnet-mobility"):
        return "partnet_mobility"
    return "unknown"


def infer_source_asset_id(row: dict) -> str:
    """Infer the CAD asset id used for leakage-safe split grouping.

    For 3D AffordanceNet, one original object shape is one CAD asset. If a row
    only has object_id=3danet_full_xxx, that object_id becomes source_asset_id.
    """

    for key in SOURCE_ASSET_KEYS:
        value = str(row.get(key) or "").strip()
        if value:
            return value
    object_id = str(row.get("object_id") or "").strip()
    if object_id:
        return object_id
    source_sample_id = str(row.get("source_sample_id") or "").strip()
    if source_sample_id:
        return source_sample_id
    sample_id = str(row.get("sample_id") or "").strip()
    if sample_id:
        return sample_id
    return "unknown_asset"


def make_asset_uid(row: dict) -> str:
    return f"{infer_source_dataset(row)}:{infer_source_asset_id(row)}"
