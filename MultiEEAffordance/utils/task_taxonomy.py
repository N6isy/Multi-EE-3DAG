"""Central task taxonomy for legacy candidate generation and five-task review.

The legacy pipeline still generates candidate proposals for coarse tasks.  Human
annotation now uses the finer five-task taxonomy.  Legacy candidates are reused
only as proposals and must not be treated as five-task ground truth.
"""

from __future__ import annotations

from collections.abc import Iterable


TASK_TAXONOMY_VERSION = "v0_2_5tasks"

EXECUTOR_ORDER = ["gripper", "suction", "hook", "dexterous_hand"]

NEW_TASKS = ["lift", "open", "pull", "press", "push"]
LEGACY_TASKS = ["pick_up", "lift_carry", "open_pull", "press_push"]
LEGACY_DEFAULT_ACTIVE_TASKS = ["pick_up", "open_pull", "press_push"]
NEW_DEFAULT_ACTIVE_TASKS = list(NEW_TASKS)
ALL_TASKS = list(LEGACY_TASKS) + list(NEW_TASKS)
TASK_SUFFIXES = tuple(f"_{task}" for task in ALL_TASKS)

LEGACY_TO_NEW_TASKS = {
    "pick_up": ["lift"],
    "open_pull": ["open", "pull"],
    "press_push": ["press", "push"],
    "lift_carry": ["lift"],
}

TASK_DISPLAY = {
    "lift": "Lift",
    "open": "Open",
    "pull": "Pull",
    "press": "Press",
    "push": "Push",
    "pick_up": "Pick Up",
    "lift_carry": "Lift Carry",
    "open_pull": "Open Pull",
    "press_push": "Press Push",
}

TASK_INSTRUCTIONS = {
    "lift": "Lift the object from the supporting surface.",
    "open": "Open an articulated or openable component.",
    "pull": "Pull a handle, ring, lip, panel, or movable component along the pull direction.",
    "press": "Press a button, key, switch, or local pressable part.",
    "push": "Push a panel, surface, button, or movable component along the push direction.",
    "pick_up": "Legacy coarse task: pick up the object. Expand to lift for five-task review.",
    "lift_carry": "Legacy coarse task: lift and carry the object. Expand to lift for five-task review.",
    "open_pull": "Legacy coarse task: open or pull. Expand to open and pull for five-task review.",
    "press_push": "Legacy coarse task: press or push. Expand to press and push for five-task review.",
}


def is_new_task(task: str) -> bool:
    return str(task) in NEW_TASKS


def is_legacy_task(task: str) -> bool:
    return str(task) in LEGACY_TASKS


def is_known_task(task: str) -> bool:
    return is_new_task(task) or is_legacy_task(task)


def expand_legacy_task(task: str) -> list[str]:
    task = str(task)
    if task in LEGACY_TO_NEW_TASKS:
        return list(LEGACY_TO_NEW_TASKS[task])
    if task in NEW_TASKS:
        return [task]
    return []


def task_display(task: str) -> str:
    return TASK_DISPLAY.get(str(task), str(task))


def task_instruction(task: str) -> str:
    return TASK_INSTRUCTIONS.get(str(task), f"Perform task: {task}.")


def parse_task_list(value: str, *, known_tasks: Iterable[str], allow_all: bool = True) -> list[str] | None:
    raw = str(value or "").strip()
    if raw.lower() == "all":
        if allow_all:
            return None
        raise ValueError("'all' is not allowed here.")
    tasks = [item.strip() for item in raw.split(",") if item.strip()]
    known = set(known_tasks)
    unknown = sorted(set(tasks).difference(known))
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}. Known tasks: {sorted(known)}")
    return tasks
