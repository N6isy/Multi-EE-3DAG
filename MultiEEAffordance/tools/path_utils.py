"""Path helpers for local-Windows to remote-Linux dataset workflows."""

from __future__ import annotations

from pathlib import Path


def as_posix(value: str | Path) -> str:
    return str(value).replace("\\", "/")


def relative_to_dataset(root: Path, path: Path) -> str:
    """Return a POSIX path relative to the dataset root when possible."""
    root = root.resolve()
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return as_posix(path)


def dataset_relative_suffix(root: Path, value: str | Path) -> str | None:
    """Extract the path after the dataset root name from a saved path string.

    This handles stale manifests created on Windows, for example:
      D:\\VSCode\\...\\MultiEEAffordance\\processed\\vlm_pilot\\...

    On a Linux server the string above is not a valid absolute path, but the
    suffix after "MultiEEAffordance/" is still portable.
    """
    normalized = as_posix(value).strip()
    root_name = root.name
    marker = f"/{root_name}/"
    if marker in normalized:
        return normalized.split(marker, 1)[1]
    prefix = f"{root_name}/"
    if normalized.startswith(prefix):
        return normalized[len(prefix) :]
    return None


def resolve_portable_path(root: Path, value: str | Path, base_dir: Path | None = None) -> Path:
    """Resolve relative, native absolute, and stale cross-OS dataset paths."""
    raw = str(value).strip()
    if not raw:
        return Path("")

    path = Path(raw)
    if path.is_absolute() and path.exists():
        return path

    if base_dir is not None and not path.is_absolute():
        candidate = base_dir / raw
        if candidate.exists():
            return candidate

    suffix = dataset_relative_suffix(root, raw)
    if suffix:
        return root / suffix

    if not path.is_absolute():
        return root / raw
    return path
