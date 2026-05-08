#!/usr/bin/env python3
"""Rewrite render view manifests to store dataset-relative POSIX paths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from path_utils import dataset_relative_suffix, relative_to_dataset, resolve_portable_path


PATH_KEYS = ("point_cloud_path", "point_index_path", "depth_path", "render_path")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize Multi-EE render manifests to relative paths.")
    parser.add_argument("--dataset-root", default=".", help="Dataset root.")
    parser.add_argument("--renders-root", default="processed/vlm_pilot/renders", help="Renders root relative to dataset root.")
    parser.add_argument("--backup", action="store_true", help="Write .bak copy before modifying a manifest.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned changes without writing files.")
    return parser.parse_args()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def normalize_value(root: Path, manifest_dir: Path, value: str) -> str:
    if not value:
        return value
    suffix = dataset_relative_suffix(root, value)
    if suffix:
        return suffix
    resolved = resolve_portable_path(root, value, manifest_dir)
    return relative_to_dataset(root, resolved)


def normalize_manifest(root: Path, path: Path) -> tuple[dict[str, Any], int]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    changed = 0
    for key in PATH_KEYS:
        if key in data and isinstance(data[key], str):
            new_value = normalize_value(root, path.parent, data[key])
            if new_value != data[key]:
                data[key] = new_value
                changed += 1
    for view in data.get("views", []):
        if not isinstance(view, dict):
            continue
        for key in PATH_KEYS:
            if key in view and isinstance(view[key], str):
                new_value = normalize_value(root, path.parent, view[key])
                if new_value != view[key]:
                    view[key] = new_value
                    changed += 1
    return data, changed


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    renders_root = resolve_path(root, args.renders_root)
    if not renders_root.exists():
        raise FileNotFoundError(f"Renders root not found: {renders_root}")
    manifests = sorted(renders_root.glob("*/view_manifest.json"))
    summary = {"manifests": len(manifests), "changed_manifests": 0, "changed_paths": 0}
    for manifest_path in manifests:
        data, changed = normalize_manifest(root, manifest_path)
        if changed == 0:
            continue
        summary["changed_manifests"] += 1
        summary["changed_paths"] += changed
        if args.dry_run:
            print(f"DRY-RUN {manifest_path}: {changed} paths")
            continue
        if args.backup:
            backup_path = manifest_path.with_suffix(".json.bak")
            backup_path.write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"normalized {manifest_path}: {changed} paths")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
