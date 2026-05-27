#!/usr/bin/env python3
"""Package local review inputs for offline human annotation.

The maintainer may generate v3 candidates under an external storage mirror such
as /home/lzq/data/MultiEEAffordance, while reviewers run the web app from a
normal project checkout.  This tool copies the reviewer samples JSONL plus every
referenced point/mask/candidate file into a portable archive that can be
extracted at the project root.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any


PATH_KEYS = {
    "point_cloud_path",
    "multi_channel_mask_path",
    "checked_mask_path",
    "source_mask_path",
    "candidate_manifest",
    "candidate_npz",
    "projected_votes",
    "semantic_plan",
    "selection_path",
    "render_manifest",
    "overlay_manifest",
    "rule_filter_path",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package reviewer input files for local point-level review.")
    parser.add_argument("--dataset-root", default="MultiEEAffordance", help="Project dataset root.")
    parser.add_argument(
        "--storage-dataset-root",
        default="",
        help="External storage dataset mirror, e.g. /home/lzq/data/MultiEEAffordance.",
    )
    parser.add_argument("--samples", required=True, help="Reviewer samples JSONL, relative to storage/project dataset root or absolute.")
    parser.add_argument(
        "--staging-root",
        default="",
        help="Temporary staging directory. Defaults next to the output archive.",
    )
    parser.add_argument("--output-tar", required=True, help="Output .tar.gz package path.")
    parser.add_argument("--reviewer", default="", help="Optional reviewer id recorded in PACKAGE_MANIFEST.json.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def dataset_suffix(root_name: str, value: str | Path) -> str | None:
    text = str(value).replace("\\", "/")
    marker = f"/{root_name}/"
    if marker in text:
        return text.split(marker, 1)[1]
    prefix = f"{root_name}/"
    if text.startswith(prefix):
        return text[len(prefix) :]
    return None


def candidate_roots(dataset_root: Path, storage_root: Path | None) -> list[Path]:
    roots: list[Path] = []
    if storage_root is not None:
        roots.append(storage_root.resolve())
    roots.append(dataset_root.resolve())
    return roots


def resolve_existing(value: str | Path | None, roots: list[Path], base_dir: Path | None = None) -> Path | None:
    if value in (None, ""):
        return None
    raw = str(value).strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute() and path.exists():
        return path.resolve()
    if base_dir is not None and not path.is_absolute():
        candidate = base_dir / raw
        if candidate.exists():
            return candidate.resolve()
    for root in roots:
        if not path.is_absolute():
            candidate = root / raw
            if candidate.exists():
                return candidate.resolve()
        suffix = dataset_suffix(root.name, raw)
        if suffix:
            candidate = root / suffix
            if candidate.exists():
                return candidate.resolve()
    return None


def portable_rel(path: Path, roots: list[Path]) -> Path | None:
    for root in roots:
        try:
            rel = path.resolve().relative_to(root.resolve())
        except ValueError:
            continue
        return Path(root.name) / rel
    return None


def collect_paths(obj: Any, paths: set[Path], roots: list[Path], base_dir: Path | None = None) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            lower = str(key).lower()
            if lower.endswith("_path") or lower in PATH_KEYS:
                found = resolve_existing(value, roots, base_dir)
                if found and found.is_file():
                    paths.add(found)
            collect_paths(value, paths, roots, base_dir)
    elif isinstance(obj, list):
        for item in obj:
            collect_paths(item, paths, roots, base_dir)


def add_manifest(paths: set[Path], manifest_path: Path, roots: list[Path]) -> None:
    if not manifest_path.exists() or not manifest_path.is_file():
        return
    paths.add(manifest_path.resolve())
    try:
        manifest = read_json(manifest_path)
    except Exception:
        return
    collect_paths(manifest, paths, roots, manifest_path.parent)


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    storage_root = Path(args.storage_dataset_root).resolve() if args.storage_dataset_root else None
    roots = candidate_roots(dataset_root, storage_root)
    output_tar = Path(args.output_tar).resolve()
    staging_root = Path(args.staging_root).resolve() if args.staging_root else output_tar.with_suffix("").with_suffix("")

    if output_tar.exists() and not args.overwrite:
        raise FileExistsError(f"Output package exists. Use --overwrite: {output_tar}")
    if staging_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Staging directory exists. Use --overwrite: {staging_root}")
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)

    samples_path = resolve_existing(args.samples, roots)
    if samples_path is None:
        raise FileNotFoundError(f"Reviewer samples not found: {args.samples}")

    rows = read_jsonl(samples_path)
    paths: set[Path] = {samples_path}
    collect_paths(rows, paths, roots, samples_path.parent)

    for row in rows:
        pilot_id = row.get("pilot_id") or row.get("review_id")
        if not pilot_id:
            continue
        for root in roots:
            add_manifest(paths, root / "processed/vlm_candidate_v3/3d_candidates" / str(pilot_id) / "candidate_manifest.json", roots)
            add_manifest(paths, root / "processed/vlm_candidate_v3/candidate_overlays" / str(pilot_id) / "overlay_manifest.json", roots)
            add_manifest(paths, root / "processed/vlm_candidate_v3/vlm_selection" / str(pilot_id) / "combined_selection.json", roots)
            add_manifest(paths, root / "processed/vlm_candidate_v3/rule_filter" / str(pilot_id) / "filtered_candidates.json", roots)

    copied: list[str] = []
    skipped: list[str] = []
    for src in sorted(paths):
        rel = portable_rel(src, roots)
        if rel is None:
            skipped.append(str(src))
            continue
        dst = staging_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(rel.as_posix())

    package_manifest = {
        "reviewer": args.reviewer,
        "samples": portable_rel(samples_path, roots).as_posix() if portable_rel(samples_path, roots) else str(samples_path),
        "rows": len(rows),
        "files_copied": len(copied),
        "files_skipped": skipped,
        "extract_at": "project root, so MultiEEAffordance/ lands inside the checkout",
    }
    write_json(staging_root / "PACKAGE_MANIFEST.json", package_manifest)

    output_tar.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_tar, "w:gz") as tar:
        for item in sorted(staging_root.rglob("*")):
            tar.add(item, arcname=item.relative_to(staging_root))

    print(json.dumps({**package_manifest, "output_tar": str(output_tar), "staging_root": str(staging_root)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
