#!/usr/bin/env python3
"""
Split v3 candidate samples into reviewer batches and package all referenced files.

Typical usage on the data server:

python package_annotation_batches_from_samples.py \
  --dataset-root /home/lzq/data/MultiEEAffordance \
  --input processed/metadata/v3_candidate_samples_v0_1.jsonl \
  --batch-dir processed/annotation_batches/v0_1 \
  --calibration-objects 0 \
  --archive-format tar.gz \
  --overwrite

The generated archives contain a MultiEEAffordance/ directory with local-safe
relative paths, so reviewers can extract it beside/over their local project data
folder and run the annotation web UI without logging into the server.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import tarfile
import tempfile
import zipfile
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.task_taxonomy import TASK_SUFFIXES

LEGACY_AND_NEW_TASK_SUFFIXES = TASK_SUFFIXES

PATH_LIKE_KEY_RE = re.compile(
    r"(path|file|manifest|npz|npy|ply|mask|candidate|point|render|image|jsonl|json)$",
    re.IGNORECASE,
)
PATH_LIKE_SUFFIXES = (
    ".json",
    ".jsonl",
    ".npz",
    ".npy",
    ".ply",
    ".pcd",
    ".csv",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".html",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split candidate samples into reviewer batches and package referenced "
            "point clouds, masks, candidate manifests and candidate npz files."
        )
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="Dataset root on the server, e.g. /home/lzq/data/MultiEEAffordance.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input v3 candidate samples JSONL, absolute or relative to dataset-root.",
    )
    parser.add_argument(
        "--batch-dir",
        default="processed/annotation_batches/v0_1",
        help="Output batch directory, absolute or relative to dataset-root.",
    )
    parser.add_argument(
        "--reviewers",
        default="reviewer_a,reviewer_b",
        help="Comma-separated reviewer ids. Current splitter alternates object groups over these reviewers.",
    )
    parser.add_argument(
        "--calibration-objects",
        type=int,
        default=0,
        help="Number of leading object groups duplicated to every reviewer for agreement calibration.",
    )
    parser.add_argument(
        "--archive-format",
        choices=["tar.gz", "zip", "none"],
        default="tar.gz",
        help="Archive type for reviewer packages.",
    )
    parser.add_argument(
        "--package-prefix",
        default="MultiEEAffordance",
        help="Top-level directory name inside each archive.",
    )
    parser.add_argument(
        "--max-json-parse-mb",
        type=float,
        default=200.0,
        help="Maximum JSON/JSONL file size to recursively parse for additional file references.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only split and report dependencies; do not copy/package files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing reviewer JSONL files and archives.",
    )
    return parser.parse_args()


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_no}, got {type(row).__name__}")
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            clean = dict(row)
            clean.pop("_line_no", None)
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def strip_task_suffix(sample_id: str) -> str:
    for suffix in LEGACY_AND_NEW_TASK_SUFFIXES:
        if sample_id.endswith(suffix):
            return sample_id[: -len(suffix)]
    return sample_id


def object_key(row: dict[str, Any]) -> str:
    object_id = row.get("object_id") or row.get("object")
    if object_id:
        return str(object_id)
    sample_id = str(row.get("sample_id") or "")
    return strip_task_suffix(sample_id)


def sample_key(row: dict[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("row_key") or ""),
            str(row.get("pilot_id") or row.get("review_id") or ""),
            str(row.get("sample_id") or ""),
            str(row.get("task") or row.get("target_task") or ""),
            str(row.get("target_executor") or row.get("executor") or ""),
        ]
    )


def validate_input_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    row_keys: Counter[str] = Counter()
    triplets: Counter[str] = Counter()
    missing_row_key = 0
    for row in rows:
        explicit = str(row.get("row_key") or "").strip()
        if not explicit:
            missing_row_key += 1
        key = explicit or sample_key(row)
        row_keys[key] += 1
        triplet = "|".join(
            [
                str(row.get("sample_id") or ""),
                str(row.get("task") or row.get("target_task") or ""),
                str(row.get("target_executor") or row.get("executor") or ""),
            ]
        )
        triplets[triplet] += 1
    duplicate_row_keys = {key: count for key, count in row_keys.items() if count > 1}
    duplicate_triplets = {key: count for key, count in triplets.items() if count > 1}
    if duplicate_row_keys:
        raise ValueError(f"Duplicate row_key/sample keys in input: {duplicate_row_keys}")
    if duplicate_triplets:
        raise ValueError(f"Duplicate sample_id+task+executor rows in input: {duplicate_triplets}")
    return {
        "rows": len(rows),
        "missing_row_key": missing_row_key,
        "unique_row_keys": len(row_keys),
        "unique_sample_task_executor": len(triplets),
    }


def split_by_object_groups(
    rows: list[dict[str, Any]], reviewer_ids: list[str], calibration_objects: int
) -> tuple[dict[str, list[dict[str, Any]]], list[tuple[str, list[dict[str, Any]]]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[object_key(row)].append(row)

    group_items = sorted(groups.items(), key=lambda item: item[0])
    assignments = {reviewer_id: [] for reviewer_id in reviewer_ids}

    for idx, (_, group_rows) in enumerate(group_items):
        reviewer_id = reviewer_ids[idx % len(reviewer_ids)]
        assignments[reviewer_id].extend(group_rows)

    if calibration_objects > 0:
        calibration_groups = group_items[:calibration_objects]
        calibration_rows = [row for _, group_rows in calibration_groups for row in group_rows]
        for reviewer_id in reviewer_ids:
            existing = {sample_key(row) for row in assignments[reviewer_id]}
            assignments[reviewer_id].extend(
                [row for row in calibration_rows if sample_key(row) not in existing]
            )

    return assignments, group_items


def is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def to_dataset_relative(path: Path, root: Path) -> str | None:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None


def normalize_string_for_local(value: str, root: Path) -> str:
    """Strip dataset-root prefix from absolute paths inside JSON rows/manifests."""
    root_str = root.resolve().as_posix()
    value2 = value.replace(root_str + "/", "")
    # Also handle a common case where values include the dataset dir name as a prefix.
    dataset_name = root.name
    if value2.startswith(dataset_name + "/"):
        value2 = value2[len(dataset_name) + 1 :]
    return value2


def normalize_obj_for_local(obj: Any, root: Path) -> Any:
    if isinstance(obj, str):
        return normalize_string_for_local(obj, root)
    if isinstance(obj, list):
        return [normalize_obj_for_local(x, root) for x in obj]
    if isinstance(obj, tuple):
        return [normalize_obj_for_local(x, root) for x in obj]
    if isinstance(obj, dict):
        return {k: normalize_obj_for_local(v, root) for k, v in obj.items()}
    return obj


def maybe_path_string(value: str) -> bool:
    if not value or "\n" in value or "\0" in value:
        return False
    lowered = value.lower().split("?", 1)[0]
    if lowered.endswith(PATH_LIKE_SUFFIXES):
        return True
    if "/" in value and not value.startswith("http://") and not value.startswith("https://"):
        if any(ch.isspace() for ch in value.strip()):
            return False
        return True
    return False


def extract_path_like_values(obj: Any, parent_key: str = "") -> list[str]:
    values: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_str = str(key)
            if isinstance(value, str):
                if PATH_LIKE_KEY_RE.search(key_str) or maybe_path_string(value):
                    values.append(value)
            else:
                values.extend(extract_path_like_values(value, key_str))
    elif isinstance(obj, list):
        for item in obj:
            values.extend(extract_path_like_values(item, parent_key))
    elif isinstance(obj, str):
        if maybe_path_string(obj):
            values.append(obj)
    return values


def resolve_candidate_paths(value: str, root: Path, base_dir: Path | None = None) -> list[Path]:
    value = value.strip()
    if not value or value.startswith("http://") or value.startswith("https://"):
        return []

    # Remove dataset root prefix if embedded.
    normalized = normalize_string_for_local(value, root)
    candidates: list[Path] = []

    original = Path(value)
    if original.is_absolute():
        candidates.append(original)
    else:
        candidates.append(root / normalized)
        if base_dir is not None:
            candidates.append(base_dir / value)
            candidates.append(base_dir / normalized)

    # Deduplicate while preserving order.
    out: list[Path] = []
    seen: set[str] = set()
    for p in candidates:
        key = p.as_posix()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def load_json_or_jsonl(path: Path) -> list[Any]:
    if path.suffix == ".jsonl":
        objs = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    objs.append(json.loads(line))
        return objs
    with path.open("r", encoding="utf-8") as f:
        return [json.load(f)]


def collect_dependencies(
    rows: list[dict[str, Any]],
    root: Path,
    max_json_parse_mb: float,
) -> tuple[set[Path], list[dict[str, str]]]:
    """Collect files referenced by rows and by referenced JSON/JSONL manifests."""
    files: set[Path] = set()
    missing: list[dict[str, str]] = []
    queue: deque[tuple[str, Path | None, str]] = deque()

    for row in rows:
        for value in extract_path_like_values(row):
            queue.append((value, None, "sample_row"))

    processed_values: set[tuple[str, str]] = set()
    parsed_json_files: set[Path] = set()
    max_json_bytes = int(max_json_parse_mb * 1024 * 1024)

    while queue:
        value, base_dir, source = queue.popleft()
        norm_key = (value, base_dir.as_posix() if base_dir else "")
        if norm_key in processed_values:
            continue
        processed_values.add(norm_key)

        matched_existing = False
        for candidate in resolve_candidate_paths(value, root, base_dir):
            if candidate.exists():
                matched_existing = True
                if candidate.is_file():
                    files.add(candidate.resolve())
                    rel = to_dataset_relative(candidate, root)
                    # Recursively parse JSON/JSONL manifests to discover candidate npz/mask paths.
                    if (
                        candidate.suffix in {".json", ".jsonl"}
                        and candidate.resolve() not in parsed_json_files
                        and candidate.stat().st_size <= max_json_bytes
                    ):
                        parsed_json_files.add(candidate.resolve())
                        try:
                            for obj in load_json_or_jsonl(candidate):
                                for nested in extract_path_like_values(obj):
                                    queue.append((nested, candidate.parent, f"json:{rel or candidate.name}"))
                        except Exception as exc:  # keep packaging robust
                            missing.append(
                                {
                                    "value": value,
                                    "source": source,
                                    "reason": f"failed_to_parse_json_manifest:{type(exc).__name__}:{exc}",
                                }
                            )
                elif candidate.is_dir():
                    # Avoid accidentally copying huge directories unless explicitly referenced.
                    # The sample/candidate rows should normally point to files.
                    for child in candidate.rglob("*"):
                        if child.is_file():
                            files.add(child.resolve())
                break

        if not matched_existing and maybe_path_string(value):
            missing.append({"value": value, "source": source, "reason": "not_found"})

    return files, missing


def copy_or_rewrite_file(src: Path, dst: Path, root: Path, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing packaged file: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Rewrite JSON/JSONL manifests so absolute server paths become local relative paths.
    if src.suffix in {".json", ".jsonl"}:
        try:
            if src.suffix == ".jsonl":
                with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8", newline="\n") as fout:
                    for line in fin:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        obj = normalize_obj_for_local(obj, root)
                        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                return
            with src.open("r", encoding="utf-8") as fin:
                obj = json.load(fin)
            obj = normalize_obj_for_local(obj, root)
            with dst.open("w", encoding="utf-8") as fout:
                json.dump(obj, fout, ensure_ascii=False, indent=2)
                fout.write("\n")
            return
        except Exception:
            # If it is not valid JSON despite suffix, fall back to raw copy.
            pass

    shutil.copy2(src, dst)


def write_package_readme(path: Path, reviewer_id: str, sample_relpath: str) -> None:
    sample_dir = Path(sample_relpath).parent.as_posix()
    text = f"""# MultiEEAffordance annotation package: {reviewer_id}

This package contains only the samples assigned to `{reviewer_id}` and the files
referenced by those samples: point clouds, masks, candidate manifests and candidate
npz files.

## How to use locally

1. Extract this archive into the parent directory of your local `MultiEEAffordance/`
   data folder, or merge the extracted `MultiEEAffordance/` directory into your
   project data directory.

2. From the project repository root, start the annotation web app with the reviewer
   sample file below. Adjust the script arguments according to your local version:

```bash
python MultiEEAffordance/tools/serve_v2_annotation_app.py \\
  --dataset-root MultiEEAffordance \\
  --samples {sample_relpath} \\
  --review-jsonl {sample_dir}/{reviewer_id}_review_records.jsonl \\
  --output-mask-root {sample_dir}/manual_refined_masks_{reviewer_id} \\
  --output-samples {sample_dir}/{reviewer_id}_refined_samples.jsonl \\
  --port 8765 \\
  --top-k-candidates 12
```

If your local tool expects an absolute dataset root, use the absolute path to the
extracted `MultiEEAffordance` directory.

## Important

- Do not edit the JSONL file manually.
- Keep the generated annotation outputs under `processed/annotation_batches/`.
- Paths in this package are rewritten to be relative to the local
  `MultiEEAffordance/` directory, so the server path `/home/lzq/data/...` is not
  required on the reviewer's machine.
"""
    path.write_text(text, encoding="utf-8")


def create_archive(stage_dir: Path, archive_path: Path, archive_format: str, overwrite: bool) -> None:
    if archive_format == "none":
        return
    if archive_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing archive: {archive_path}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_format == "tar.gz":
        with tarfile.open(archive_path, "w:gz") as tar:
            for child in stage_dir.iterdir():
                tar.add(child, arcname=child.name)
    elif archive_format == "zip":
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file in stage_dir.rglob("*"):
                if file.is_file():
                    zf.write(file, arcname=file.relative_to(stage_dir).as_posix())
    else:
        raise ValueError(f"Unsupported archive format: {archive_format}")


def package_reviewer(
    reviewer_id: str,
    rows: list[dict[str, Any]],
    root: Path,
    batch_dir: Path,
    package_prefix: str,
    archive_format: str,
    max_json_parse_mb: float,
    overwrite: bool,
    dry_run: bool,
) -> dict[str, Any]:
    reviewer_samples_rel = Path(batch_dir).relative_to(root).as_posix() + f"/{reviewer_id}_samples.jsonl"
    normalized_rows = [normalize_obj_for_local(dict(row), root) for row in rows]

    for idx, row in enumerate(normalized_rows):
        row.pop("_line_no", None)
        row.setdefault("reviewer_id", reviewer_id)
        row.setdefault(
            "row_key",
            "|".join(
                [
                    str(row.get("pilot_id") or row.get("review_id") or idx),
                    str(row.get("sample_id") or ""),
                    str(row.get("task") or row.get("target_task") or ""),
                    str(row.get("target_executor") or row.get("executor") or ""),
                ]
            ),
        )

    reviewer_samples_path = batch_dir / f"{reviewer_id}_samples.jsonl"
    write_jsonl(reviewer_samples_path, normalized_rows, overwrite)

    deps, missing = collect_dependencies(normalized_rows, root, max_json_parse_mb)

    archive_suffix = ".tar.gz" if archive_format == "tar.gz" else ".zip"
    archive_path = batch_dir / f"{reviewer_id}_annotation_package{archive_suffix}"
    stage_dir_path = batch_dir / f"{reviewer_id}_package_dir"

    copied_files: list[str] = []
    skipped_external: list[str] = []

    if not dry_run:
        with tempfile.TemporaryDirectory(prefix=f"{reviewer_id}_package_") as tmp:
            stage_dir = Path(tmp)
            package_root = stage_dir / package_prefix
            package_root.mkdir(parents=True, exist_ok=True)

            # Write reviewer samples into package.
            packaged_samples_path = package_root / reviewer_samples_rel
            write_jsonl(packaged_samples_path, normalized_rows, overwrite=True)

            # Copy/rewrite dependencies preserving dataset-root relative layout.
            for src in sorted(deps, key=lambda p: p.as_posix()):
                rel = to_dataset_relative(src, root)
                if rel is None:
                    skipped_external.append(src.as_posix())
                    continue
                dst = package_root / rel
                copy_or_rewrite_file(src, dst, root, overwrite=True)
                copied_files.append(rel)

            # Add local instructions and a package manifest.
            write_package_readme(
                stage_dir / f"README_{reviewer_id}.md",
                reviewer_id,
                reviewer_samples_rel,
            )
            package_manifest = {
                "reviewer_id": reviewer_id,
                "samples": reviewer_samples_rel,
                "rows": len(normalized_rows),
                "objects": len({object_key(row) for row in normalized_rows}),
                "tasks": dict(Counter(str(row.get("task", "")) for row in normalized_rows)),
                "executors": dict(Counter(
                    str(row.get("target_executor") or row.get("executor") or "")
                    for row in normalized_rows
                )),
                "files_copied": len(copied_files),
                "missing_references": missing,
                "skipped_external_files": skipped_external,
                "path_policy": "All paths under dataset-root are rewritten relative to MultiEEAffordance/.",
            }
            with (stage_dir / f"package_manifest_{reviewer_id}.json").open("w", encoding="utf-8") as f:
                json.dump(package_manifest, f, ensure_ascii=False, indent=2)
                f.write("\n")

            if archive_format == "none":
                if stage_dir_path.exists():
                    if overwrite:
                        shutil.rmtree(stage_dir_path)
                    else:
                        raise FileExistsError(f"Package dir exists: {stage_dir_path}")
                shutil.copytree(stage_dir, stage_dir_path)
            else:
                create_archive(stage_dir, archive_path, archive_format, overwrite)

    return {
        "samples": reviewer_samples_path.as_posix(),
        "rows": len(normalized_rows),
        "objects": len({object_key(row) for row in normalized_rows}),
        "tasks": Counter(str(row.get("task", "")) for row in normalized_rows),
        "executors": Counter(
            str(row.get("target_executor") or row.get("executor") or "") for row in normalized_rows
        ),
        "dependencies_found": len(deps),
        "missing_references": missing,
        "archive": None if archive_format == "none" else archive_path.as_posix(),
        "package_dir": stage_dir_path.as_posix() if archive_format == "none" else None,
    }


def json_default(obj: Any) -> Any:
    if isinstance(obj, Counter):
        return dict(obj)
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    input_path = resolve_path(root, args.input)
    batch_dir = resolve_path(root, args.batch_dir)
    reviewer_ids = [x.strip() for x in args.reviewers.split(",") if x.strip()]
    if len(reviewer_ids) < 1:
        raise ValueError("At least one reviewer id is required.")

    rows = read_jsonl(input_path)
    input_validation = validate_input_rows(rows)
    assignments, group_items = split_by_object_groups(rows, reviewer_ids, args.calibration_objects)
    batch_dir.mkdir(parents=True, exist_ok=True)

    reviewer_summaries = {}
    for reviewer_id in reviewer_ids:
        reviewer_summaries[reviewer_id] = package_reviewer(
            reviewer_id=reviewer_id,
            rows=assignments[reviewer_id],
            root=root,
            batch_dir=batch_dir,
            package_prefix=args.package_prefix,
            archive_format=args.archive_format,
            max_json_parse_mb=args.max_json_parse_mb,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )

    summary = {
        "input": input_path.as_posix(),
        "dataset_root": root.as_posix(),
        "batch_dir": batch_dir.as_posix(),
        "rows_total": len(rows),
        "input_validation": input_validation,
        "object_groups_total": len(group_items),
        "reviewers": reviewer_summaries,
        "calibration_objects": args.calibration_objects,
        "archive_format": args.archive_format,
        "dry_run": args.dry_run,
    }

    manifest_path = batch_dir / "batch_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing manifest: {manifest_path}")
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=json_default)
        f.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
