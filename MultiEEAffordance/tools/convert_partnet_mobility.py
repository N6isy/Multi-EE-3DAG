#!/usr/bin/env python3
"""Extract and convert PartNet-Mobility assets into portable Multi-EE proposals.

The converter keeps raw PartNet-Mobility link annotations as proposal metadata.
It does not create final affordance ground truth and does not infer executor
labels. Each converted object contains:

  - points.npy: normalized object point cloud with shape [N, 3]
  - parts.npz: link-level proposal masks with shape [K, N]
  - candidate_manifest.json: readable link, joint, and provenance metadata
  - objects_manifest.jsonl: one portable object-level row per converted asset

The output can later be expanded into five-task review rows. A link-level part
proposal is only an annotation aid; reviewers still decide the final [N, 4]
executor mask for lift/open/pull/press/push.
"""

from __future__ import annotations

import argparse
import json
import math
import posixpath
import re
import shutil
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.path_utils import relative_to_dataset  # noqa: E402


DEFAULT_SUPPLEMENT_CATEGORIES = (
    "Box",
    "Bucket",
    "Cabinet",
    "Camera",
    "CoffeeMachine",
    "Dispenser",
    "Kettle",
    "Lighter",
    "Mouse",
    "Oven",
    "Phone",
    "Pliers",
    "Remote",
    "Safe",
    "Stapler",
    "Suitcase",
    "Switch",
    "Toaster",
    "Toilet",
    "WashingMachine",
    "Window",
)


@dataclass
class MeshData:
    vertices: np.ndarray
    triangles: np.ndarray


@dataclass
class PartGeometry:
    link_name: str
    mesh_refs: list[str] = field(default_factory=list)
    resolved_meshes: list[str] = field(default_factory=list)
    missing_meshes: list[str] = field(default_factory=list)
    vertices: list[np.ndarray] = field(default_factory=list)
    triangles: list[np.ndarray] = field(default_factory=list)
    movable: bool = False
    parent_joint: str = ""

    @property
    def merged_vertices(self) -> np.ndarray:
        if not self.vertices:
            return np.zeros((0, 3), dtype=np.float32)
        return np.concatenate(self.vertices, axis=0).astype(np.float32, copy=False)

    @property
    def merged_triangles(self) -> np.ndarray:
        if not self.triangles:
            return np.zeros((0, 3, 3), dtype=np.float32)
        return np.concatenate(self.triangles, axis=0).astype(np.float32, copy=False)

    @property
    def surface_area(self) -> float:
        triangles = self.merged_triangles
        if triangles.shape[0] == 0:
            return float(self.merged_vertices.shape[0])
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        return float((0.5 * np.linalg.norm(cross, axis=1)).sum())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract PartNet-Mobility and convert URDF link meshes into Multi-EE part proposals."
    )
    parser.add_argument("--dataset-root", default=".", help="Dataset mirror root used for portable output paths.")
    parser.add_argument(
        "--zip",
        default="raw/partnet_mobility/partnet-mobility-v0.zip",
        help="Source PartNet-Mobility zip, relative to --dataset-root unless absolute.",
    )
    parser.add_argument(
        "--extract-dir",
        default="raw/partnet_mobility/partnet-mobility-v0",
        help="Extraction directory, relative to --dataset-root unless absolute.",
    )
    parser.add_argument(
        "--stages",
        default="extract,convert",
        help="Comma-separated stages: extract,convert. Use convert to reuse an existing extraction.",
    )
    parser.add_argument("--points-dir", default="processed/points/partnet_mobility_v0_supplement_21cat")
    parser.add_argument("--candidate-dir", default="processed/candidates/partnet_mobility_v0_supplement_21cat")
    parser.add_argument("--manifest", default="manifests/partnet_mobility_v0_supplement_21cat_objects_manifest.jsonl")
    parser.add_argument(
        "--summary",
        default="processed/metadata/partnet_mobility_v0_supplement_21cat_conversion_summary.json",
    )
    parser.add_argument("--sample-size", type=int, default=2048, help="Points sampled per converted object.")
    parser.add_argument(
        "--min-points-per-part",
        type=int,
        default=12,
        help="Minimum sampling budget reserved for each non-empty URDF link when possible.",
    )
    parser.add_argument("--normalize", choices=["none", "unit_sphere", "unit_bbox"], default="unit_sphere")
    parser.add_argument(
        "--categories",
        default=",".join(DEFAULT_SUPPLEMENT_CATEGORIES),
        help=(
            "Comma-separated model_cat filter. Defaults to the selected PartNet-Mobility supplement categories. "
            "Use --categories all to convert every discovered category."
        ),
    )
    parser.add_argument("--max-objects", type=int, help="Optional conversion limit after category filtering.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing extracted files and converted object outputs when possible.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing generated files.")
    parser.add_argument("--strict", action="store_true", help="Stop on the first malformed object instead of recording a skip.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect inputs and print the planned conversion without writing files.")
    return parser.parse_args()


def error(message: str) -> None:
    raise ValueError(message)


def resolve_path(root: Path, path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else root / path


def parse_stages(value: str) -> list[str]:
    stages = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(stages).difference({"extract", "convert"}))
    if unknown:
        error(f"Unknown stages: {unknown}")
    if not stages:
        error("--stages cannot be empty")
    return stages


def parse_categories(value: str | None) -> set[str] | None:
    if not value:
        return None
    if value.strip().lower() in {"all", "*"}:
        return None
    categories = {normalize_category(item) for item in value.split(",") if item.strip()}
    return categories or None


def normalize_category(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sanitize_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("_") or "unnamed"


def print_progress(index: int, total: int, label: str, every: int = 100) -> None:
    if index == total or index == 1 or index % every == 0:
        print(f"[{label}] {index}/{total}", flush=True)


def safe_extract(zip_path: Path, extract_dir: Path, overwrite: bool, resume: bool, dry_run: bool) -> dict[str, int]:
    if not zip_path.exists():
        error(f"PartNet-Mobility zip does not exist: {zip_path}")
    if not zipfile.is_zipfile(zip_path):
        error(f"Not a valid zip file: {zip_path}")

    counts: Counter[str] = Counter()
    with zipfile.ZipFile(zip_path, "r") as zf:
        infos = zf.infolist()
        print(f"[extract] archive_entries={len(infos)} target={extract_dir}")
        for index, info in enumerate(infos, start=1):
            name = info.filename.replace("\\", "/")
            if name.startswith("/") or ".." in Path(name).parts:
                raise ValueError(f"Unsafe zip member path: {info.filename}")
            target = extract_dir / Path(name)
            try:
                target.resolve().relative_to(extract_dir.resolve())
            except ValueError as exc:
                raise ValueError(f"Unsafe zip member path: {info.filename}") from exc

            if info.is_dir():
                counts["directories"] += 1
                if not dry_run:
                    target.mkdir(parents=True, exist_ok=True)
                print_progress(index, len(infos), "extract", every=2000)
                continue

            if target.exists() and resume and not overwrite:
                counts["files_reused"] += 1
                print_progress(index, len(infos), "extract", every=2000)
                continue

            counts["files_written"] += 1
            if not dry_run:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            print_progress(index, len(infos), "extract", every=2000)
    return dict(counts)


def discover_object_dirs(extract_dir: Path) -> list[Path]:
    if not extract_dir.exists():
        error(f"Extraction directory does not exist: {extract_dir}")
    found: dict[str, Path] = {}
    for urdf_path in extract_dir.rglob("mobility.urdf"):
        object_dir = urdf_path.parent
        if object_dir.name.isdigit():
            found.setdefault(object_dir.name, object_dir)
    return [found[key] for key in sorted(found, key=lambda value: int(value))]


def get_category(object_dir: Path) -> str:
    meta_path = object_dir / "meta.json"
    if not meta_path.exists():
        return "Unknown"
    try:
        meta = read_json(meta_path)
    except Exception:
        return "Unknown"
    if isinstance(meta, dict):
        for key in ("model_cat", "category", "cat", "model_category", "object_category"):
            value = meta.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return "Unknown"


def parse_obj(path: Path) -> MeshData:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                tokens = line.strip().split()
                if len(tokens) >= 4:
                    vertices.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
            elif line.startswith("f "):
                raw = line.strip().split()[1:]
                indices: list[int] = []
                for token in raw:
                    head = token.split("/", 1)[0]
                    if not head:
                        continue
                    value = int(head)
                    indices.append(value - 1 if value > 0 else len(vertices) + value)
                if len(indices) >= 3:
                    for pos in range(1, len(indices) - 1):
                        faces.append([indices[0], indices[pos], indices[pos + 1]])

    vertex_array = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    if not faces:
        return MeshData(vertex_array, np.zeros((0, 3, 3), dtype=np.float32))
    face_array = np.asarray(faces, dtype=np.int64)
    valid = np.all((face_array >= 0) & (face_array < vertex_array.shape[0]), axis=1)
    face_array = face_array[valid]
    triangles = vertex_array[face_array] if face_array.shape[0] else np.zeros((0, 3, 3), dtype=np.float32)
    return MeshData(vertex_array, triangles.astype(np.float32, copy=False))


def resolve_mesh_path(object_dir: Path, mesh_ref: str) -> Path | None:
    ref = mesh_ref.replace("\\", "/").strip()
    for prefix in ("package://", "file://"):
        if ref.startswith(prefix):
            ref = ref[len(prefix) :]
    ref = ref.lstrip("/")
    candidates = [
        object_dir / ref,
        object_dir / "textured_objs" / Path(ref).name,
        object_dir / Path(ref).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(object_dir.rglob(Path(ref).name))
    return matches[0] if matches else None


def parse_joint_metadata(root: ET.Element) -> tuple[list[dict[str, str]], dict[str, str]]:
    joints: list[dict[str, str]] = []
    child_to_joint: dict[str, str] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        origin = joint.find("origin")
        axis = joint.find("axis")
        limit = joint.find("limit")
        child_link = child.attrib.get("link", "") if child is not None else ""
        joint_name = joint.attrib.get("name", "")
        if child_link:
            child_to_joint[child_link] = joint_name
        joints.append(
            {
                "name": joint_name,
                "type": joint.attrib.get("type", ""),
                "parent": parent.attrib.get("link", "") if parent is not None else "",
                "child": child_link,
                "origin_xyz": origin.attrib.get("xyz", "") if origin is not None else "",
                "origin_rpy": origin.attrib.get("rpy", "") if origin is not None else "",
                "axis_xyz": axis.attrib.get("xyz", "") if axis is not None else "",
                "limit_lower": limit.attrib.get("lower", "") if limit is not None else "",
                "limit_upper": limit.attrib.get("upper", "") if limit is not None else "",
            }
        )
    return joints, child_to_joint


def parse_parts(object_dir: Path) -> tuple[list[PartGeometry], list[dict[str, str]], list[str]]:
    urdf_path = object_dir / "mobility.urdf"
    root = ET.parse(urdf_path).getroot()
    joints, child_to_joint = parse_joint_metadata(root)
    movable_children = {
        joint["child"]
        for joint in joints
        if joint["type"].lower() not in {"", "fixed"} and joint["child"]
    }
    cache: dict[Path, MeshData] = {}
    missing_meshes: list[str] = []
    parts: list[PartGeometry] = []

    for link in root.findall("link"):
        link_name = link.attrib.get("name", "") or f"link_{len(parts)}"
        part = PartGeometry(
            link_name=link_name,
            movable=link_name in movable_children,
            parent_joint=child_to_joint.get(link_name, ""),
        )
        for visual in link.findall("visual"):
            mesh = visual.find("./geometry/mesh")
            if mesh is None:
                continue
            mesh_ref = mesh.attrib.get("filename", "")
            if not mesh_ref:
                continue
            part.mesh_refs.append(mesh_ref)
            mesh_path = resolve_mesh_path(object_dir, mesh_ref)
            if mesh_path is None:
                part.missing_meshes.append(mesh_ref)
                missing_meshes.append(mesh_ref)
                continue
            part.resolved_meshes.append(mesh_path.relative_to(object_dir).as_posix())
            if mesh_path not in cache:
                cache[mesh_path] = parse_obj(mesh_path)
            mesh_data = cache[mesh_path]
            if mesh_data.vertices.shape[0]:
                part.vertices.append(mesh_data.vertices)
            if mesh_data.triangles.shape[0]:
                part.triangles.append(mesh_data.triangles)
        if part.vertices:
            parts.append(part)

    if parts:
        return parts, joints, missing_meshes

    fallback_meshes = sorted((object_dir / "textured_objs").glob("*.obj"))
    if not fallback_meshes:
        fallback_meshes = sorted(object_dir.rglob("*.obj"))
    if not fallback_meshes:
        return [], joints, missing_meshes

    fallback = PartGeometry(link_name="full_object_fallback")
    for mesh_path in fallback_meshes:
        mesh_data = parse_obj(mesh_path)
        fallback.resolved_meshes.append(mesh_path.relative_to(object_dir).as_posix())
        if mesh_data.vertices.shape[0]:
            fallback.vertices.append(mesh_data.vertices)
        if mesh_data.triangles.shape[0]:
            fallback.triangles.append(mesh_data.triangles)
    return ([fallback] if fallback.vertices else []), joints, missing_meshes


def allocate_counts(parts: list[PartGeometry], total: int, minimum: int, rng: np.random.Generator) -> np.ndarray:
    if total <= 0:
        error("--sample-size must be positive")
    if not parts:
        return np.zeros((0,), dtype=np.int64)
    n_parts = len(parts)
    base = min(max(1, minimum), max(1, total // n_parts))
    counts = np.full(n_parts, base, dtype=np.int64)
    if int(counts.sum()) > total:
        counts[:] = 0
        counts[:total] = 1
        return counts
    remaining = total - int(counts.sum())
    if remaining == 0:
        return counts
    weights = np.asarray([max(part.surface_area, 1e-8) for part in parts], dtype=np.float64)
    weights = weights / weights.sum()
    counts += rng.multinomial(remaining, weights)
    return counts


def sample_part(part: PartGeometry, count: int, rng: np.random.Generator) -> np.ndarray:
    if count <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    triangles = part.merged_triangles
    if triangles.shape[0]:
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        valid = areas > 1e-12
        triangles = triangles[valid]
        areas = areas[valid]
        if triangles.shape[0]:
            chosen = rng.choice(triangles.shape[0], size=count, replace=True, p=areas / areas.sum())
            tri = triangles[chosen]
            u = rng.random(count)
            v = rng.random(count)
            sqrt_u = np.sqrt(u)
            weights_a = 1.0 - sqrt_u
            weights_b = sqrt_u * (1.0 - v)
            weights_c = sqrt_u * v
            return (
                tri[:, 0] * weights_a[:, None]
                + tri[:, 1] * weights_b[:, None]
                + tri[:, 2] * weights_c[:, None]
            ).astype(np.float32)
    vertices = part.merged_vertices
    if vertices.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    chosen = rng.choice(vertices.shape[0], size=count, replace=vertices.shape[0] < count)
    return vertices[chosen].astype(np.float32, copy=False)


def normalize_points(points: np.ndarray, mode: str) -> tuple[np.ndarray, dict[str, Any]]:
    if mode == "none":
        return points.astype(np.float32, copy=False), {"mode": mode}
    if mode == "unit_sphere":
        center = points.mean(axis=0)
        shifted = points - center
        scale = float(np.linalg.norm(shifted, axis=1).max())
    elif mode == "unit_bbox":
        min_xyz = points.min(axis=0)
        max_xyz = points.max(axis=0)
        center = (min_xyz + max_xyz) / 2.0
        shifted = points - center
        scale = float((max_xyz - min_xyz).max())
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")
    if scale <= 1e-12:
        raise ValueError("Degenerate point cloud cannot be normalized")
    return (shifted / scale).astype(np.float32), {"mode": mode, "center": center.tolist(), "scale": scale}


def convert_object(
    root: Path,
    object_dir: Path,
    points_dir: Path,
    candidate_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    raw_model_id = object_dir.name
    object_id = f"partnet_mobility_{raw_model_id}"
    category = get_category(object_dir)
    parts, joints, missing_meshes = parse_parts(object_dir)
    if not parts:
        raise ValueError("No usable visual OBJ mesh found")

    rng = np.random.default_rng(args.seed + int(raw_model_id))
    counts = allocate_counts(parts, args.sample_size, args.min_points_per_part, rng)
    sampled: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    kept_parts: list[PartGeometry] = []
    for part, count in zip(parts, counts, strict=True):
        points = sample_part(part, int(count), rng)
        if points.shape[0] == 0:
            continue
        kept_parts.append(part)
        sampled.append(points)
        labels.append(np.full(points.shape[0], len(kept_parts) - 1, dtype=np.int32))
    if not sampled:
        raise ValueError("Sampling produced no points")

    points = np.concatenate(sampled, axis=0)
    point_part_index = np.concatenate(labels, axis=0)
    order = rng.permutation(points.shape[0])
    points = points[order]
    point_part_index = point_part_index[order]
    points, normalization = normalize_points(points, args.normalize)

    candidate_masks = np.zeros((len(kept_parts), points.shape[0]), dtype=np.uint8)
    for index in range(len(kept_parts)):
        candidate_masks[index, point_part_index == index] = 1
    candidate_ids = [f"part_{index:03d}_{sanitize_id(part.link_name)}" for index, part in enumerate(kept_parts)]

    object_candidate_dir = candidate_dir / object_id
    points_path = points_dir / f"{object_id}.npy"
    npz_path = object_candidate_dir / "parts.npz"
    candidate_manifest_path = object_candidate_dir / "candidate_manifest.json"
    if not args.dry_run:
        points_path.parent.mkdir(parents=True, exist_ok=True)
        object_candidate_dir.mkdir(parents=True, exist_ok=True)
        np.save(points_path, points)
        np.savez_compressed(
            npz_path,
            candidate_masks=candidate_masks,
            point_part_index=point_part_index,
            candidate_ids=np.asarray(candidate_ids, dtype=np.str_),
            candidate_names=np.asarray([part.link_name for part in kept_parts], dtype=np.str_),
            source_links=np.asarray([part.link_name for part in kept_parts], dtype=np.str_),
            is_movable=np.asarray([part.movable for part in kept_parts], dtype=np.uint8),
        )

    candidates = []
    for index, (part, candidate_id) in enumerate(zip(kept_parts, candidate_ids, strict=True)):
        point_count = int(candidate_masks[index].sum())
        candidates.append(
            {
                "candidate_id": candidate_id,
                "candidate_name": part.link_name,
                "candidate_family": "partnet_mobility_urdf_link",
                "source_link": part.link_name,
                "source_joint": part.parent_joint,
                "is_movable": part.movable,
                "point_count": point_count,
                "point_fraction": point_count / max(1, points.shape[0]),
                "mesh_refs": part.mesh_refs,
                "resolved_meshes": part.resolved_meshes,
                "missing_meshes": part.missing_meshes,
                "provenance": "partnet_mobility_mobility_urdf_visual_link",
                "need_review": True,
            }
        )

    candidate_manifest = {
        "version": "v0_1",
        "pipeline": "partnet_mobility_urdf_link_proposals",
        "proposal_only": True,
        "object_id": object_id,
        "raw_model_id": raw_model_id,
        "source_dataset": "partnet_mobility",
        "object_category": category,
        "point_cloud_path": relative_to_dataset(root, points_path),
        "candidate_npz": relative_to_dataset(root, npz_path),
        "candidate_count": len(candidates),
        "point_count": int(points.shape[0]),
        "normalization": normalization,
        "joints": joints,
        "missing_visual_mesh_refs": sorted(set(missing_meshes)),
        "candidates": candidates,
        "notes": (
            "URDF link masks are proposal regions only. They are not executor labels and must not be used "
            "as five-task ground truth without task-aware rule checks and human review."
        ),
    }
    if not args.dry_run:
        write_json(candidate_manifest_path, candidate_manifest)

    return {
        "object_id": object_id,
        "raw_model_id": raw_model_id,
        "source_dataset": "partnet_mobility",
        "object_category": category,
        "point_cloud_path": relative_to_dataset(root, points_path),
        "candidate_manifest": relative_to_dataset(root, candidate_manifest_path),
        "candidate_npz": relative_to_dataset(root, npz_path),
        "raw_object_dir": relative_to_dataset(root, object_dir),
        "raw_urdf_path": relative_to_dataset(root, object_dir / "mobility.urdf"),
        "raw_meta_path": relative_to_dataset(root, object_dir / "meta.json") if (object_dir / "meta.json").exists() else "",
        "raw_result_path": relative_to_dataset(root, object_dir / "result.json") if (object_dir / "result.json").exists() else "",
        "point_count": int(points.shape[0]),
        "part_candidate_count": len(candidates),
        "movable_part_count": sum(1 for part in kept_parts if part.movable),
        "joint_count": len(joints),
        "missing_visual_mesh_ref_count": len(set(missing_meshes)),
        "proposal_only": True,
        "task_taxonomy_version": "v0_2_5tasks",
        "notes": "PartNet-Mobility URDF link proposals. Expand into task rows before manual review.",
    }


def main() -> int:
    args = parse_args()
    try:
        root = Path(args.dataset_root).resolve()
        stages = parse_stages(args.stages)
        categories = parse_categories(args.categories)
        zip_path = resolve_path(root, args.zip)
        extract_dir = resolve_path(root, args.extract_dir)
        points_dir = resolve_path(root, args.points_dir)
        candidate_dir = resolve_path(root, args.candidate_dir)
        manifest_path = resolve_path(root, args.manifest)
        summary_path = resolve_path(root, args.summary)

        extract_summary: dict[str, int] = {}
        if "extract" in stages:
            extract_summary = safe_extract(zip_path, extract_dir, args.overwrite, args.resume, args.dry_run)
        if "convert" not in stages:
            print(json.dumps({"status": "ok", "stages": stages, "extract": extract_summary}, indent=2))
            return 0

        object_dirs = discover_object_dirs(extract_dir)
        print(f"[convert] discovered_objects={len(object_dirs)} source={extract_dir}")
        rows: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        category_counts: Counter[str] = Counter()
        total_part_candidates = 0
        total_movable_parts = 0

        selected_dirs: list[Path] = []
        for object_dir in object_dirs:
            category = get_category(object_dir)
            if categories is not None and normalize_category(category) not in categories:
                continue
            selected_dirs.append(object_dir)
            if args.max_objects is not None and len(selected_dirs) >= args.max_objects:
                break

        for index, object_dir in enumerate(selected_dirs, start=1):
            object_id = object_dir.name
            output_points = points_dir / f"partnet_mobility_{object_id}.npy"
            output_manifest = candidate_dir / f"partnet_mobility_{object_id}" / "candidate_manifest.json"
            if args.resume and not args.overwrite and output_points.exists() and output_manifest.exists():
                existing = read_json(output_manifest)
                existing_candidates = existing.get("candidates", []) if isinstance(existing, dict) else []
                movable_count = sum(
                    1 for candidate in existing_candidates if isinstance(candidate, dict) and candidate.get("is_movable")
                )
                row = {
                    "object_id": f"partnet_mobility_{object_id}",
                    "raw_model_id": object_id,
                    "source_dataset": "partnet_mobility",
                    "object_category": get_category(object_dir),
                    "point_cloud_path": relative_to_dataset(root, output_points),
                    "candidate_manifest": relative_to_dataset(root, output_manifest),
                    "candidate_npz": relative_to_dataset(root, output_manifest.parent / "parts.npz"),
                    "raw_object_dir": relative_to_dataset(root, object_dir),
                    "raw_urdf_path": relative_to_dataset(root, object_dir / "mobility.urdf"),
                    "raw_meta_path": relative_to_dataset(root, object_dir / "meta.json") if (object_dir / "meta.json").exists() else "",
                    "raw_result_path": relative_to_dataset(root, object_dir / "result.json") if (object_dir / "result.json").exists() else "",
                    "point_count": int(existing.get("point_count", 0)) if isinstance(existing, dict) else 0,
                    "part_candidate_count": len(existing_candidates),
                    "movable_part_count": movable_count,
                    "joint_count": len(existing.get("joints", [])) if isinstance(existing, dict) else 0,
                    "missing_visual_mesh_ref_count": len(existing.get("missing_visual_mesh_refs", [])) if isinstance(existing, dict) else 0,
                    "proposal_only": True,
                    "task_taxonomy_version": "v0_2_5tasks",
                    "notes": "Existing converted PartNet-Mobility proposal reused by --resume.",
                }
                rows.append(row)
                category_counts[row["object_category"]] += 1
                total_part_candidates += int(row["part_candidate_count"])
                total_movable_parts += int(row["movable_part_count"])
                print_progress(index, len(selected_dirs), "convert")
                continue
            try:
                row = convert_object(root, object_dir, points_dir, candidate_dir, args)
                rows.append(row)
                category_counts[row["object_category"]] += 1
                total_part_candidates += int(row["part_candidate_count"])
                total_movable_parts += int(row["movable_part_count"])
            except Exception as exc:
                skipped.append({"raw_model_id": object_id, "error": str(exc)})
                if args.strict:
                    raise
            print_progress(index, len(selected_dirs), "convert")

        summary = {
            "version": "v0_1",
            "pipeline": "partnet_mobility_extract_and_convert",
            "proposal_only": True,
            "dataset_root": str(root),
            "zip": str(zip_path),
            "extract_dir": str(extract_dir),
            "stages": stages,
            "dry_run": args.dry_run,
            "category_filter": sorted(categories) if categories is not None else "all",
            "discovered_raw_objects": len(object_dirs),
            "selected_raw_objects": len(selected_dirs),
            "converted_or_reused_objects": len(rows),
            "skipped_objects": len(skipped),
            "total_part_candidates": total_part_candidates,
            "total_movable_parts": total_movable_parts,
            "categories": dict(category_counts.most_common()),
            "extract": extract_summary,
            "manifest": relative_to_dataset(root, manifest_path),
            "skipped": skipped[:200],
            "notes": (
                "Raw object count and downstream five-task sample-row count are different quantities. "
                "PartNet-Mobility link proposals are not final [N,4] affordance labels."
            ),
        }
        if not args.dry_run:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with manifest_path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            write_json(summary_path, summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
