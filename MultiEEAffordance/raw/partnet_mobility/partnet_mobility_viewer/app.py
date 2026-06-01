#!/usr/bin/env python3
"""PartNet-Mobility zip viewer.

A lightweight FastAPI app for browsing a PartNet-Mobility zip without extracting
all data in advance. It indexes object IDs, metadata/json/URDF files and meshes,
then serves a small web UI for inspection.
"""
from __future__ import annotations

import argparse
import io
import json
import mimetypes
import os
import posixpath
import re
import tempfile
import textwrap
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from xml.etree import ElementTree as ET

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"

TEXT_EXTS = {
    ".txt", ".json", ".urdf", ".xml", ".yaml", ".yml", ".csv", ".mtl", ".obj", ".log"
}
MESH_EXTS = {".obj", ".ply", ".stl", ".dae", ".off"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
JSON_NAMES = {"meta.json", "result.json", "mobility_vhacd.urdf.json", "mobility.urdf.json"}
URDF_NAMES = {"mobility.urdf", "mobility_vhacd.urdf", "semantic.urdf"}


def safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception as exc:  # pragma: no cover
        return {"__parse_error__": str(exc), "__raw_prefix__": text[:4000]}


def find_category(obj: Any) -> Optional[str]:
    """Best-effort category finder for PartNet-Mobility meta/result JSON variants."""
    preferred_keys = [
        "model_cat", "category", "cat", "model_category", "object_category", "name"
    ]
    if isinstance(obj, dict):
        for key in preferred_keys:
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        # Shallow search first, because result.json can be hierarchical.
        for val in obj.values():
            if isinstance(val, dict):
                cat = find_category(val)
                if cat:
                    return cat
    elif isinstance(obj, list):
        for val in obj[:10]:
            cat = find_category(val)
            if cat:
                return cat
    return None


def summarize_json(obj: Any, max_items: int = 8) -> Any:
    """Make a compact preview that is safe to display in a list/table."""
    if isinstance(obj, dict):
        out = {}
        for i, (k, v) in enumerate(obj.items()):
            if i >= max_items:
                out["…"] = f"{len(obj) - max_items} more keys"
                break
            if isinstance(v, (dict, list)):
                out[k] = summarize_json(v, max_items=max_items)
            else:
                out[k] = v
        return out
    if isinstance(obj, list):
        return [summarize_json(x, max_items=max_items) for x in obj[:max_items]] + (
            [f"… {len(obj) - max_items} more items"] if len(obj) > max_items else []
        )
    return obj


def normalize_zip_path(path: str) -> str:
    path = path.replace("\\", "/").lstrip("/")
    return posixpath.normpath(path)


def get_object_id_from_zip_path(path: str) -> Optional[Tuple[str, str]]:
    """Return (object_id, relative_path_under_object) if a numeric dir is found."""
    parts = normalize_zip_path(path).split("/")
    for i, part in enumerate(parts[:-1]):
        # PartNet-Mobility object folders are numeric ids like 100248.
        if re.fullmatch(r"\d+", part):
            return part, "/".join(parts[i + 1 :])
    return None


@dataclass
class ObjectEntry:
    object_id: str
    files: Dict[str, str] = field(default_factory=dict)  # rel_path -> full_zip_path
    category: str = "Unknown"

    @property
    def file_count(self) -> int:
        return len(self.files)

    def files_by_ext(self, exts: set[str]) -> List[str]:
        return sorted([p for p in self.files if Path(p).suffix.lower() in exts])

    def first_existing(self, names: Iterable[str]) -> Optional[str]:
        lower_map = {p.lower(): p for p in self.files}
        for name in names:
            if name.lower() in lower_map:
                return lower_map[name.lower()]
        # fallback: basename match in nested dir
        for p in self.files:
            if posixpath.basename(p).lower() in {n.lower() for n in names}:
                return p
        return None


class PartNetZipIndex:
    def __init__(self, zip_path: str):
        self.zip_path = str(Path(zip_path).expanduser())
        if not os.path.exists(self.zip_path):
            raise FileNotFoundError(self.zip_path)
        if not zipfile.is_zipfile(self.zip_path):
            raise ValueError(f"Not a valid zip file: {self.zip_path}")
        self.zf = zipfile.ZipFile(self.zip_path, "r")
        self.objects: Dict[str, ObjectEntry] = {}
        self._build_index()
        self._load_categories()

    def _build_index(self) -> None:
        for info in self.zf.infolist():
            if info.is_dir():
                continue
            parsed = get_object_id_from_zip_path(info.filename)
            if not parsed:
                continue
            object_id, rel_path = parsed
            if not rel_path:
                continue
            entry = self.objects.setdefault(object_id, ObjectEntry(object_id=object_id))
            # Keep the first if duplicate rel paths exist.
            entry.files.setdefault(normalize_zip_path(rel_path), info.filename)

    def _load_categories(self) -> None:
        for entry in self.objects.values():
            for candidate in ["meta.json", "result.json"]:
                rel = entry.first_existing([candidate])
                if not rel:
                    continue
                try:
                    obj = self.read_json(entry.object_id, rel)
                    cat = find_category(obj)
                    if cat:
                        entry.category = cat
                        break
                except Exception:
                    continue

    def stats(self) -> Dict[str, Any]:
        categories = Counter(e.category for e in self.objects.values())
        ext_counter = Counter()
        mesh_total = 0
        image_total = 0
        json_total = 0
        for e in self.objects.values():
            for rel in e.files:
                ext = Path(rel).suffix.lower() or "[no_ext]"
                ext_counter[ext] += 1
                if ext in MESH_EXTS:
                    mesh_total += 1
                elif ext in IMAGE_EXTS:
                    image_total += 1
                elif ext == ".json":
                    json_total += 1
        return {
            "zip_path": self.zip_path,
            "object_count": len(self.objects),
            "category_count": len(categories),
            "categories": dict(categories.most_common()),
            "top_extensions": dict(ext_counter.most_common(20)),
            "mesh_file_count": mesh_total,
            "image_file_count": image_total,
            "json_file_count": json_total,
        }

    def list_objects(self, q: str = "", category: str = "", limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        items = list(self.objects.values())
        if category:
            items = [e for e in items if e.category.lower() == category.lower()]
        if q:
            q_lower = q.lower()
            items = [
                e for e in items
                if q_lower in e.object_id.lower() or q_lower in e.category.lower() or any(q_lower in p.lower() for p in e.files)
            ]
        items.sort(key=lambda e: (e.category, int(e.object_id) if e.object_id.isdigit() else e.object_id))
        total = len(items)
        page = items[offset: offset + limit]
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "items": [self.entry_summary(e) for e in page],
        }

    def entry_summary(self, e: ObjectEntry) -> Dict[str, Any]:
        json_files = e.files_by_ext({".json"})
        mesh_files = e.files_by_ext(MESH_EXTS)
        image_files = e.files_by_ext(IMAGE_EXTS)
        urdf_files = [p for p in sorted(e.files) if posixpath.basename(p).lower() in URDF_NAMES or p.lower().endswith(".urdf")]
        return {
            "object_id": e.object_id,
            "category": e.category,
            "file_count": e.file_count,
            "json_count": len(json_files),
            "mesh_count": len(mesh_files),
            "image_count": len(image_files),
            "urdf_count": len(urdf_files),
            "key_files": [p for p in [e.first_existing(["meta.json"]), e.first_existing(["result.json"]), e.first_existing(["mobility.urdf"])] if p],
        }

    def get_entry(self, object_id: str) -> ObjectEntry:
        try:
            return self.objects[object_id]
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown object_id: {object_id}")

    def resolve(self, object_id: str, rel_path: str) -> str:
        entry = self.get_entry(object_id)
        rel_path = normalize_zip_path(rel_path)
        if rel_path not in entry.files:
            raise HTTPException(status_code=404, detail=f"File not found under object {object_id}: {rel_path}")
        return entry.files[rel_path]

    def read_bytes(self, object_id: str, rel_path: str) -> bytes:
        full = self.resolve(object_id, rel_path)
        return self.zf.read(full)

    def read_text(self, object_id: str, rel_path: str, max_bytes: Optional[int] = None) -> str:
        data = self.read_bytes(object_id, rel_path)
        if max_bytes is not None and len(data) > max_bytes:
            data = data[:max_bytes]
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("latin-1", errors="replace")

    def read_json(self, object_id: str, rel_path: str) -> Any:
        return safe_json_loads(self.read_text(object_id, rel_path))

    def object_detail(self, object_id: str) -> Dict[str, Any]:
        entry = self.get_entry(object_id)
        files = sorted(entry.files)
        json_files = entry.files_by_ext({".json"})
        mesh_files = entry.files_by_ext(MESH_EXTS)
        image_files = entry.files_by_ext(IMAGE_EXTS)
        text_files = [p for p in files if Path(p).suffix.lower() in TEXT_EXTS]
        urdf_files = [p for p in files if p.lower().endswith(".urdf")]
        meta_rel = entry.first_existing(["meta.json"])
        result_rel = entry.first_existing(["result.json"])
        urdf_rel = entry.first_existing(["mobility.urdf", "mobility_vhacd.urdf", "semantic.urdf"])
        detail = {
            "summary": self.entry_summary(entry),
            "files": files,
            "json_files": json_files,
            "mesh_files": mesh_files,
            "image_files": image_files,
            "text_files": text_files,
            "urdf_files": urdf_files,
            "meta_preview": None,
            "result_preview": None,
            "urdf_summary": None,
        }
        if meta_rel:
            detail["meta_preview"] = summarize_json(self.read_json(object_id, meta_rel))
        if result_rel:
            detail["result_preview"] = summarize_json(self.read_json(object_id, result_rel))
        if urdf_rel:
            detail["urdf_summary"] = self.parse_urdf(object_id, urdf_rel)
        return detail

    def parse_urdf(self, object_id: str, rel_path: str) -> Dict[str, Any]:
        text = self.read_text(object_id, rel_path)
        try:
            root = ET.fromstring(text)
        except Exception as exc:
            return {"parse_error": str(exc), "raw_prefix": text[:2000]}

        links = []
        for link in root.findall("link"):
            name = link.attrib.get("name", "")
            meshes = []
            for mesh in link.findall(".//mesh"):
                filename = mesh.attrib.get("filename")
                if filename:
                    meshes.append(filename)
            links.append({"name": name, "mesh_refs": meshes})

        joints = []
        for joint in root.findall("joint"):
            parent = joint.find("parent")
            child = joint.find("child")
            origin = joint.find("origin")
            axis = joint.find("axis")
            limit = joint.find("limit")
            joints.append({
                "name": joint.attrib.get("name", ""),
                "type": joint.attrib.get("type", ""),
                "parent": parent.attrib.get("link", "") if parent is not None else "",
                "child": child.attrib.get("link", "") if child is not None else "",
                "origin_xyz": origin.attrib.get("xyz", "") if origin is not None else "",
                "origin_rpy": origin.attrib.get("rpy", "") if origin is not None else "",
                "axis_xyz": axis.attrib.get("xyz", "") if axis is not None else "",
                "limit_lower": limit.attrib.get("lower", "") if limit is not None else "",
                "limit_upper": limit.attrib.get("upper", "") if limit is not None else "",
            })
        return {
            "file": rel_path,
            "robot_name": root.attrib.get("name", ""),
            "link_count": len(links),
            "joint_count": len(joints),
            "links": links,
            "joints": joints,
        }


index: Optional[PartNetZipIndex] = None
app = FastAPI(title="PartNet-Mobility Zip Viewer", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/stats")
def api_stats() -> Dict[str, Any]:
    assert index is not None
    return index.stats()


@app.get("/api/objects")
def api_objects(
    q: str = "",
    category: str = "",
    limit: int = Query(50, ge=1, le=300),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    assert index is not None
    return index.list_objects(q=q, category=category, limit=limit, offset=offset)


@app.get("/api/object/{object_id}")
def api_object(object_id: str) -> Dict[str, Any]:
    assert index is not None
    return index.object_detail(object_id)


@app.get("/api/file_text/{object_id}")
def api_file_text(object_id: str, path: str, max_bytes: int = Query(2_000_000, ge=1, le=20_000_000)) -> Dict[str, Any]:
    assert index is not None
    rel = normalize_zip_path(path)
    ext = Path(rel).suffix.lower()
    if ext not in TEXT_EXTS:
        raise HTTPException(status_code=400, detail=f"Not a text-like file: {rel}")
    data = index.read_bytes(object_id, rel)
    truncated = len(data) > max_bytes
    text = data[:max_bytes].decode("utf-8", errors="replace")
    return {"object_id": object_id, "path": rel, "size_bytes": len(data), "truncated": truncated, "text": text}


@app.get("/api/file_json/{object_id}")
def api_file_json(object_id: str, path: str) -> JSONResponse:
    assert index is not None
    rel = normalize_zip_path(path)
    if Path(rel).suffix.lower() != ".json":
        raise HTTPException(status_code=400, detail=f"Not a json file: {rel}")
    obj = index.read_json(object_id, rel)
    return JSONResponse(obj)


@app.get("/api/file_raw/{object_id}")
def api_file_raw(object_id: str, path: str) -> Response:
    assert index is not None
    rel = normalize_zip_path(path)
    data = index.read_bytes(object_id, rel)
    media_type, _ = mimetypes.guess_type(rel)
    media_type = media_type or "application/octet-stream"
    headers = {"Content-Disposition": f'inline; filename="{posixpath.basename(rel)}"'}
    return Response(content=data, media_type=media_type, headers=headers)


def create_app(zip_path: str) -> FastAPI:
    global index
    index = PartNetZipIndex(zip_path)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a local web viewer for PartNet-Mobility zip.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Example:
          python app.py --zip /home/lzq/Multi-EE-3DAG/MultiEEAffordance/raw/partnet_mobility/partnet-mobility-v0.zip --host 127.0.0.1 --port 7860
        """),
    )
    parser.add_argument("--zip", default=os.environ.get("PARTNET_ZIP", ""), help="Path to partnet-mobility-v0.zip")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7860")))
    args = parser.parse_args()
    if not args.zip:
        raise SystemExit("Please pass --zip or set PARTNET_ZIP")
    create_app(args.zip)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


# Uvicorn factory support: uvicorn app:create_app --factory --app-dir .
if __name__ == "__main__":
    main()
