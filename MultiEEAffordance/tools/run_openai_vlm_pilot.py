#!/usr/bin/env python3
"""Run an OpenAI vision model on VLM pilot renders and save 2D masks.

The model is asked to output approximate polygons in pixel coordinates. This
script rasterizes those polygons into binary per-view masks. The masks are
candidate regions only; they still need geometric projection and human review.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from path_utils import resolve_portable_path


EXECUTOR_DEFINITIONS = {
    "gripper": (
        "两指夹爪。只标注可被两侧夹持并稳定受力的区域，例如把手外侧、细长柄部、可夹持边缘。"
        "不要标注大平面中心、孔洞内部、过小按钮或只适合吸盘的大平面。"
    ),
    "suction": (
        "吸盘。只标注局部平整、低曲率、法向一致、面积足够的表面，例如门板/抽屉面板/盒子顶面。"
        "不要标注孔洞、边缘、细杆、把手、拉环、曲率大的区域。"
    ),
    "hook": (
        "钩爪。只标注可挂接或可勾住的结构，例如把手内孔、拉环、孔洞边界、提手开口。"
        "不要标注普通平面、无孔洞表面、按钮或只能吸附的光滑区域。"
    ),
    "dexterous_hand": (
        "灵巧手。只标注类人多指手能稳定包覆、抓握、按压、旋转或精细操作的区域。"
        "不要把所有可接触表面都标为正样本。"
    ),
}


TASK_DEFINITIONS = {
    "pick_up": "抓取并拿起物体，重点是稳定施力并抬起。",
    "lift_carry": "提起并搬运物体，重点是承重、稳定抓握/吸附/挂接。",
    "open_pull": "拉开或打开可动部件，重点是能施加拉力的把手、拉环、面板。",
    "press_push": "按压或推动，重点是按钮、开关、可推动面板、可按压结构。",
}


VIEW_ORDER = ["front", "back", "left", "right", "top", "iso"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenAI VLM on Multi-EE pilot renders.")
    parser.add_argument("--dataset-root", default=".", help="Dataset root.")
    parser.add_argument(
        "--pilot-csv",
        default="processed/metadata/vlm_pilot_samples_v0_1.csv",
        help="Pilot CSV relative to dataset root.",
    )
    parser.add_argument(
        "--renders-root",
        default="processed/vlm_pilot/renders",
        help="Render root relative to dataset root.",
    )
    parser.add_argument(
        "--output-mask-root",
        default="processed/vlm_pilot/vlm_2d_masks",
        help="Output 2D mask root relative to dataset root.",
    )
    parser.add_argument(
        "--output-response-root",
        default="processed/vlm_pilot/vlm_responses",
        help="Output raw VLM response root relative to dataset root.",
    )
    parser.add_argument("--model", default=os.getenv("OPENAI_VLM_MODEL", "gpt-4o-mini"), help="OpenAI vision model.")
    parser.add_argument("--pilot-id", default=None, help="Run only one pilot_id.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of pilot rows.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing masks/responses.")
    parser.add_argument("--dry-run", action="store_true", help="Write prompt package but do not call the API.")
    return parser.parse_args()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_pilot_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Pilot CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"View manifest not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def image_data_url(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Render image not found: {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_prompt(row: dict[str, str], manifest: dict[str, Any]) -> str:
    image_size = int(manifest["image_size"])
    executor = row["executor"]
    task = row["task"]
    notes = row.get("pilot_reason") or row.get("notes") or ""
    return f"""
你是一个用于 3D affordance 数据集构建的视觉标注助手。你会看到同一个 3D 点云物体的 6 个深度渲染视角。

请只针对当前执行器输出候选可操作区域，多边形坐标使用像素坐标，范围是 0 到 {image_size - 1}。

物体类别：{row['object_category']}
任务：{task}
任务说明：{TASK_DEFINITIONS.get(task, task)}
当前执行器：{executor}
执行器标准：{EXECUTOR_DEFINITIONS[executor]}
人工审查指出的问题：{row['issue_type']} / {row['decision']}
pilot 目的：{notes}

请严格输出 JSON，不要输出解释性正文。JSON 格式：
{{
  "sample_id": "{row['sample_id']}",
  "executor": "{executor}",
  "image_size": {image_size},
  "views": [
    {{
      "view": "front",
      "feasible": true,
      "confidence": 0.0,
      "polygons": [
        {{"points": [[x1, y1], [x2, y2], [x3, y3]]}}
      ],
      "notes": "简短中文说明"
    }}
  ]
}}

要求：
1. `views` 必须包含 front、back、left、right、top、iso 六个视角。
2. 如果某个视角看不清或该执行器不可行，设置 `feasible=false` 且 `polygons=[]`。
3. 宁可少标，不要把普通接触面泛化成正样本。
4. 多边形只覆盖可操作区域，不要覆盖背景或整张图。
5. 对 hook 必须看到可挂接/孔洞/环/把手内侧结构才标注。
6. 对 suction 必须是相对平整且面积足够的表面，避免边缘和把手。
"""


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def call_openai(model: str, prompt: str, view_images: list[tuple[str, Path]]) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI()
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for view, path in view_images:
        content.append({"type": "text", "text": f"视角：{view}"})
        content.append({"type": "image_url", "image_url": {"url": image_data_url(path), "detail": "low"}})

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你只输出可解析的 JSON。"},
            {"role": "user", "content": content},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    text = response.choices[0].message.content or "{}"
    return extract_json(text)


def normalize_points(points: Any, image_size: int) -> list[tuple[int, int]]:
    normalized: list[tuple[int, int]] = []
    if not isinstance(points, list):
        return normalized
    for item in points:
        if not isinstance(item, list | tuple) or len(item) < 2:
            continue
        try:
            x = int(round(float(item[0])))
            y = int(round(float(item[1])))
        except (TypeError, ValueError):
            continue
        x = max(0, min(image_size - 1, x))
        y = max(0, min(image_size - 1, y))
        normalized.append((x, y))
    return normalized


def rasterize_response(response: dict[str, Any], image_size: int, output_dir: Path) -> dict[str, int]:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        raise RuntimeError("Rasterizing VLM polygons requires Pillow/PIL.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    view_entries = {entry.get("view"): entry for entry in response.get("views", []) if isinstance(entry, dict)}
    positive_counts: dict[str, int] = {}

    for view in VIEW_ORDER:
        mask = Image.new("L", (image_size, image_size), 0)
        draw = ImageDraw.Draw(mask)
        entry = view_entries.get(view, {})
        polygons = entry.get("polygons", []) if entry.get("feasible", True) is not False else []
        if isinstance(polygons, list):
            for polygon in polygons:
                points = polygon.get("points", polygon) if isinstance(polygon, dict) else polygon
                xy = normalize_points(points, image_size)
                if len(xy) >= 3:
                    draw.polygon(xy, fill=255)
        array = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
        np.save(output_dir / f"{view}.npy", array)
        mask.save(output_dir / f"{view}.png")
        positive_counts[view] = int(array.sum())
    return positive_counts


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    pilot_csv = resolve_path(root, args.pilot_csv)
    renders_root = resolve_path(root, args.renders_root)
    mask_root = resolve_path(root, args.output_mask_root)
    response_root = resolve_path(root, args.output_response_root)

    rows = read_pilot_rows(pilot_csv)
    if args.pilot_id:
        rows = [row for row in rows if row.get("pilot_id") == args.pilot_id]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No pilot rows selected.")

    results = []
    for row in rows:
        sample_id = row["sample_id"]
        executor = row["executor"]
        pilot_id = row["pilot_id"]
        manifest_path = renders_root / sample_id / "view_manifest.json"
        manifest = read_manifest(manifest_path)
        views = {
            entry["view"]: resolve_portable_path(root, entry["render_path"], manifest_path.parent)
            for entry in manifest["views"]
        }
        view_images = [(view, views[view]) for view in VIEW_ORDER]
        prompt = build_prompt(row, manifest)

        response_dir = response_root / pilot_id
        mask_dir = mask_root / sample_id / executor
        response_path = response_dir / "response.json"
        prompt_path = response_dir / "prompt.txt"
        if response_path.exists() and mask_dir.exists() and not args.overwrite:
            results.append({"pilot_id": pilot_id, "sample_id": sample_id, "executor": executor, "status": "skipped"})
            continue
        response_dir.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt, encoding="utf-8")

        if args.dry_run:
            response = {
                "sample_id": sample_id,
                "executor": executor,
                "image_size": int(manifest["image_size"]),
                "views": [{"view": view, "feasible": False, "confidence": 0.0, "polygons": [], "notes": "dry-run"} for view in VIEW_ORDER],
            }
        else:
            response = call_openai(args.model, prompt, view_images)

        with response_path.open("w", encoding="utf-8") as f:
            json.dump(response, f, indent=2, ensure_ascii=False)
            f.write("\n")
        counts = rasterize_response(response, int(manifest["image_size"]), mask_dir)
        results.append(
            {
                "pilot_id": pilot_id,
                "sample_id": sample_id,
                "executor": executor,
                "status": "done",
                "mask_dir": str(mask_dir),
                "positive_pixels": counts,
            }
        )

    print(json.dumps({"model": args.model, "rows": results}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
