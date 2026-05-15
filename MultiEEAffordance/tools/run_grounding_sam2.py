#!/usr/bin/env python3
"""Ground semantic part queries in 2D and optionally segment them with SAM2.

This is an integration skeleton for the semantic-part pipeline. It supports a
stable manual/dry-run path now, and leaves explicit backend slots for
GroundingDINO or Florence-2.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from path_utils import relative_to_dataset, resolve_portable_path
from render_multiview import write_png_rgb
from run_qwen3vl_sam2_pilot import load_sam2_predictor, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run open-vocabulary grounding + optional SAM2 segmentation.")
    parser.add_argument("--dataset-root", default=".", help="Dataset root directory.")
    parser.add_argument("--config", default="configs/qwen3vl_sam2_pilot.yaml", help="YAML config relative to dataset root.")
    parser.add_argument(
        "--pilot-csv",
        default="processed/metadata/vlm_pilot_samples_v0_1.csv",
        help="Pilot CSV relative to dataset root.",
    )
    parser.add_argument(
        "--renders-root",
        default="processed/vlm_semantic_part/renders",
        help="VLM-friendly render root relative to dataset root.",
    )
    parser.add_argument(
        "--part-plan-root",
        default="processed/vlm_semantic_part/part_plans",
        help="Part-plan root relative to dataset root.",
    )
    parser.add_argument(
        "--output-root",
        default="processed/vlm_semantic_part/grounded_2d",
        help="Output 2D grounding root relative to dataset root.",
    )
    parser.add_argument("--pilot-id", default=None, help="Run only one pilot row.")
    parser.add_argument("--limit", type=int, default=None, help="Limit selected pilot rows.")
    parser.add_argument(
        "--backend",
        default="manual-json",
        choices=["manual-json", "grounding-dino", "florence2"],
        help="Grounding backend. Current robust path is manual-json; other backends are explicit integration slots.",
    )
    parser.add_argument(
        "--manual-boxes",
        default=None,
        help="Optional JSON with per-view boxes. Format: {'views': {'view': [{'query': str, 'box': [x1,y1,x2,y2], 'score': 0.9}]}}.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Use zoom-crop boxes as placeholder boxes.")
    parser.add_argument("--run-sam2", action="store_true", help="Run SAM2 on grounded boxes.")
    parser.add_argument("--box-mask-only", action="store_true", help="Use rectangular box masks instead of SAM2 masks.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing grounding outputs.")
    parser.add_argument("--validate-only", action="store_true", help="Validate files only.")
    return parser.parse_args()


def resolve_path(root: Path, value: str | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Pilot CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def selected_rows(root: Path, args: argparse.Namespace) -> list[dict[str, str]]:
    rows = read_csv(resolve_path(root, args.pilot_csv))
    if args.pilot_id:
        rows = [row for row in rows if row.get("pilot_id") == args.pilot_id]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No pilot rows selected.")
    return rows


def normalize_box(box: Any, image_size: int) -> list[int] | None:
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    try:
        values = [int(round(float(v))) for v in box[:4]]
    except (TypeError, ValueError):
        return None
    x1, y1, x2, y2 = values
    x1, x2 = sorted([max(0, min(image_size - 1, x1)), max(0, min(image_size - 1, x2))])
    y1, y2 = sorted([max(0, min(image_size - 1, y1)), max(0, min(image_size - 1, y2))])
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def load_manual_boxes(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if path is None:
        return {}
    data = read_json(path)
    views = data.get("views", data)
    if not isinstance(views, dict):
        raise ValueError("Manual boxes JSON must contain a 'views' object or be keyed by view name.")
    out: dict[str, list[dict[str, Any]]] = {}
    for view, boxes in views.items():
        if isinstance(boxes, dict):
            boxes = [boxes]
        if not isinstance(boxes, list):
            continue
        out[str(view)] = [box for box in boxes if isinstance(box, dict)]
    return out


def dry_run_boxes(view_entry: dict[str, Any], queries: list[str]) -> list[dict[str, Any]]:
    crop = view_entry.get("zoom_crop_bbox")
    if not crop or len(crop) < 4 or not queries:
        return []
    return [
        {
            "query": queries[0],
            "box": [int(v) for v in crop[:4]],
            "score": 0.0,
            "source": "dry_run_zoom_crop",
        }
    ]


def boxes_from_backend(args: argparse.Namespace, row: dict[str, str], view_entry: dict[str, Any], queries: list[str]) -> list[dict[str, Any]]:
    if args.backend == "manual-json":
        return []
    if args.backend == "grounding-dino":
        raise RuntimeError(
            "GroundingDINO backend is not wired in this repository yet. "
            "Run with --backend manual-json and --manual-boxes, or add a server-side GroundingDINO adapter here."
        )
    if args.backend == "florence2":
        raise RuntimeError(
            "Florence-2 backend is not wired in this repository yet. "
            "Run with --backend manual-json and --manual-boxes, or add a server-side Florence-2 adapter here."
        )
    raise ValueError(f"Unsupported backend: {args.backend}")


def rectangular_mask(shape: tuple[int, int], boxes: list[dict[str, Any]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    h, w = shape
    for item in boxes:
        box = normalize_box(item.get("box"), min(h, w))
        if box is None:
            continue
        x1, y1, x2, y2 = box
        mask[y1 : y2 + 1, x1 : x2 + 1] = True
    return mask


def sam2_mask_from_boxes(image: np.ndarray, boxes: list[dict[str, Any]], predictor: Any, image_size: int) -> np.ndarray:
    import torch

    combined = np.zeros(image.shape[:2], dtype=bool)
    if not boxes:
        return combined
    predictor.set_image(image)
    with torch.inference_mode():
        for item in boxes:
            box = normalize_box(item.get("box"), image_size)
            if box is None:
                continue
            masks, scores, _ = predictor.predict(box=np.asarray(box, dtype=np.float32), multimask_output=True)
            masks = np.asarray(masks)
            if masks.ndim == 2:
                combined |= masks > 0
            elif masks.ndim == 3:
                score_arr = np.asarray(scores) if scores is not None else np.zeros((masks.shape[0],), dtype=np.float32)
                combined |= masks[int(np.argmax(score_arr))] > 0
    return combined


def save_mask_png(mask: np.ndarray, path: Path) -> None:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[..., :] = np.array([14, 16, 20], dtype=np.uint8)
    rgb[mask > 0] = np.array([255, 90, 90], dtype=np.uint8)
    write_png_rgb(path, rgb)


def run_for_row(root: Path, args: argparse.Namespace, cfg: dict[str, Any], row: dict[str, str], manual_boxes: dict[str, list[dict[str, Any]]], predictor: Any) -> dict[str, Any]:
    pilot_id = row["pilot_id"]
    sample_id = row["sample_id"]
    manifest_path = resolve_path(root, args.renders_root) / sample_id / "view_manifest.json"
    part_plan_path = resolve_path(root, args.part_plan_root) / pilot_id / "combined_part_plan.json"
    manifest = read_json(manifest_path)
    part_plan = read_json(part_plan_path)
    queries = [str(item) for item in part_plan.get("grounding_queries", []) if str(item).strip()]
    output_dir = resolve_path(root, args.output_root) / pilot_id
    output_dir.mkdir(parents=True, exist_ok=True)

    view_results: list[dict[str, Any]] = []
    for entry in manifest.get("views", []):
        view = entry["view"]
        image_path = resolve_portable_path(root, entry.get("dense_render_path"), manifest_path.parent)
        if not image_path.exists():
            raise FileNotFoundError(f"Dense render not found: {image_path}")
        if args.validate_only:
            view_results.append({"view": view, "status": "validated"})
            continue

        if args.dry_run:
            boxes = dry_run_boxes(entry, queries)
        elif manual_boxes.get(view):
            boxes = manual_boxes[view]
        else:
            boxes = boxes_from_backend(args, row, entry, queries)

        image = np.asarray(Image.open(image_path).convert("RGB"))
        normalized_boxes: list[dict[str, Any]] = []
        for item in boxes:
            box = normalize_box(item.get("box"), image.shape[0])
            if box is None:
                continue
            normalized = dict(item)
            normalized["box"] = box
            normalized_boxes.append(normalized)

        if args.run_sam2:
            mask = sam2_mask_from_boxes(image, normalized_boxes, predictor, image.shape[0])
            mask_source = "sam2"
        elif args.box_mask_only or args.dry_run:
            mask = rectangular_mask(image.shape[:2], normalized_boxes)
            mask_source = "box_mask"
        else:
            mask = np.zeros(image.shape[:2], dtype=bool)
            mask_source = "boxes_only_no_mask"

        boxes_path = output_dir / f"{view}_boxes.json"
        mask_path = output_dir / f"{view}_mask.npy"
        mask_png_path = output_dir / f"{view}_mask.png"
        write_json(
            boxes_path,
            {
                "pilot_id": pilot_id,
                "sample_id": sample_id,
                "view": view,
                "queries": queries,
                "boxes": normalized_boxes,
                "mask_source": mask_source,
            },
            args.overwrite,
        )
        np.save(mask_path, mask.astype(np.uint8))
        save_mask_png(mask, mask_png_path)
        view_results.append(
            {
                "view": view,
                "boxes_path": relative_to_dataset(root, boxes_path),
                "mask_path": relative_to_dataset(root, mask_path),
                "positive_pixels": int(mask.sum()),
                "mask_source": mask_source,
            }
        )

    summary = {
        "pilot_id": pilot_id,
        "sample_id": sample_id,
        "object_category": row.get("object_category", ""),
        "task": row.get("task", ""),
        "executor": row.get("executor", ""),
        "part_plan": relative_to_dataset(root, part_plan_path),
        "views": view_results,
        "notes": "2D grounding output. It must be projected to 3D and reviewed.",
    }
    if not args.validate_only:
        write_json(output_dir / "grounding_summary.json", summary, args.overwrite)
    return summary


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    cfg = load_yaml(resolve_path(root, args.config))
    rows = selected_rows(root, args)
    manual_boxes = load_manual_boxes(resolve_path(root, args.manual_boxes) if args.manual_boxes else None)
    predictor = None
    if args.run_sam2 and not args.validate_only:
        predictor = load_sam2_predictor(cfg, root)
    outputs = [run_for_row(root, args, cfg, row, manual_boxes, predictor) for row in rows]
    print(json.dumps({"rows": len(outputs), "validate_only": args.validate_only, "outputs": outputs}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
