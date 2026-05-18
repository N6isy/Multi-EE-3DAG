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
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from path_utils import relative_to_dataset, resolve_portable_path
from render_multiview import write_png_rgb
from run_qwen3vl_sam2_pilot import load_sam2_predictor, load_yaml


class Florence2Grounder:
    """Small wrapper around Florence-2 phrase grounding."""

    def __init__(self, model: Any, processor: Any, device: str, torch_dtype: Any, cfg: dict[str, Any]):
        self.model = model
        self.processor = processor
        self.device = device
        self.torch_dtype = torch_dtype
        self.cfg = cfg


FLORENCE2_GENERATION_DEFAULTS = {
    "forced_bos_token_id": None,
    "forced_eos_token_id": None,
    "suppress_tokens": None,
    "begin_suppress_tokens": None,
}


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
        help="Grounding backend. Florence-2 is wired; GroundingDINO remains an explicit integration slot.",
    )
    parser.add_argument(
        "--manual-boxes",
        default=None,
        help="Optional JSON with per-view boxes. Format: {'views': {'view': [{'query': str, 'box': [x1,y1,x2,y2], 'score': 0.9}]}}.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Use zoom-crop boxes as placeholder boxes.")
    parser.add_argument("--run-sam2", action="store_true", help="Run SAM2 on grounded boxes.")
    parser.add_argument("--box-mask-only", action="store_true", help="Use rectangular box masks instead of SAM2 masks.")
    parser.add_argument("--max-boxes-per-query", type=int, default=3, help="Maximum Florence-2 boxes kept per query/view.")
    parser.add_argument("--min-box-area", type=int, default=16, help="Minimum accepted 2D box area in pixels.")
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


def resolve_hf_model_source(root: Path, section: dict[str, Any], default_model_id: str, name: str) -> str:
    model_path = section.get("model_path")
    if model_path:
        path = Path(str(model_path))
        resolved = path if path.is_absolute() else root / path
        if not resolved.exists():
            raise FileNotFoundError(
                f"{name} local model_path does not exist: {resolved}. "
                "Download/transfer the model directory first, or use model_id when the server has network access."
            )
        if not (resolved / "config.json").exists():
            raise FileNotFoundError(f"{name} model_path is missing config.json: {resolved}")
        return str(resolved)
    return str(section.get("model_id", default_model_id))


def torch_dtype_from_config(torch_module: Any, value: Any) -> Any:
    text = str(value or "auto").lower()
    if text in {"float16", "fp16", "half"}:
        return torch_module.float16
    if text in {"bfloat16", "bf16"}:
        return torch_module.bfloat16
    if text in {"float32", "fp32"}:
        return torch_module.float32
    return torch_module.float16 if torch_module.cuda.is_available() else torch_module.float32


def ensure_attr(obj: Any, name: str, value: Any) -> None:
    """Set a missing config attribute without overwriting existing values."""
    if obj is None:
        return
    try:
        getattr(obj, name)
    except Exception:
        setattr(obj, name, value)


def collect_config_objects(obj: Any, max_depth: int = 3) -> list[Any]:
    """Collect config-like objects nested inside a model or config tree."""
    out: list[Any] = []
    seen: set[int] = set()

    def visit(value: Any, depth: int) -> None:
        if value is None or depth > max_depth:
            return
        value_id = id(value)
        if value_id in seen:
            return
        seen.add(value_id)

        class_name = value.__class__.__name__.lower()
        is_config_like = "config" in class_name or hasattr(value, "to_dict")
        if is_config_like:
            out.append(value)

        for attr in (
            "config",
            "generation_config",
            "text_config",
            "language_config",
            "vision_config",
            "decoder",
            "decoder_config",
            "encoder",
            "encoder_config",
            "language_model",
            "model",
        ):
            try:
                child = getattr(value, attr)
            except Exception:
                continue
            visit(child, depth + 1)

    visit(obj, 0)
    return out


def patch_florence2_generation_config(target: Any) -> None:
    """Patch Florence-2 config fields required by some transformers versions.

    Some Florence-2 checkpoints / remote-code combinations expose
    Florence2LanguageConfig without every generation attribute expected by newer
    transformers generation utilities. The missing `forced_bos_token_id` field
    has been observed on the remote server. Adding these attributes with neutral
    defaults keeps generation behavior unchanged while avoiding AttributeError.
    """
    for config in collect_config_objects(target):
        config_class = config.__class__
        for name, value in FLORENCE2_GENERATION_DEFAULTS.items():
            if not hasattr(config_class, name):
                setattr(config_class, name, value)
            ensure_attr(config, name, value)


def patch_florence2_config_classes(config_class: Any) -> None:
    """Patch Florence-2 config classes before any config instance is created."""
    module = sys.modules.get(getattr(config_class, "__module__", ""))
    candidate_classes = [config_class]
    if module is not None:
        for class_name in ("Florence2Config", "Florence2LanguageConfig", "Florence2VisionConfig"):
            candidate = getattr(module, class_name, None)
            if candidate is not None:
                candidate_classes.append(candidate)

    seen: set[int] = set()
    for candidate in candidate_classes:
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        for name, value in FLORENCE2_GENERATION_DEFAULTS.items():
            if not hasattr(candidate, name):
                setattr(candidate, name, value)


def load_florence2_config_class(model_source: str, shared_kwargs: dict[str, Any]) -> Any | None:
    """Load the remote Florence-2 config class without instantiating it.

    `AutoConfig.from_pretrained` instantiates Florence2LanguageConfig internally.
    Some Florence-2 remote-code revisions access `self.forced_bos_token_id`
    before PretrainedConfig has populated it. Loading the class first lets us
    add neutral class defaults before the first instance is created.
    """
    config_path = Path(model_source) / "config.json"
    if not config_path.exists():
        return None
    with config_path.open("r", encoding="utf-8") as f:
        config_dict = json.load(f)
    auto_map = config_dict.get("auto_map", {})
    config_ref = auto_map.get("AutoConfig") if isinstance(auto_map, dict) else None
    if isinstance(config_ref, (list, tuple)):
        config_ref = config_ref[0]
    if not config_ref:
        return None

    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    dynamic_kwargs = {
        key: value
        for key, value in shared_kwargs.items()
        if key in {"cache_dir", "revision", "local_files_only"}
    }
    return get_class_from_dynamic_module(str(config_ref), model_source, **dynamic_kwargs)


def load_florence2_auto_class(model_source: str, shared_kwargs: dict[str, Any], auto_key: str) -> Any | None:
    """Load a Florence-2 remote auto class without instantiating it."""
    config_path = Path(model_source) / "config.json"
    if not config_path.exists():
        return None
    with config_path.open("r", encoding="utf-8") as f:
        config_dict = json.load(f)
    auto_map = config_dict.get("auto_map", {})
    class_ref = auto_map.get(auto_key) if isinstance(auto_map, dict) else None
    if isinstance(class_ref, (list, tuple)):
        class_ref = class_ref[0]
    if not class_ref:
        return None

    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    dynamic_kwargs = {
        key: value
        for key, value in shared_kwargs.items()
        if key in {"cache_dir", "revision", "local_files_only"}
    }
    return get_class_from_dynamic_module(str(class_ref), model_source, **dynamic_kwargs)


def patch_florence2_model_class(model_class: Any) -> None:
    """Patch remote Florence-2 model class attributes expected by transformers."""
    if model_class is None:
        return
    # Newer transformers queries this class attribute during attention dispatch.
    # Older Florence-2 remote-code revisions may not define it.
    if not hasattr(model_class, "_supports_sdpa"):
        setattr(model_class, "_supports_sdpa", False)
    if not hasattr(model_class, "_supports_flash_attn_2"):
        setattr(model_class, "_supports_flash_attn_2", False)


def patch_tokenizer_additional_special_tokens() -> None:
    """Provide tokenizer.additional_special_tokens for Florence-2 processors.

    Some Florence-2 processor revisions access `tokenizer.additional_special_tokens`
    directly. In newer transformers/tokenizer combinations that attribute can be
    absent even though the same information is available from the tokenizer's
    special-token maps. A base-class property keeps the remote processor code
    compatible without mutating the vocabulary.
    """
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    except Exception:
        return

    if isinstance(getattr(PreTrainedTokenizerBase, "additional_special_tokens", None), property):
        return

    def get_additional_special_tokens(tokenizer: Any) -> list[Any]:
        for attr in ("special_tokens_map_extended", "special_tokens_map"):
            try:
                mapping = getattr(tokenizer, attr)
            except Exception:
                continue
            if isinstance(mapping, dict) and mapping.get("additional_special_tokens") is not None:
                tokens = mapping["additional_special_tokens"]
                return list(tokens) if isinstance(tokens, (list, tuple)) else [tokens]
        return list(getattr(tokenizer, "_additional_special_tokens", []))

    def set_additional_special_tokens(tokenizer: Any, value: Any) -> None:
        tokenizer.__dict__["_additional_special_tokens"] = list(value or [])

    setattr(
        PreTrainedTokenizerBase,
        "additional_special_tokens",
        property(get_additional_special_tokens, set_additional_special_tokens),
    )


def apply_attn_implementation(config: Any, value: str | None) -> None:
    """Apply attention implementation to all nested config-like objects."""
    if not value:
        return
    for config_obj in collect_config_objects(config):
        for attr in ("_attn_implementation", "_attn_implementation_internal", "attn_implementation"):
            try:
                setattr(config_obj, attr, value)
            except Exception:
                continue


def load_florence2_grounder(cfg: dict[str, Any], root: Path) -> Florence2Grounder:
    florence_cfg = cfg.get("florence2", {})
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
    except Exception as exc:
        raise RuntimeError(
            "Florence-2 dependencies are missing. Install latest transformers, torch, accelerate, and pillow."
        ) from exc

    model_source = resolve_hf_model_source(root, florence_cfg, "microsoft/Florence-2-large", "Florence-2")
    shared_kwargs: dict[str, Any] = {}
    if florence_cfg.get("cache_dir"):
        shared_kwargs["cache_dir"] = str(resolve_path(root, florence_cfg["cache_dir"]))
    if florence_cfg.get("revision"):
        shared_kwargs["revision"] = florence_cfg["revision"]
    if florence_cfg.get("local_files_only") is not None:
        shared_kwargs["local_files_only"] = bool(florence_cfg["local_files_only"])
    trust_remote_code = bool(florence_cfg.get("trust_remote_code", True))
    torch_dtype = torch_dtype_from_config(torch, florence_cfg.get("dtype", "auto"))
    device = str(florence_cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"))
    attn_implementation = florence_cfg.get("attn_implementation", "eager")
    if str(attn_implementation).lower() in {"", "none", "null"}:
        attn_implementation = "eager"
    attn_implementation = str(attn_implementation)

    config_class = load_florence2_config_class(model_source, shared_kwargs) if trust_remote_code else None
    if config_class is not None:
        patch_florence2_config_classes(config_class)
        model_config = config_class.from_pretrained(model_source, **shared_kwargs)
    else:
        model_config = AutoConfig.from_pretrained(model_source, trust_remote_code=trust_remote_code, **shared_kwargs)
    patch_florence2_generation_config(model_config)
    apply_attn_implementation(model_config, attn_implementation)
    model_class = load_florence2_auto_class(model_source, shared_kwargs, "AutoModelForCausalLM") if trust_remote_code else None
    patch_florence2_model_class(model_class)

    model_kwargs = {
        "config": model_config,
        "trust_remote_code": trust_remote_code,
        "torch_dtype": torch_dtype,
        "attn_implementation": attn_implementation,
        **shared_kwargs,
    }
    model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs).to(device)
    patch_florence2_generation_config(model)
    patch_tokenizer_additional_special_tokens()
    processor = AutoProcessor.from_pretrained(model_source, trust_remote_code=trust_remote_code, **shared_kwargs)
    model.eval()
    return Florence2Grounder(model=model, processor=processor, device=device, torch_dtype=torch_dtype, cfg=florence_cfg)


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


def parse_florence_boxes(parsed: Any, task_prompt: str, query: str, image_size: tuple[int, int], max_boxes: int) -> list[dict[str, Any]]:
    payload = parsed.get(task_prompt, parsed) if isinstance(parsed, dict) else {}
    if not isinstance(payload, dict):
        return []
    bboxes = payload.get("bboxes", payload.get("boxes", []))
    labels = payload.get("labels", [])
    scores = payload.get("scores", [])
    out: list[dict[str, Any]] = []
    for idx, box in enumerate(bboxes if isinstance(bboxes, list) else []):
        label = labels[idx] if isinstance(labels, list) and idx < len(labels) else query
        score = scores[idx] if isinstance(scores, list) and idx < len(scores) else None
        out.append(
            {
                "query": query,
                "label": str(label),
                "box": box,
                "score": float(score) if isinstance(score, (int, float)) else None,
                "source": "florence2",
                "image_size": [int(image_size[0]), int(image_size[1])],
            }
        )
        if len(out) >= max_boxes:
            break
    return out


def florence2_ground_queries(grounder: Florence2Grounder, image: Image.Image, queries: list[str], args: argparse.Namespace) -> list[dict[str, Any]]:
    import torch

    task_prompt = str(grounder.cfg.get("task_prompt", "<CAPTION_TO_PHRASE_GROUNDING>"))
    max_new_tokens = int(grounder.cfg.get("max_new_tokens", 1024))
    num_beams = int(grounder.cfg.get("num_beams", 3))
    boxes: list[dict[str, Any]] = []
    width, height = image.size

    for query in queries:
        prompt = f"{task_prompt}{query}"
        inputs = grounder.processor(text=prompt, images=image, return_tensors="pt")
        moved: dict[str, Any] = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                moved[key] = value.to(grounder.device)
                if key == "pixel_values" and moved[key].is_floating_point():
                    moved[key] = moved[key].to(grounder.torch_dtype)
            else:
                moved[key] = value
        with torch.inference_mode():
            patch_florence2_generation_config(grounder.model)
            generated_ids = grounder.model.generate(
                input_ids=moved.get("input_ids"),
                pixel_values=moved.get("pixel_values"),
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
            )
        generated_text = grounder.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = grounder.processor.post_process_generation(generated_text, task=task_prompt, image_size=(width, height))
        parsed_boxes = parse_florence_boxes(parsed, task_prompt, query, (width, height), args.max_boxes_per_query)
        for item in parsed_boxes:
            item["raw_text"] = generated_text
        boxes.extend(parsed_boxes)
    return boxes


def boxes_from_backend(
    args: argparse.Namespace,
    row: dict[str, str],
    view_entry: dict[str, Any],
    queries: list[str],
    image: Image.Image,
    grounder: Florence2Grounder | None,
) -> list[dict[str, Any]]:
    if args.backend == "manual-json":
        return []
    if args.backend == "grounding-dino":
        raise RuntimeError(
            "GroundingDINO backend is not wired in this repository yet. "
            "Run with --backend manual-json and --manual-boxes, or add a server-side GroundingDINO adapter here."
        )
    if args.backend == "florence2":
        if grounder is None:
            raise RuntimeError("Florence-2 backend requested but the grounder was not loaded.")
        return florence2_ground_queries(grounder, image, queries, args)
    raise ValueError(f"Unsupported backend: {args.backend}")


def rectangular_mask(shape: tuple[int, int], boxes: list[dict[str, Any]], min_area: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    h, w = shape
    for item in boxes:
        box = normalize_box(item.get("box"), min(h, w))
        if box is None:
            continue
        x1, y1, x2, y2 = box
        if (x2 - x1) * (y2 - y1) < min_area:
            continue
        mask[y1 : y2 + 1, x1 : x2 + 1] = True
    return mask


def sam2_mask_from_boxes(image: np.ndarray, boxes: list[dict[str, Any]], predictor: Any, image_size: int, min_area: int) -> np.ndarray:
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
            if (box[2] - box[0]) * (box[3] - box[1]) < min_area:
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


def run_for_row(
    root: Path,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    row: dict[str, str],
    manual_boxes: dict[str, list[dict[str, Any]]],
    predictor: Any,
    grounder: Florence2Grounder | None,
) -> dict[str, Any]:
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
        image_pil = Image.open(image_path).convert("RGB")
        image = np.asarray(image_pil)

        if args.dry_run:
            boxes = dry_run_boxes(entry, queries)
        elif manual_boxes.get(view):
            boxes = manual_boxes[view]
        else:
            boxes = boxes_from_backend(args, row, entry, queries, image_pil, grounder)

        normalized_boxes: list[dict[str, Any]] = []
        for item in boxes:
            box = normalize_box(item.get("box"), image.shape[0])
            if box is None:
                continue
            if (box[2] - box[0]) * (box[3] - box[1]) < args.min_box_area:
                continue
            normalized = dict(item)
            normalized["box"] = box
            normalized_boxes.append(normalized)

        if args.run_sam2:
            mask = sam2_mask_from_boxes(image, normalized_boxes, predictor, image.shape[0], args.min_box_area)
            mask_source = "sam2"
        elif args.box_mask_only or args.dry_run:
            mask = rectangular_mask(image.shape[:2], normalized_boxes, args.min_box_area)
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
                "backend": args.backend,
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
                "boxes": len(normalized_boxes),
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
    grounder = None
    if args.backend == "florence2" and not args.validate_only and not args.dry_run:
        grounder = load_florence2_grounder(cfg, root)
    outputs = [run_for_row(root, args, cfg, row, manual_boxes, predictor, grounder) for row in rows]
    print(json.dumps({"rows": len(outputs), "validate_only": args.validate_only, "outputs": outputs}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(2)
