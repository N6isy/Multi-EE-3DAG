#!/usr/bin/env python3
"""Fuse projected multi-view votes into a point-level mask.

The output can be either a full [N,4] mask or an updated copy of an existing
mask. This keeps the VLM pilot separate from the current checked v0.1 masks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


EXECUTOR_ORDER = ["gripper", "suction", "hook", "dexterous_hand"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse multi-view projected point scores.")
    parser.add_argument("--projection-npz", required=True, help="Output .npz from project_2d_masks_to_3d.py.")
    parser.add_argument("--output-mask", required=True, help="Output [N,4] mask .npy path.")
    parser.add_argument("--existing-mask", default=None, help="Optional existing [N,4] mask to update.")
    parser.add_argument(
        "--executor",
        choices=EXECUTOR_ORDER,
        default=None,
        help="Fuse only one executor channel. Omit to fuse all channels.",
    )
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Minimum projected score.")
    parser.add_argument("--min-visible", type=int, default=1, help="Minimum visible pixels per point.")
    parser.add_argument("--stats-json", default=None, help="Optional output statistics JSON path.")
    return parser.parse_args()


def load_projection(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Projection file not found: {path}")
    data = np.load(path, allow_pickle=True)
    if "scores" not in data or "visible" not in data:
        raise ValueError(f"Projection file must contain scores and visible arrays: {path}")
    scores = data["scores"].astype(np.float32)
    visible = data["visible"].astype(np.float32)
    if scores.ndim != 2 or scores.shape[1] != len(EXECUTOR_ORDER):
        raise ValueError(f"Expected scores shape [N,4], got {scores.shape}")
    if visible.shape[0] != scores.shape[0]:
        raise ValueError(f"visible length {visible.shape[0]} does not match scores N {scores.shape[0]}")
    return scores, visible


def main() -> int:
    args = parse_args()
    scores, visible = load_projection(Path(args.projection_npz))
    num_points = scores.shape[0]

    if args.existing_mask:
        mask = np.load(args.existing_mask)
        if mask.shape != (num_points, len(EXECUTOR_ORDER)):
            raise ValueError(f"Existing mask shape {mask.shape} does not match projection shape {(num_points, 4)}")
        fused = mask.astype(np.uint8).copy()
    else:
        fused = np.zeros((num_points, len(EXECUTOR_ORDER)), dtype=np.uint8)

    channels = [EXECUTOR_ORDER.index(args.executor)] if args.executor else list(range(len(EXECUTOR_ORDER)))
    visible_ok = visible >= args.min_visible
    for channel in channels:
        fused[:, channel] = ((scores[:, channel] >= args.score_threshold) & visible_ok).astype(np.uint8)

    output_path = Path(args.output_mask)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, fused)

    stats = {
        "projection_npz": args.projection_npz,
        "output_mask": str(output_path),
        "executor_order": EXECUTOR_ORDER,
        "channels_fused": [EXECUTOR_ORDER[idx] for idx in channels],
        "score_threshold": args.score_threshold,
        "min_visible": args.min_visible,
        "positive_points": {EXECUTOR_ORDER[idx]: int(fused[:, idx].sum()) for idx in channels},
        "visible_points": int((visible >= args.min_visible).sum()),
    }
    if args.stats_json:
        stats_path = Path(args.stats_json)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
            f.write("\n")
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
