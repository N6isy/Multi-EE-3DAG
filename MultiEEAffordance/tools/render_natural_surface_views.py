#!/usr/bin/env python3
"""First-class v3 entrypoint for naturalized point-cloud surface renders.

The implementation lives in ``experiments/natural_surface_vlm`` because it was
developed as an isolated probe first.  This wrapper promotes it into the v3
toolchain with production defaults while keeping the renderer code in one
place.
"""

from __future__ import annotations

import sys
from pathlib import Path


DATASET_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = DATASET_ROOT / "experiments" / "natural_surface_vlm"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from render_natural_surface_views import main as natural_main  # noqa: E402


def ensure_default(flag: str, value: str | None = None) -> None:
    if flag in sys.argv:
        return
    sys.argv.append(flag)
    if value is not None:
        sys.argv.append(value)


def apply_v3_defaults() -> None:
    ensure_default("--output-root", "processed/vlm_candidate_v3/natural_renders")
    ensure_default("--image-size", "768")
    ensure_default("--splat-radius", "10")
    ensure_default("--fill-radius", "10")
    ensure_default("--blur-radius", "1.4")
    ensure_default("--edge-mode", "silhouette")
    ensure_default("--densify-threshold-multiplier", "2.2")
    ensure_default("--densify-max-neighbors", "2")
    ensure_default("--densify-midpoints")


if __name__ == "__main__":
    apply_v3_defaults()
    raise SystemExit(natural_main())
