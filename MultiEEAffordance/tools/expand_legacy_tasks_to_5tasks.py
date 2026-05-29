#!/usr/bin/env python3
"""Compatibility entry point for expanding legacy v3 candidate rows to five tasks."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).with_name("expand_v3_candidate_tasks.py")
    runpy.run_path(str(target), run_name="__main__")
