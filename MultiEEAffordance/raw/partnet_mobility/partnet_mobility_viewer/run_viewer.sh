#!/usr/bin/env bash
set -euo pipefail
ZIP_PATH="${1:-${PARTNET_ZIP:-/home/lzq/Multi-EE-3DAG/MultiEEAffordance/raw/partnet_mobility/partnet-mobility-v0.zip}}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-7860}"
python app.py --zip "$ZIP_PATH" --host "$HOST" --port "$PORT"
