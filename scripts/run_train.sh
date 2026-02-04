#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
python tools/train.py --config-file configs/3dp/cls-3dp-ptv3-v1m1-0-run.py
