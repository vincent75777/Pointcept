#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
try:
    import pointops
    print("pointops:", pointops.__file__)
except Exception as exc:
    raise SystemExit(f"pointops import failed: {exc}")
PY
