#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="pointcept3dp"

if ! command -v conda >/dev/null 2>&1; then
  echo "❌ conda not found. Install Miniconda/Anaconda first."
  exit 1
fi

if conda env list | awk '{print $1}' | grep -q "^${ENV_NAME}$"; then
  echo "✅ Using existing conda env: ${ENV_NAME}"
else
  echo "🧰 Creating conda env: ${ENV_NAME}"
  conda create -y -n "${ENV_NAME}" python=3.10
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip
python -m pip install numpy pandas tqdm rich wandb pyyaml
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
echo "✅ Environment ready. PYTHONPATH set."
