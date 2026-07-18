#!/usr/bin/env bash
# Provision a fresh CUDA box for fine-tuning SAM 3 + LocateAnything.
# Usage: bash scripts/setup_gpu.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY_VER="${PY_VER:-3.12}"           # SAM 3 requires Python >= 3.12
CUDA_WHL="${CUDA_WHL:-cu128}"      # match your driver (see docs/GPU_SETUP.md)

echo "==> Creating venv (python${PY_VER})"
python${PY_VER} -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip

echo "==> Installing PyTorch (${CUDA_WHL})"
pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_WHL}"

echo "==> Installing motion (gpu + train + data + dev)"
pip install -e ".[gpu,train,data,dev]"

echo "==> Cloning + installing SAM 3"
mkdir -p third_party
if [ ! -d third_party/sam3 ]; then
  git clone https://github.com/facebookresearch/sam3.git third_party/sam3
fi
pip install -e "third_party/sam3[train,dev]"

echo "==> Hugging Face auth (needed for gated SAM 3 checkpoints)"
if [ -n "${HF_TOKEN:-}" ]; then
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || true
else
  echo "   Set HF_TOKEN (see .env.example) then run: huggingface-cli login"
  echo "   Request checkpoint access: https://huggingface.co/facebook/sam3"
fi

python scripts/bootstrap_dirs.py
echo
echo "==> Done. Verify with:  motion doctor"
echo "    Then follow docs/FINETUNING.md"
