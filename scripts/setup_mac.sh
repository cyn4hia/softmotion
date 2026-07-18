#!/usr/bin/env bash
# Set up a Mac (mini) for RENDERING: CPU core + LocateAnything via ggml (Metal).
# Usage: bash scripts/setup_mac.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Creating venv"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip

echo "==> Installing motion (CPU core)"
pip install -e .

echo "==> Building locate-anything.cpp (Metal)"
mkdir -p third_party
if [ ! -d third_party/locate-anything.cpp ]; then
  git clone --recursive https://github.com/mudler/locate-anything.cpp third_party/locate-anything.cpp
fi
pushd third_party/locate-anything.cpp >/dev/null
cmake -B build -DLA_BUILD_CLI=ON -DLA_GGML_METAL=ON
cmake --build build -j
popd >/dev/null

CLI_PATH="$REPO_ROOT/third_party/locate-anything.cpp/build/bin/locate-anything-cli"
echo "==> locate-anything-cli → $CLI_PATH"

echo "==> Fetching a GGUF model (q8_0)"
pip install huggingface_hub >/dev/null
python scripts/download_models.py --gguf q8_0 || echo "   (skip/download later)"

python scripts/bootstrap_dirs.py

cat <<EOF

==> Done. Add these to your .env (or shell profile):
    LOCATE_ANYTHING_CLI=$CLI_PATH
    LOCATE_ANYTHING_GGUF=$REPO_ROOT/checkpoints/gguf/locate-anything-q8_0.gguf

Verify:   motion doctor
Render:   motion merge a.png b.png -o outputs/merge.mp4 --prompt-a "cat" --prompt-b "dog"
Smoke:    make smoke
EOF
