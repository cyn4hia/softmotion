#!/usr/bin/env python3
"""Fetch model assets.

- LocateAnything GGUF for the Mac (from mudler/locate-anything.cpp-gguf).
- Prints instructions for the *gated* SAM 3 checkpoints (manual access request).

Usage:
    python scripts/download_models.py --gguf q8_0
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GGUF_REPO = "mudler/locate-anything.cpp-gguf"
GGUF_FILES = {
    "f16": "locate-anything-f16.gguf",
    "q8_0": "locate-anything-q8_0.gguf",  # ~6.3GB, box-identical to f32 (recommended)
    "q6_k": "locate-anything-q6_k.gguf",
    "q5_k": "locate-anything-q5_k.gguf",
    "q4_k": "locate-anything-q4_k.gguf",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", choices=list(GGUF_FILES), default="q8_0")
    ap.add_argument("--out", default=str(ROOT / "checkpoints" / "gguf"))
    args = ap.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        raise SystemExit("pip install huggingface_hub  (or `pip install -e '.[gpu]'`)") from e

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fname = GGUF_FILES[args.gguf]
    print(f"downloading {GGUF_REPO}/{fname} → {out}")
    path = hf_hub_download(repo_id=GGUF_REPO, filename=fname, local_dir=str(out))
    print("✓", path)

    print(
        "\nSAM 3 checkpoints are GATED. To download:\n"
        "  1. Request access: https://huggingface.co/facebook/sam3\n"
        "  2. `huggingface-cli login` with your HF_TOKEN\n"
        "  3. They download automatically on first `Sam3Segmenter` use, or:\n"
        "     huggingface-cli download facebook/sam3 --local-dir checkpoints/sam3\n"
    )


if __name__ == "__main__":
    main()
