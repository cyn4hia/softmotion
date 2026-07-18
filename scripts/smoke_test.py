#!/usr/bin/env python3
"""End-to-end CPU smoke test: synthesize two objects, morph them with both backends,
and assert the outputs are sane. No models, no network — just proves the render path
works on this machine. Run via `make smoke`."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from motion.io.images import save_image  # noqa: E402
from motion.io.masks import save_mask  # noqa: E402
from motion.pipeline.merge import MergeConfig, MergePipeline  # noqa: E402

OUT = ROOT / "outputs" / "smoke"


def _circle(size=256):
    img = np.full((size, size, 3), 40, np.uint8)
    cv2.circle(img, (size // 2, size // 2), size // 3, (220, 60, 60), -1)
    mask = np.zeros((size, size), np.uint8)
    cv2.circle(mask, (size // 2, size // 2), size // 3, 255, -1)
    return img, mask > 0


def _square(size=256):
    img = np.full((size, size, 3), 40, np.uint8)
    s = size // 3
    c = size // 2
    cv2.rectangle(img, (c - s, c - s), (c + s, c + s), (60, 90, 230), -1)
    mask = np.zeros((size, size), np.uint8)
    cv2.rectangle(mask, (c - s, c - s), (c + s, c + s), 255, -1)
    return img, mask > 0


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    img_a, mask_a = _circle()
    img_b, mask_b = _square()
    save_image(OUT / "a.png", img_a)
    save_image(OUT / "b.png", img_b)
    save_mask(OUT / "mask_a.png", mask_a)
    save_mask(OUT / "mask_b.png", mask_b)

    ok = True
    for backend in ("classical", "flow"):
        cfg = MergeConfig(backend=backend, n_frames=24, fps=24, max_side=0)
        # perception off → masks supplied for classical; flow ignores them
        pipe = MergePipeline(cfg, segmenter=None, localizer=None)
        manifest = pipe.merge_pair(
            OUT / "a.png",
            OUT / "b.png",
            mask_a=str(OUT / "mask_a.png"),
            mask_b=str(OUT / "mask_b.png"),
            out_video=OUT / f"merge_{backend}.mp4",
        )
        video = Path(manifest.output_video)
        size_ok = video.exists() and video.stat().st_size > 0
        frames_ok = manifest.total_frames == 24
        print(f"[{backend}] video={video.name} exists={size_ok} frames={manifest.total_frames}")
        ok = ok and size_ok and frames_ok

    print("\nSMOKE TEST:", "PASS ✓" if ok else "FAIL ✗")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
