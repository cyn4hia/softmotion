"""Blending helpers for the final "merge" moment — feathered alpha compositing and
Poisson (seamless) cloning so the two objects fuse without a hard seam."""

from __future__ import annotations

import cv2
import numpy as np

from motion.io.images import ensure_rgb


def cross_dissolve(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    a = ensure_rgb(a).astype(np.float32)
    b = ensure_rgb(b).astype(np.float32)
    return np.clip((1 - t) * a + t * b, 0, 255).astype(np.uint8)


def feather_mask(mask: np.ndarray, radius: int = 9) -> np.ndarray:
    """Soft-edged alpha in [0, 1] from a binary mask."""
    m = (np.asarray(mask).astype(np.float32)) * 255.0
    if radius > 0:
        k = radius * 2 + 1
        m = cv2.GaussianBlur(m, (k, k), 0)
    return np.clip(m / 255.0, 0, 1)


def alpha_composite(
    foreground: np.ndarray, background: np.ndarray, alpha: np.ndarray
) -> np.ndarray:
    """Composite ``foreground`` over ``background`` with a soft ``alpha`` (H, W)."""
    fg = ensure_rgb(foreground).astype(np.float32)
    bg = ensure_rgb(background).astype(np.float32)
    a = np.asarray(alpha, dtype=np.float32)
    if a.ndim == 2:
        a = a[:, :, None]
    return np.clip(fg * a + bg * (1 - a), 0, 255).astype(np.uint8)


def seamless_clone(
    src: np.ndarray, dst: np.ndarray, mask: np.ndarray, center: tuple[int, int] | None = None
) -> np.ndarray:
    """Poisson-blend ``src`` into ``dst`` over ``mask`` (cv2.seamlessClone).

    cv2 builds a mask-bbox-sized ROI centered at ``center`` and raises when it
    exceeds ``dst`` (common for objects hugging a frame edge/corner). We fall back
    to a feathered alpha composite in that case so the merge never crashes.
    """
    s = cv2.cvtColor(ensure_rgb(src), cv2.COLOR_RGB2BGR)
    d = cv2.cvtColor(ensure_rgb(dst), cv2.COLOR_RGB2BGR)
    m = (np.asarray(mask).astype(np.uint8)) * 255
    if center is None:
        ys, xs = np.where(m > 0)
        if len(xs) == 0:
            return ensure_rgb(dst)
        center = (int(xs.mean()), int(ys.mean()))
    try:
        out = cv2.seamlessClone(s, d, m, center, cv2.NORMAL_CLONE)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    except cv2.error:
        return alpha_composite(src, dst, feather_mask(np.asarray(mask).astype(bool)))
