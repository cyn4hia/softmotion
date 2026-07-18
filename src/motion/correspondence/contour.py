"""Contour extraction, uniform resampling, and rotation/direction alignment.

The object outline is the strongest correspondence signal we have without a
learned matcher, so we sample it densely and align the two outlines by trying
every cyclic start offset (and both winding directions) and keeping the lowest
sum-of-squared-distance pairing after centroid normalization.
"""

from __future__ import annotations

import cv2
import numpy as np


def mask_to_contour(mask: np.ndarray) -> np.ndarray:
    """Largest external contour of a binary mask, as (K, 2) float32 xy points."""
    m = np.asarray(mask).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("mask has no contour (is it empty?)")
    largest = max(contours, key=cv2.contourArea)
    return largest.reshape(-1, 2).astype(np.float32)


def resample_contour(contour: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed contour to exactly ``n`` points, uniform by arc length."""
    pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 0:
        # degenerate (single point) — just tile it
        return np.repeat(pts[:1], n, axis=0)
    targets = np.linspace(0.0, total, n, endpoint=False)
    x = np.interp(targets, cum, closed[:, 0])
    y = np.interp(targets, cum, closed[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def _normalized(pts: np.ndarray) -> np.ndarray:
    c = pts.mean(axis=0)
    centered = pts - c
    scale = np.sqrt((centered**2).sum(axis=1)).mean() + 1e-6
    return centered / scale


def align_contours(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Re-order/flip ``b`` (already resampled to len(a)) to best match ``a``.

    Returns a copy of ``b`` cyclically shifted and possibly reversed so that
    ``b[i]`` pairs with ``a[i]``. Comparison is scale/translation invariant.
    """
    if len(a) != len(b):
        raise ValueError("align_contours needs equal-length contours")
    na = _normalized(a)
    n = len(a)
    best_cost = np.inf
    best: np.ndarray = b
    for flip in (False, True):
        cand = b[::-1] if flip else b
        ncand = _normalized(cand)
        for shift in range(n):
            rolled = np.roll(ncand, -shift, axis=0)
            cost = float(np.sum((na - rolled) ** 2))
            if cost < best_cost:
                best_cost = cost
                best = np.roll(cand[::1], -shift, axis=0)
    return best.astype(np.float32)
