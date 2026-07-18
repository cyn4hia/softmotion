"""Model-free segmentation fallback (GrabCut). Lets the CPU pipeline produce an
object mask — and therefore a mesh morph — with no SAM 3 weights present. Lower
quality than SAM 3, but keeps the whole pipeline runnable today."""

from __future__ import annotations

import cv2
import numpy as np

from motion.io.images import ensure_rgb


def grabcut_segment(
    image: np.ndarray,
    rect: tuple[int, int, int, int] | None = None,
    iterations: int = 5,
    margin: float = 0.08,
) -> np.ndarray:
    """Foreground mask via GrabCut. ``rect`` is (x, y, w, h); defaults to the image
    inset by ``margin`` on each side (assumes a roughly centered subject)."""
    img = cv2.cvtColor(ensure_rgb(image), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    if rect is None:
        mx, my = int(w * margin), int(h * margin)
        rect = (mx, my, w - 2 * mx, h - 2 * my)

    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, rect, bgd, fgd, iterations, cv2.GC_INIT_WITH_RECT)
    fg = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD))
    # keep the largest connected component to avoid speckle
    fg_u8 = fg.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg_u8, connectivity=8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        fg = labels == largest
    return fg
