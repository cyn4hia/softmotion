"""Binary mask I/O and visualization. Masks are boolean ``(H, W)`` arrays."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from motion.io.images import ensure_rgb


def save_mask(path: str | Path, mask: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img = (np.asarray(mask).astype(bool).astype(np.uint8)) * 255
    if not cv2.imwrite(str(path), img):
        raise OSError(f"failed to write mask: {path}")


def load_mask(path: str | Path) -> np.ndarray:
    data = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if data is None:
        raise FileNotFoundError(f"could not read mask: {path}")
    return data > 127


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 128),
    alpha: float = 0.5,
) -> np.ndarray:
    """Tint the masked region of an image for quick visual sanity checks."""
    img = ensure_rgb(image).astype(np.float32)
    m = np.asarray(mask).astype(bool)
    tint = np.array(color, dtype=np.float32)
    img[m] = (1 - alpha) * img[m] + alpha * tint
    return np.clip(img, 0, 255).astype(np.uint8)
