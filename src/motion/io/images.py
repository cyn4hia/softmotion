"""Image loading/saving. Convention: images are ``np.uint8`` arrays in **RGB**
order, shape ``(H, W, 3)``. (OpenCV is BGR internally; we convert at the edges so
the rest of the codebase never has to think about channel order.)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as RGB uint8 (H, W, 3)."""
    data = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if data is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cv2.cvtColor(data, cv2.COLOR_BGR2RGB)


def save_image(path: str | Path, image: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(ensure_rgb(image), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise OSError(f"failed to write image: {path}")


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Coerce to contiguous uint8 RGB (H, W, 3)."""
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        pass
    else:
        raise ValueError(f"unsupported image shape {arr.shape}")
    return np.ascontiguousarray(arr)


def resize_to_match(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize to ``(H, W)`` using area interp for down-, cubic for up-scaling."""
    h, w = size
    ih, iw = image.shape[:2]
    if (ih, iw) == (h, w):
        return image
    interp = cv2.INTER_AREA if (h * w) < (ih * iw) else cv2.INTER_CUBIC
    return cv2.resize(image, (w, h), interpolation=interp)


def common_canvas(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Put two images on a common (max) canvas size so the morph mesh lines up.

    We resize B to A's dimensions. (A letterbox variant lives in the pipeline for
    when aspect ratios differ wildly; matching sizes is the common case.)
    """
    a = ensure_rgb(a)
    b = ensure_rgb(b)
    if a.shape[:2] != b.shape[:2]:
        b = resize_to_match(b, a.shape[:2])
    return a, b
