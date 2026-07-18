"""Optional texture-based correspondence via ORB features. Useful when the two
objects share visual detail (logos, faces, products); merge these pairs with the
contour points for a richer mesh. Pure OpenCV, CPU."""

from __future__ import annotations

import cv2
import numpy as np


def orb_correspondence(
    image_a: np.ndarray,
    image_b: np.ndarray,
    max_points: int = 200,
    ratio: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """Return matched (points_a, points_b) via ORB + Lowe ratio test.

    Returns two ``(K, 2)`` float32 arrays (possibly empty if nothing matches).
    """
    ga = cv2.cvtColor(image_a, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(image_b, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=max_points * 4)
    ka, da = orb.detectAndCompute(ga, None)
    kb, db = orb.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 2 or len(kb) < 2:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = matcher.knnMatch(da, db, k=2)
    pa, pb = [], []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            pa.append(ka[m.queryIdx].pt)
            pb.append(kb[m.trainIdx].pt)
    if not pa:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    order = np.argsort([-1] * len(pa))  # keep insertion order; cap below
    pa_arr = np.array(pa, np.float32)[order][:max_points]
    pb_arr = np.array(pb, np.float32)[order][:max_points]
    return pa_arr, pb_arr
