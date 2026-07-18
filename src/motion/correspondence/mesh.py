"""Delaunay triangulation (via OpenCV's Subdiv2D — no scipy dependency) and the
fixed frame-boundary points that make the mesh cover the *entire* canvas so the
background cross-dissolves cleanly while the object morphs."""

from __future__ import annotations

import cv2
import numpy as np


def frame_points(size: tuple[int, int], per_edge: int = 2) -> np.ndarray:
    """Corner + evenly-spaced edge points for an ``(H, W)`` canvas.

    These are identical in image A and B (the frame doesn't move), which anchors
    the triangulation to the full image rectangle.
    """
    h, w = size
    xs = np.linspace(0, w - 1, per_edge + 2)
    ys = np.linspace(0, h - 1, per_edge + 2)
    pts: list[tuple[float, float]] = []
    for x in xs:
        pts.append((x, 0))
        pts.append((x, h - 1))
    for y in ys:
        pts.append((0, y))
        pts.append((w - 1, y))
    uniq = {(round(x, 3), round(y, 3)) for x, y in pts}
    return np.array(sorted(uniq), dtype=np.float32)


def delaunay_triangles(points: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Delaunay triangulation of ``points``; returns (M, 3) index triples.

    Vertices returned by Subdiv2D are matched back to input indices by nearest
    neighbor (robust to the tiny coordinate rounding Subdiv2D introduces).
    """
    h, w = size
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    # Clamp strictly inside the rect Subdiv2D requires (x in [0, w), y in [0, h)).
    pts_clamped = pts.copy()
    pts_clamped[:, 0] = np.clip(pts_clamped[:, 0], 0, w - 1e-3)
    pts_clamped[:, 1] = np.clip(pts_clamped[:, 1], 0, h - 1e-3)

    subdiv = cv2.Subdiv2D((0, 0, w, h))
    for x, y in pts_clamped:
        subdiv.insert((float(x), float(y)))

    tri_list = subdiv.getTriangleList()
    triangles: list[tuple[int, int, int]] = []
    for t in tri_list:
        verts = t.reshape(3, 2)
        # Skip triangles touching Subdiv2D's virtual super-triangle (outside rect).
        if (
            np.any(verts[:, 0] < 0)
            or np.any(verts[:, 0] > w)
            or np.any(verts[:, 1] < 0)
            or np.any(verts[:, 1] > h)
        ):
            continue
        idx = tuple(int(np.argmin(np.sum((pts_clamped - v) ** 2, axis=1))) for v in verts)
        if len(set(idx)) == 3:
            triangles.append(idx)  # type: ignore[arg-type]
    if not triangles:
        raise ValueError("triangulation produced no triangles")
    return np.array(triangles, dtype=np.int32)
