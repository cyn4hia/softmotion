"""Assemble a :class:`Correspondence` from object masks (preferred) or fall back
to a mask-free grid (pure cross-dissolve) when no segmentation is available."""

from __future__ import annotations

import numpy as np

from motion.correspondence.base import Correspondence
from motion.correspondence.contour import align_contours, mask_to_contour, resample_contour
from motion.correspondence.mesh import delaunay_triangles, frame_points


def _inner_ring(contour: np.ndarray, shrink: float = 0.5) -> np.ndarray:
    """A scaled-in copy of the contour toward its centroid — cheap interior
    structure that stays inside the object so warps don't collapse."""
    c = contour.mean(axis=0)
    return (c + shrink * (contour - c)).astype(np.float32)


def correspondence_from_masks(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    size: tuple[int, int],
    n_contour: int = 64,
    add_interior: bool = True,
    per_edge: int = 3,
) -> Correspondence:
    """Build matched control points from two object masks.

    Points = fixed frame boundary (anchors the background) + aligned object
    outlines + an interior ring. Triangulated once on the midpoint positions so
    the topology is valid across the whole interpolation.
    """
    ca = resample_contour(mask_to_contour(mask_a), n_contour)
    cb = resample_contour(mask_to_contour(mask_b), n_contour)
    cb = align_contours(ca, cb)

    frame = frame_points(size, per_edge=per_edge)
    parts_a = [frame, ca]
    parts_b = [frame, cb]
    if add_interior:
        parts_a += [_inner_ring(ca), ca.mean(axis=0, keepdims=True)]
        parts_b += [_inner_ring(cb), cb.mean(axis=0, keepdims=True)]

    pts_a = np.concatenate(parts_a, axis=0).astype(np.float32)
    pts_b = np.concatenate(parts_b, axis=0).astype(np.float32)

    mid = 0.5 * (pts_a + pts_b)
    triangles = delaunay_triangles(mid, size)
    return Correspondence(points_a=pts_a, points_b=pts_b, triangles=triangles, size=size)


def correspondence_from_grid(
    size: tuple[int, int],
    nx: int = 8,
    ny: int = 8,
    displacement_b: np.ndarray | None = None,
) -> Correspondence:
    """Regular grid correspondence. With ``displacement_b`` (an (N,2) offset per
    grid point, e.g. sampled from optical flow) this becomes a warp; without it
    the points coincide and the classical backend degenerates to a clean
    cross-dissolve — the always-available fallback."""
    h, w = size
    xs = np.linspace(0, w - 1, nx)
    ys = np.linspace(0, h - 1, ny)
    grid = np.array([(x, y) for y in ys for x in xs], dtype=np.float32)
    pts_a = grid
    pts_b = grid.copy()
    if displacement_b is not None:
        pts_b = grid + np.asarray(displacement_b, dtype=np.float32).reshape(-1, 2)
    triangles = delaunay_triangles(0.5 * (pts_a + pts_b), size)
    return Correspondence(points_a=pts_a, points_b=pts_b, triangles=triangles, size=size)
