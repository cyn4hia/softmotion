"""Establish how object A maps onto object B: matched point pairs + a shared
Delaunay triangulation the morph engine warps across."""

from motion.correspondence.base import Correspondence
from motion.correspondence.builder import (
    correspondence_from_grid,
    correspondence_from_masks,
)
from motion.correspondence.contour import mask_to_contour, resample_contour
from motion.correspondence.mesh import delaunay_triangles, frame_points

__all__ = [
    "Correspondence",
    "correspondence_from_masks",
    "correspondence_from_grid",
    "mask_to_contour",
    "resample_contour",
    "delaunay_triangles",
    "frame_points",
]
