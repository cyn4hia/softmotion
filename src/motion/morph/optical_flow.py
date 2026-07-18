"""Optical-flow morphing — no correspondence needed. Estimates dense motion both
ways with OpenCV's DIS optical flow and warps each image toward the other, then
cross-dissolves. Great mask-free fallback and strong for same-object motion; less
controlled than the mesh morph for wildly different shapes.
"""

from __future__ import annotations

import cv2
import numpy as np

from motion.correspondence.base import Correspondence
from motion.io.images import common_canvas
from motion.morph.base import MorphBackend
from motion.morph.easing import schedule


def _warp_by_flow(image: np.ndarray, flow: np.ndarray, alpha: float) -> np.ndarray:
    """Sample ``image`` along ``alpha``-scaled flow (backward remap)."""
    h, w = image.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x + alpha * flow[..., 0]
    map_y = grid_y + alpha * flow[..., 1]
    return cv2.remap(
        image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )


class OpticalFlowMorph(MorphBackend):
    name = "flow"
    requires_correspondence = False
    MIN_SIZE = 24  # DIS optical flow needs at least this many px per side

    @staticmethod
    def _cross_dissolve(
        a: np.ndarray, b: np.ndarray, n_frames: int, easing: str, include_endpoints: bool
    ) -> list[np.ndarray]:
        af, bf = a.astype(np.float32), b.astype(np.float32)
        out = []
        for t in schedule(n_frames, easing, include_endpoints):
            t = float(t)
            out.append(np.clip((1.0 - t) * af + t * bf, 0, 255).astype(np.uint8))
        return out

    def render(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        correspondence: Correspondence | None = None,
        n_frames: int = 48,
        easing: str = "ease_in_out",
        include_endpoints: bool = True,
        **kwargs,
    ) -> list[np.ndarray]:
        a, b = common_canvas(image_a, image_b)
        # DIS optical flow needs a minimum working resolution; below it, calc()
        # raises. Fall back to a plain cross-dissolve rather than crash.
        if min(a.shape[:2]) < self.MIN_SIZE:
            return self._cross_dissolve(a, b, n_frames, easing, include_endpoints)

        ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        flow_ab = dis.calc(ga, gb, None)  # motion a -> b
        flow_ba = dis.calc(gb, ga, None)  # motion b -> a

        frames: list[np.ndarray] = []
        for t in schedule(n_frames, easing, include_endpoints):
            t = float(t)
            wa = _warp_by_flow(a, flow_ab, t).astype(np.float32)
            wb = _warp_by_flow(b, flow_ba, 1.0 - t).astype(np.float32)
            out = (1.0 - t) * wa + t * wb
            frames.append(np.clip(out, 0, 255).astype(np.uint8))
        return frames
