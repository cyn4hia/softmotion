"""Classical mesh-warp + cross-dissolve morphing (the Beier–Neely / triangle-mesh
family). Deterministic, CPU-only, and the default engine.

For each frame at parameter ``t`` the control points move to their interpolated
positions; every triangle of image A is affine-warped to that target, every
triangle of B likewise, and the two warped images are cross-dissolved with weight
``t``. Because the frame boundary is part of the mesh, the whole canvas warps — the
object deforms while the background dissolves.
"""

from __future__ import annotations

import cv2
import numpy as np

from motion.correspondence.base import Correspondence
from motion.io.images import common_canvas
from motion.morph.base import MorphBackend
from motion.morph.easing import schedule


def _triangle_area(pts: np.ndarray) -> float:
    (x1, y1), (x2, y2), (x3, y3) = pts
    return 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))


def _warp_triangle(src: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray, dst: np.ndarray) -> None:
    """Affine-warp one triangle of ``src`` into float accumulator ``dst`` in place."""
    t_src = np.asarray(t_src, dtype=np.float32)
    t_dst = np.asarray(t_dst, dtype=np.float32)
    if _triangle_area(t_src) < 1e-3 or _triangle_area(t_dst) < 1e-3:
        return

    x1, y1, w1, h1 = cv2.boundingRect(t_src)
    x2, y2, w2, h2 = cv2.boundingRect(t_dst)
    if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
        return

    Hs, Ws = src.shape[:2]
    Hd, Wd = dst.shape[:2]
    # Clamp the SOURCE rect to the source image. A source vertex with a negative
    # coordinate (reachable when off-frame correspondence points — e.g. an
    # optical-flow grid displacement — are warped) would otherwise make the slice
    # `src[y1:y1+h1, x1:x1+w1]` negative-index into the wrong region. warpAffine's
    # border reflection fills any part of the triangle that fell off-image.
    x1c, y1c = max(x1, 0), max(y1, 0)
    x1e, y1e = min(x1 + w1, Ws), min(y1 + h1, Hs)
    if x1e <= x1c or y1e <= y1c:
        return

    # Destination triangles live inside the frame (convex combo of in-frame pts),
    # but clamp defensively so a boundingRect that grazes the edge can't overflow.
    x2c, y2c = max(x2, 0), max(y2, 0)
    x2e, y2e = min(x2 + w2, Wd), min(y2 + h2, Hd)
    if x2e <= x2c or y2e <= y2c:
        return

    # Keep float32: subtracting a Python int list would upcast to float64 and
    # cv2.getAffineTransform requires exactly CV_32F.
    src_local = (t_src - np.array([x1c, y1c], dtype=np.float32)).astype(np.float32)
    dst_local = (t_dst - np.array([x2, y2], dtype=np.float32)).astype(np.float32)
    src_crop = src[y1c:y1e, x1c:x1e].astype(np.float32)
    if src_crop.size == 0:
        return

    matrix = cv2.getAffineTransform(src_local, dst_local)
    warped = cv2.warpAffine(
        src_crop,
        matrix,
        (w2, h2),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    mask = np.zeros((h2, w2), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(dst_local), 1.0, cv2.LINE_AA, 0)
    mask3 = mask[:, :, None]

    # Crop everything to the clamped region before compositing.
    oy0, ox0 = y2c - y2, x2c - x2
    oy1, ox1 = oy0 + (y2e - y2c), ox0 + (x2e - x2c)
    region = dst[y2c:y2e, x2c:x2e]
    m = mask3[oy0:oy1, ox0:ox1]
    region[:] = region * (1 - m) + warped[oy0:oy1, ox0:ox1] * m


def morph_pair(
    image_a: np.ndarray,
    image_b: np.ndarray,
    correspondence: Correspondence,
    t: float,
) -> np.ndarray:
    """Single morphed frame at parameter ``t`` in [0, 1]."""
    h, w = correspondence.size
    pts_t = correspondence.points_at(t)
    warp_a = np.zeros((h, w, 3), dtype=np.float32)
    warp_b = np.zeros((h, w, 3), dtype=np.float32)
    for tri in correspondence.triangles:
        _warp_triangle(image_a, correspondence.points_a[tri], pts_t[tri], warp_a)
        _warp_triangle(image_b, correspondence.points_b[tri], pts_t[tri], warp_b)
    out = (1.0 - t) * warp_a + t * warp_b
    return np.clip(out, 0, 255).astype(np.uint8)


class ClassicalMorph(MorphBackend):
    name = "classical"
    requires_correspondence = True

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
        if correspondence is None:
            raise ValueError("ClassicalMorph requires a correspondence")
        a, b = common_canvas(image_a, image_b)
        if a.shape[:2] != correspondence.size:
            raise ValueError(
                f"image size {a.shape[:2]} != correspondence size {correspondence.size}"
            )
        ts = schedule(n_frames, easing, include_endpoints)
        return [morph_pair(a, b, correspondence, float(t)) for t in ts]
