import numpy as np

from motion.correspondence.builder import correspondence_from_masks
from motion.morph.classical import ClassicalMorph


def _mad(a, b):
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def test_classical_endpoints_and_count(circle, square):
    img_a, ma = circle
    img_b, mb = square
    corr = correspondence_from_masks(ma, mb, img_a.shape[:2], n_contour=64)
    frames = ClassicalMorph().render(img_a, img_b, corr, n_frames=12)

    assert len(frames) == 12
    assert all(f.shape == img_a.shape and f.dtype == np.uint8 for f in frames)
    # Frame 0 ≈ A, last ≈ B (identity warp reconstructs the source).
    assert _mad(frames[0], img_a) < 8.0
    assert _mad(frames[-1], img_b) < 8.0


def test_classical_midframe_is_a_blend(circle, square):
    img_a, ma = circle
    img_b, mb = square
    corr = correspondence_from_masks(ma, mb, img_a.shape[:2])
    frames = ClassicalMorph().render(img_a, img_b, corr, n_frames=5, easing="linear")
    mid = frames[2]
    # The middle frame should differ from both endpoints (real morph, not a copy).
    assert _mad(mid, img_a) > 3.0
    assert _mad(mid, img_b) > 3.0
