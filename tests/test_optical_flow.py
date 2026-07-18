import numpy as np

from motion.morph.optical_flow import OpticalFlowMorph


def _mad(a, b):
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def test_flow_endpoints_and_count(circle, square):
    img_a, _ = circle
    img_b, _ = square
    frames = OpticalFlowMorph().render(img_a, img_b, None, n_frames=8)
    assert len(frames) == 8
    # At t=0 the warp is identity → frame 0 == A (and last == B).
    assert _mad(frames[0], img_a) < 1.0
    assert _mad(frames[-1], img_b) < 1.0


def test_flow_needs_no_correspondence():
    assert OpticalFlowMorph.requires_correspondence is False
