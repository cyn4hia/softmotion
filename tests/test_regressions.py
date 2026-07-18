"""Regression tests locking in the fixes from the adversarial review."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from motion.config import _expand
from motion.correspondence.builder import correspondence_from_grid
from motion.data.extract_frames import _safe_id
from motion.morph.classical import ClassicalMorph
from motion.morph.compositor import seamless_clone
from motion.morph.optical_flow import OpticalFlowMorph


def test_optical_flow_tiny_frame_no_crash():
    # Below DIS min size → must fall back to cross-dissolve, not raise.
    a = np.zeros((10, 10, 3), np.uint8)
    b = np.full((10, 10, 3), 200, np.uint8)
    frames = OpticalFlowMorph().render(a, b, None, n_frames=5)
    assert len(frames) == 5
    assert frames[0].shape == (10, 10, 3)
    assert np.mean(np.abs(frames[0].astype(int) - a)) < 1  # t=0 ≈ A


def test_classical_offframe_grid_no_wrong_region(circle, square):
    # Grid correspondence with a displacement that pushes points off-frame used to
    # negative-index the source. Now it must render without crashing.
    img_a, _ = circle
    h, w = img_a.shape[:2]
    n = 36
    disp = np.tile(np.array([-40.0, -30.0], np.float32), (n, 1))  # shove off-frame
    corr = correspondence_from_grid((h, w), nx=6, ny=6, displacement_b=disp)
    frames = ClassicalMorph().render(img_a, square[0], corr, n_frames=4)
    assert len(frames) == 4
    assert all(f.shape == img_a.shape and f.dtype == np.uint8 for f in frames)


def test_seamless_clone_border_mask_falls_back():
    src = np.full((80, 80, 3), 120, np.uint8)
    dst = np.zeros((80, 80, 3), np.uint8)
    mask = np.zeros((80, 80), bool)
    mask[:60, :3] = True  # tall strip hugging the left edge → cv2 ROI would overflow
    out = seamless_clone(src, dst, mask)  # must not raise
    assert out.shape == (80, 80, 3)
    assert out.dtype == np.uint8


def test_config_env_default_and_lowercase(monkeypatch):
    monkeypatch.delenv("some_lower", raising=False)
    assert _expand("${some_lower:fallback}") == "fallback"  # lowercase names supported
    monkeypatch.setenv("some_lower", "hit")
    assert _expand("${some_lower}") == "hit"


def test_config_env_unset_no_default_raises(monkeypatch):
    monkeypatch.delenv("MOTION_MISSING_XYZ", raising=False)
    with pytest.raises(KeyError):
        _expand("${MOTION_MISSING_XYZ}")


def test_safe_id_unique_across_subdirs():
    a = _safe_id(Path("a/clip.mp4"))
    b = _safe_id(Path("b/clip.mp4"))
    assert a != b
    assert a == "a_clip" and b == "b_clip"


def test_example_images_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "examples" / "a.png").exists()
    assert (root / "examples" / "b.png").exists()
