import numpy as np

from motion.correspondence.builder import correspondence_from_grid, correspondence_from_masks
from motion.correspondence.contour import mask_to_contour, resample_contour


def test_resample_contour_length(circle):
    _, mask = circle
    contour = mask_to_contour(mask)
    r = resample_contour(contour, 64)
    assert r.shape == (64, 2)


def test_correspondence_from_masks(circle, square):
    _, ma = circle
    _, mb = square
    corr = correspondence_from_masks(ma, mb, ma.shape[:2], n_contour=48)
    assert corr.points_a.shape == corr.points_b.shape
    assert corr.triangles.ndim == 2 and corr.triangles.shape[1] == 3
    assert corr.triangles.min() >= 0
    assert corr.triangles.max() < corr.n_points
    assert corr.size == ma.shape[:2]


def test_points_at_interpolates(circle, square):
    _, ma = circle
    _, mb = square
    corr = correspondence_from_masks(ma, mb, ma.shape[:2])
    np.testing.assert_allclose(corr.points_at(0.0), corr.points_a, atol=1e-4)
    np.testing.assert_allclose(corr.points_at(1.0), corr.points_b, atol=1e-4)


def test_grid_fallback():
    corr = correspondence_from_grid((200, 200), nx=6, ny=6)
    assert corr.n_points == 36
    assert corr.triangles.shape[1] == 3
