import numpy as np
import pytest

from motion.morph.easing import apply_easing, schedule


def test_schedule_endpoints():
    ts = schedule(10, "ease_in_out", include_endpoints=True)
    assert len(ts) == 10
    assert ts[0] == pytest.approx(0.0)
    assert ts[-1] == pytest.approx(1.0)


def test_schedule_monotonic():
    ts = schedule(20, "ease_in_out")
    assert np.all(np.diff(ts) >= -1e-9)


def test_single_frame():
    assert len(schedule(1)) == 1


def test_unknown_easing():
    with pytest.raises(ValueError):
        apply_easing(np.array([0.5]), "boing")
