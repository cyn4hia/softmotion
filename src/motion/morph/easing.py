"""Interpolation schedules. The eased ``t`` drives *both* geometry and blend, so
`ease_in_out` gives a natural slow-in/slow-out merge."""

from __future__ import annotations

import numpy as np

_EASINGS = {
    "linear": lambda t: t,
    "ease_in": lambda t: t * t,
    "ease_out": lambda t: 1 - (1 - t) ** 2,
    "ease_in_out": lambda t: t * t * (3 - 2 * t),  # smoothstep
}


def apply_easing(t: np.ndarray, easing: str = "linear") -> np.ndarray:
    if easing not in _EASINGS:
        raise ValueError(f"unknown easing {easing!r}; choose from {sorted(_EASINGS)}")
    return _EASINGS[easing](np.asarray(t, dtype=np.float64))


def schedule(
    n_frames: int, easing: str = "ease_in_out", include_endpoints: bool = True
) -> np.ndarray:
    """Return ``n_frames`` values of ``t`` in [0, 1] after easing.

    ``include_endpoints=True`` guarantees the first/last frames are exactly the
    input images (t=0 and t=1).
    """
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1")
    if n_frames == 1:
        base = np.array([0.0])
    elif include_endpoints:
        base = np.linspace(0.0, 1.0, n_frames)
    else:
        base = (np.arange(n_frames) + 0.5) / n_frames
    return apply_easing(base, easing)
