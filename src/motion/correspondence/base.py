"""The data contract every morph backend consumes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Correspondence:
    """Matched control points between image A and B plus a shared triangulation.

    ``points_a[i]`` corresponds to ``points_b[i]`` (same semantic location on the
    two objects / canvas). ``triangles`` are index triples into *both* point
    arrays — a single mesh topology valid at every interpolation ``t`` because the
    vertices move but the connectivity does not.
    """

    points_a: np.ndarray  # (N, 2) float32, pixel coords in A
    points_b: np.ndarray  # (N, 2) float32, pixel coords in B
    triangles: np.ndarray  # (M, 3) int32 indices into points_a / points_b
    size: tuple[int, int]  # (H, W) shared canvas

    def __post_init__(self) -> None:
        self.points_a = np.asarray(self.points_a, dtype=np.float32).reshape(-1, 2)
        self.points_b = np.asarray(self.points_b, dtype=np.float32).reshape(-1, 2)
        self.triangles = np.asarray(self.triangles, dtype=np.int32).reshape(-1, 3)
        if self.points_a.shape != self.points_b.shape:
            raise ValueError(f"points_a {self.points_a.shape} != points_b {self.points_b.shape}")
        n = len(self.points_a)
        if self.triangles.size and (self.triangles.min() < 0 or self.triangles.max() >= n):
            raise ValueError("triangle index out of range")

    @property
    def n_points(self) -> int:
        return len(self.points_a)

    def points_at(self, t: float) -> np.ndarray:
        """Linearly interpolated control points at ``t`` in [0, 1]."""
        return (1.0 - t) * self.points_a + t * self.points_b
