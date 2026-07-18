"""Perception data types and interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection:
    """One detected/segmented object.

    ``box`` is ``(x1, y1, x2, y2)`` in **pixels**. ``mask`` (if present) is a
    boolean ``(H, W)`` array. ``points`` are optional pixel keypoints (e.g.
    LocateAnything "point-to" outputs).
    """

    box: tuple[float, float, float, float]
    score: float = 1.0
    label: str | None = None
    mask: np.ndarray | None = None
    points: list[tuple[float, float]] = field(default_factory=list)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class Segmenter(ABC):
    """Produce pixel-precise masks from a text prompt (SAM 3)."""

    @abstractmethod
    def segment(self, image: np.ndarray, prompt: str, **kwargs) -> list[Detection]:
        raise NotImplementedError

    def best(self, image: np.ndarray, prompt: str, **kwargs) -> Detection | None:
        """Highest-scoring detection, or None."""
        dets = self.segment(image, prompt, **kwargs)
        return max(dets, key=lambda d: d.score) if dets else None


class Localizer(ABC):
    """Ground a text phrase to boxes/points (LocateAnything or a swap-in)."""

    @abstractmethod
    def locate(self, image: np.ndarray, prompt: str, **kwargs) -> list[Detection]:
        raise NotImplementedError

    def best(self, image: np.ndarray, prompt: str, **kwargs) -> Detection | None:
        dets = self.locate(image, prompt, **kwargs)
        return max(dets, key=lambda d: d.score) if dets else None
