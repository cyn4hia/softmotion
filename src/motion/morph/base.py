"""Morph backend interface. Adding a new engine (e.g. a GPU diffusion morph) means
implementing ``render`` and registering it — nothing else in the pipeline changes."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from motion.correspondence.base import Correspondence


class MorphBackend(ABC):
    """Render a transition from image A to image B as a list of RGB frames."""

    #: short, stable identifier used by the CLI / registry / manifest
    name: str = "base"

    #: whether this backend needs a :class:`Correspondence` (vs. inferring motion)
    requires_correspondence: bool = False

    @abstractmethod
    def render(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        correspondence: Correspondence | None = None,
        n_frames: int = 48,
        easing: str = "ease_in_out",
        **kwargs,
    ) -> list[np.ndarray]:
        """Return ``n_frames`` RGB uint8 frames, frame 0 ≈ A and frame -1 ≈ B."""
        raise NotImplementedError
