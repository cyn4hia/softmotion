"""Morph backends: turn two images (+ optional correspondence) into a sequence of
transition frames. All backends here are pure NumPy/OpenCV and run on CPU."""

from motion.morph.base import MorphBackend
from motion.morph.classical import ClassicalMorph, morph_pair
from motion.morph.easing import apply_easing, schedule
from motion.morph.optical_flow import OpticalFlowMorph
from motion.morph.registry import available_backends, get_morph_backend

__all__ = [
    "MorphBackend",
    "ClassicalMorph",
    "OpticalFlowMorph",
    "morph_pair",
    "schedule",
    "apply_easing",
    "get_morph_backend",
    "available_backends",
]
