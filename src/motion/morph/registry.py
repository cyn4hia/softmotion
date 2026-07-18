"""Name → backend lookup so the CLI, pipeline, and manifest can refer to morph
engines by a stable string. Register new backends here."""

from __future__ import annotations

from motion.morph.base import MorphBackend
from motion.morph.classical import ClassicalMorph
from motion.morph.optical_flow import OpticalFlowMorph

_BACKENDS: dict[str, type[MorphBackend]] = {
    ClassicalMorph.name: ClassicalMorph,
    OpticalFlowMorph.name: OpticalFlowMorph,
}


def available_backends() -> list[str]:
    return sorted(_BACKENDS)


def get_morph_backend(name: str, **kwargs) -> MorphBackend:
    if name not in _BACKENDS:
        raise ValueError(f"unknown morph backend {name!r}; available: {available_backends()}")
    return _BACKENDS[name](**kwargs)


def register_backend(cls: type[MorphBackend]) -> type[MorphBackend]:
    """Decorator to add a backend (e.g. an optional GPU diffusion morph)."""
    _BACKENDS[cls.name] = cls
    return cls
