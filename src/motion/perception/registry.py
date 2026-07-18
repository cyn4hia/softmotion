"""Backend selection. Picks the right SAM 3 / LocateAnything implementation for the
current machine and returns ``None`` (never raises) when a model isn't available,
so the pipeline can degrade gracefully to mask-free morphing."""

from __future__ import annotations

from pathlib import Path

from motion.device import device_info, resolve_device
from motion.logging import get_logger
from motion.perception.base import Localizer, Segmenter

log = get_logger(__name__)


def build_segmenter(
    checkpoint: str | Path | None = None,
    device: str | None = None,
    required: bool = False,
) -> Segmenter | None:
    """SAM 3 segmenter, or None if torch/sam3 aren't installed."""
    dev = resolve_device(device)
    if not device_info().torch_available:
        _miss("SAM 3 segmenter (torch not installed)", required)
        return None
    try:
        from motion.perception.sam3_segmenter import Sam3Segmenter

        return Sam3Segmenter(checkpoint=checkpoint, device=dev)
    except Exception as e:  # pragma: no cover
        _miss(f"SAM 3 segmenter ({e})", required)
        return None


def build_localizer(
    device: str | None = None,
    gguf: str | Path | None = None,
    model_id: str = "nvidia/LocateAnything-3B",
    prefer: str | None = None,  # "cpp" | "pytorch" | None (auto)
    required: bool = False,
) -> Localizer | None:
    """LocateAnything localizer.

    Auto policy: use the ggml CLI on CPU/MPS (Mac mini) when the binary is present;
    use PyTorch on CUDA. Returns None if neither is available.
    """
    dev = resolve_device(device)
    from motion.perception.locateanything_cpp import LocateAnythingCpp

    want_cpp = prefer == "cpp" or (prefer is None and dev != "cuda")
    if want_cpp and LocateAnythingCpp.is_available():
        return LocateAnythingCpp(gguf=gguf)

    if prefer != "cpp" and dev == "cuda" and device_info().torch_available:
        try:
            from motion.perception.locateanything_localizer import LocateAnythingLocalizer

            return LocateAnythingLocalizer(model_id=model_id, device=dev)
        except Exception as e:  # pragma: no cover
            _miss(f"LocateAnything PyTorch ({e})", required)

    # Last try: cpp even if not "preferred", as long as the binary exists.
    if LocateAnythingCpp.is_available():
        return LocateAnythingCpp(gguf=gguf)

    _miss("LocateAnything (no ggml CLI on PATH and no CUDA/torch)", required)
    return None


def _miss(what: str, required: bool) -> None:
    msg = f"Perception backend unavailable: {what}"
    if required:
        raise RuntimeError(msg)
    log.warning("%s — continuing without it.", msg)
