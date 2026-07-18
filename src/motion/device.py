"""Central device / backend resolution.

Every model-touching code path goes through :func:`resolve_device` so the *same*
source runs unchanged on a CUDA box, an Apple-Silicon Mac, or a CPU-only machine —
that is the whole "just swap to a GPU and run" promise.

`torch` is an **optional** dependency (only in the ``gpu``/``train`` extras). This
module degrades gracefully when torch is absent, so the CPU render install never
imports it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_VALID = ("cuda", "mps", "cpu")


@dataclass(frozen=True)
class DeviceInfo:
    """Resolved compute device plus everything callers need to branch on."""

    device: str  # "cuda" | "mps" | "cpu"
    torch_available: bool
    cuda_available: bool
    mps_available: bool
    detail: str

    @property
    def is_gpu(self) -> bool:
        return self.device in ("cuda", "mps")

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.device} ({self.detail})"


def _probe() -> DeviceInfo:
    """Inspect the machine once. Never raises."""
    try:
        import torch  # noqa: PLC0415  (optional, deliberately lazy)
    except Exception:
        return DeviceInfo(
            device="cpu",
            torch_available=False,
            cuda_available=False,
            mps_available=False,
            detail="torch not installed — CPU render path only",
        )

    cuda = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    mps = bool(
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )
    if cuda:
        name = torch.cuda.get_device_name(0)
        return DeviceInfo("cuda", True, True, mps, f"CUDA: {name}")
    if mps:
        return DeviceInfo("mps", True, cuda, True, "Apple Silicon (Metal / MPS)")
    return DeviceInfo("cpu", True, False, False, "torch present, no accelerator")


_CACHED: DeviceInfo | None = None


def device_info(refresh: bool = False) -> DeviceInfo:
    """Return cached machine capabilities (probe once)."""
    global _CACHED
    if _CACHED is None or refresh:
        _CACHED = _probe()
    return _CACHED


def resolve_device(prefer: str | None = None) -> str:
    """Resolve the device string to actually use.

    Resolution order:
      1. explicit ``prefer`` argument (validated),
      2. ``MOTION_DEVICE`` env var,
      3. auto-detect (cuda → mps → cpu).

    A requested accelerator that isn't available falls back to CPU with the
    reason available via :func:`device_info`.
    """
    req = (prefer or os.getenv("MOTION_DEVICE") or "auto").lower()
    if req not in (*_VALID, "auto"):
        raise ValueError(f"MOTION_DEVICE / prefer must be one of {(*_VALID, 'auto')}, got {req!r}")

    info = device_info()
    if req == "auto":
        return info.device
    if req == "cuda":
        return "cuda" if info.cuda_available else "cpu"
    if req == "mps":
        return "mps" if info.mps_available else "cpu"
    return "cpu"


def describe() -> str:
    """Human-readable summary for `motion doctor`."""
    info = device_info()
    lines = [
        f"resolved device : {resolve_device()}",
        f"torch           : {'yes' if info.torch_available else 'no'}",
        f"cuda            : {'yes' if info.cuda_available else 'no'}",
        f"mps (apple)     : {'yes' if info.mps_available else 'no'}",
        f"detail          : {info.detail}",
    ]
    return "\n".join(lines)
