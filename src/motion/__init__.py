"""motion — seamless object-merge morphing.

Public surface is intentionally small and import-light: importing `motion` must
never pull in torch or any GPU dependency (the Mac render install has none). Heavy
backends are imported lazily inside the modules that use them.
"""

from motion.device import DeviceInfo, resolve_device

__version__ = "0.1.0"

__all__ = ["__version__", "resolve_device", "DeviceInfo"]
