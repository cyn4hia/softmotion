"""Perception: segmentation (SAM 3) and open-vocabulary localization
(LocateAnything). Everything sits behind the :class:`Segmenter` / :class:`Localizer`
ABCs so backends are swappable — critical because LocateAnything is
research-license-only and may need replacing for commercial use.

Heavy model libraries are imported lazily inside each backend, so importing
``motion.perception`` never drags torch/transformers into the CPU render install.
"""

from motion.perception.base import Detection, Localizer, Segmenter

__all__ = ["Detection", "Segmenter", "Localizer"]
