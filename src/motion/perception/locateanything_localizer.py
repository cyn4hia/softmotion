"""LocateAnything-3B (NVIDIA) open-vocabulary localization — **PyTorch / CUDA** path.

Written to the model card at huggingface.co/nvidia/LocateAnything-3B:

    from transformers import pipeline
    pipe = pipeline("image-text-to-text", model="nvidia/LocateAnything-3B",
                    trust_remote_code=True)

Prompt templates (from the card):
    detection : "Locate all the instances that matches the following description: {q}"
    pointing  : "Point to: {q}"
Outputs are text with `<box><x1><y1><x2><y2></box>` (and `<box><x><y></box>` for
points), coordinates normalized to **[0, 1000]** — we denormalize to pixels.

⚠️ LICENSE: LocateAnything is NVIDIA **non-commercial / research only**. For a
commercial/AE release, swap this backend (see docs/LICENSING.md).
⚠️ VERIFY-ON-GPU: requires a CUDA GPU; not runnable in this environment. On a Mac
mini use :class:`LocateAnythingCpp` (ggml) instead.
"""

from __future__ import annotations

import re

import numpy as np

from motion.device import resolve_device
from motion.logging import get_logger
from motion.perception.base import Detection, Localizer

log = get_logger(__name__)

_BOX_RE = re.compile(r"<box>\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*</box>")
_POINT_RE = re.compile(r"<box>\s*(\d+)\s+(\d+)\s*</box>")

DETECT_TMPL = "Locate all the instances that matches the following description: {q}"
POINT_TMPL = "Point to: {q}"


def parse_boxes(text: str, width: int, height: int, coord_max: float = 1000.0) -> list[Detection]:
    """Parse `<box>x1 y1 x2 y2</box>` (normalized to ``coord_max``) into pixel boxes."""
    dets: list[Detection] = []
    for m in _BOX_RE.finditer(text):
        x1, y1, x2, y2 = (int(v) for v in m.groups())
        dets.append(
            Detection(
                box=(
                    x1 / coord_max * width,
                    y1 / coord_max * height,
                    x2 / coord_max * width,
                    y2 / coord_max * height,
                ),
                score=1.0,
            )
        )
    return dets


def parse_points(text: str, width: int, height: int, coord_max: float = 1000.0) -> list[Detection]:
    """Parse point-mode `<box>x y</box>` outputs into degenerate-box Detections
    carrying the pixel point in ``.points``."""
    dets: list[Detection] = []
    for m in _POINT_RE.finditer(text):
        x, y = (int(v) for v in m.groups())
        px, py = x / coord_max * width, y / coord_max * height
        dets.append(Detection(box=(px, py, px, py), score=1.0, points=[(px, py)]))
    return dets


class LocateAnythingLocalizer(Localizer):
    def __init__(
        self,
        model_id: str = "nvidia/LocateAnything-3B",
        device: str | None = None,
        max_new_tokens: int = 512,
    ) -> None:
        self.model_id = model_id
        self.device = resolve_device(device)
        self.max_new_tokens = max_new_tokens
        self._pipe = None

    def _ensure_loaded(self) -> None:
        if self._pipe is not None:
            return
        try:
            from transformers import pipeline
        except Exception as e:  # pragma: no cover - requires gpu extra
            raise RuntimeError(
                "transformers not installed. `pip install -e '.[gpu]'` on the GPU box."
            ) from e
        log.info("Loading %s on %s", self.model_id, self.device)
        self._pipe = pipeline(
            "image-text-to-text",
            model=self.model_id,
            trust_remote_code=True,
            device=0 if self.device == "cuda" else -1,
        )

    def locate(
        self, image: np.ndarray, prompt: str, mode: str = "detect", **kwargs
    ) -> list[Detection]:
        self._ensure_loaded()
        assert self._pipe is not None
        from PIL import Image

        h, w = image.shape[:2]
        pil = Image.fromarray(image)
        instr = (POINT_TMPL if mode == "point" else DETECT_TMPL).format(q=prompt)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil},
                    {"type": "text", "text": instr},
                ],
            }
        ]
        result = self._pipe(text=messages, max_new_tokens=self.max_new_tokens)
        text = _extract_text(result)
        dets = parse_points(text, w, h) if mode == "point" else parse_boxes(text, w, h)
        for d in dets:
            d.label = prompt
        return dets


def _extract_text(result) -> str:
    """Best-effort text extraction across transformers pipeline return shapes."""
    if isinstance(result, str):
        return result
    if isinstance(result, list) and result:
        item = result[0]
        if isinstance(item, dict):
            gen = item.get("generated_text")
            if isinstance(gen, str):
                return gen
            if isinstance(gen, list) and gen:  # chat format
                last = gen[-1]
                if isinstance(last, dict):
                    return str(last.get("content", ""))
            return str(item.get("text", item))
    return str(result)
