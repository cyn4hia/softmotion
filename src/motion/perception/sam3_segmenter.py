"""SAM 3 (Meta) text-prompted segmentation.

Written to the public API documented in facebookresearch/sam3::

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    state = processor.set_image(image)
    out = processor.set_text_prompt(state=state, prompt="a red apple")
    masks, boxes, scores = out["masks"], out["boxes"], out["scores"]

⚠️ VERIFY-ON-GPU: SAM 3 needs a CUDA machine, gated HF checkpoints, and its own
`pip install -e .` (see docs/GPU_SETUP.md). It could not be executed in this
environment. Optional builder kwargs (checkpoint path, device) are passed via
signature-introspection so this adapts to the exact installed version rather than
guessing a signature. Confirm against `sam3`'s README on first run.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from motion.device import resolve_device
from motion.logging import get_logger
from motion.perception.base import Detection, Segmenter

log = get_logger(__name__)


def _to_numpy(x):
    if hasattr(x, "detach"):  # torch tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)


class Sam3Segmenter(Segmenter):
    def __init__(
        self,
        checkpoint: str | Path | None = None,
        device: str | None = None,
        score_threshold: float = 0.5,
    ) -> None:
        self.device = resolve_device(device)
        self.score_threshold = score_threshold
        self._model = None
        self._processor = None
        self._checkpoint = str(checkpoint) if checkpoint else None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except Exception as e:  # pragma: no cover - requires GPU install
            raise RuntimeError(
                "sam3 is not installed. Install it on the GPU box: see docs/GPU_SETUP.md "
                "(git clone facebookresearch/sam3 && pip install -e .)."
            ) from e

        # Pass checkpoint/device only if the installed builder accepts them.
        sig = inspect.signature(build_sam3_image_model)
        kwargs = {}
        for key, val in (
            ("checkpoint", self._checkpoint),
            ("checkpoint_path", self._checkpoint),
            ("ckpt_path", self._checkpoint),
            ("device", self.device),
        ):
            if key in sig.parameters and val is not None:
                kwargs[key] = val
        log.info("Building SAM 3 image model on %s (kwargs=%s)", self.device, list(kwargs))
        model = build_sam3_image_model(**kwargs)
        if hasattr(model, "to"):
            model = model.to(self.device)
        if hasattr(model, "eval"):
            model.eval()
        self._model = model
        self._processor = Sam3Processor(model)

    def segment(self, image: np.ndarray, prompt: str, **kwargs) -> list[Detection]:
        self._ensure_loaded()
        assert self._processor is not None
        state = self._processor.set_image(image)
        out = self._processor.set_text_prompt(state=state, prompt=prompt)

        masks = _to_numpy(out["masks"])
        boxes = _to_numpy(out["boxes"])
        scores = _to_numpy(out["scores"]).reshape(-1)

        detections: list[Detection] = []
        for i in range(len(scores)):
            score = float(scores[i])
            if score < self.score_threshold:
                continue
            mask = masks[i]
            mask = mask[0] if mask.ndim == 3 else mask  # drop channel dim if present
            x1, y1, x2, y2 = (float(v) for v in boxes[i][:4])
            detections.append(
                Detection(
                    box=(x1, y1, x2, y2),
                    score=score,
                    label=prompt,
                    mask=mask.astype(bool),
                )
            )
        detections.sort(key=lambda d: d.score, reverse=True)
        return detections
