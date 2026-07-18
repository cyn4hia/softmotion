"""LocateAnything via the ggml C++ port (mudler/locate-anything.cpp) — the
**CPU / Apple-Silicon (Metal)** path that makes LocateAnything usable on a Mac mini.

It shells out to the compiled ``locate-anything-cli`` binary (built by
scripts/setup_mac.sh) with a GGUF model::

    locate-anything-cli detect --model <gguf> --input <img> --prompt <text> \
        --mode hybrid --output boxes.json

and parses the JSON: ``{"detections":[{"label": "...", "box": [x1,y1,x2,y2]}, ...]}``.

No Python model runtime is required at inference time. Fine-tuned PyTorch weights
are converted to GGUF by :mod:`motion.train.convert_to_gguf`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from motion.io.images import ensure_rgb
from motion.logging import get_logger
from motion.perception.base import Detection, Localizer

log = get_logger(__name__)


class LocateAnythingCpp(Localizer):
    def __init__(
        self,
        gguf: str | Path | None = None,
        cli: str | Path | None = None,
        mode: str = "hybrid",
        coord_space: str = "pixel",  # "pixel" | "norm1000" | "norm1"
        threads: int | None = None,
    ) -> None:
        self.cli = str(cli or os.getenv("LOCATE_ANYTHING_CLI") or "locate-anything-cli")
        self.gguf = str(gguf or os.getenv("LOCATE_ANYTHING_GGUF", ""))
        self.mode = mode
        self.coord_space = coord_space
        self.threads = threads

    @staticmethod
    def is_available(cli: str | None = None) -> bool:
        cli = cli or os.getenv("LOCATE_ANYTHING_CLI") or "locate-anything-cli"
        return shutil.which(cli) is not None or Path(cli).exists()

    def _rescale(self, box: list[float], w: int, h: int) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = box[:4]
        if self.coord_space == "norm1000":
            return (x1 / 1000 * w, y1 / 1000 * h, x2 / 1000 * w, y2 / 1000 * h)
        if self.coord_space == "norm1":
            return (x1 * w, y1 * h, x2 * w, y2 * h)
        return (x1, y1, x2, y2)  # pixel

    def locate(self, image: np.ndarray, prompt: str, **kwargs) -> list[Detection]:
        if not self.gguf:
            raise RuntimeError(
                "No GGUF model set. Point LOCATE_ANYTHING_GGUF at a converted model "
                "(see scripts/setup_mac.sh / motion.train.convert_to_gguf)."
            )
        h, w = image.shape[:2]
        with tempfile.TemporaryDirectory() as td:
            img_path = Path(td) / "input.png"
            out_path = Path(td) / "boxes.json"
            cv2.imwrite(str(img_path), cv2.cvtColor(ensure_rgb(image), cv2.COLOR_RGB2BGR))
            cmd = [
                self.cli,
                "detect",
                "--model",
                self.gguf,
                "--input",
                str(img_path),
                "--prompt",
                prompt,
                "--mode",
                self.mode,
                "--output",
                str(out_path),
            ]
            if self.threads:
                cmd += ["--threads", str(self.threads)]
            log.info("locate-anything.cpp: %s", " ".join(cmd))
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"locate-anything-cli failed:\n{proc.stderr or proc.stdout}")
            data = json.loads(out_path.read_text())

        dets: list[Detection] = []
        for d in data.get("detections", []):
            box = self._rescale(d["box"], w, h)
            dets.append(
                Detection(box=box, score=float(d.get("score", 1.0)), label=d.get("label", prompt))
            )
        return dets
