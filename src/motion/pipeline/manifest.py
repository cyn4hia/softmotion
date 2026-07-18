"""The `motion` manifest — a versioned JSON description of a render: inputs,
detections, morph segments, and outputs.

This is the **stable integration contract**. After Effects (or any host) drives a
render via the CLI and reads this file back to import footage and drive keyframes,
so treat it like an API: additive changes only, bump ``version`` on breaks. The
schema is a *chain* of objects and segments, so it already supports N-object merges
(not just two).
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

MANIFEST_VERSION = "1.0"


class ObjectSpec(BaseModel):
    image: str
    prompt: str | None = None
    mask: str | None = None  # optional user-supplied mask path


class DetectionRecord(BaseModel):
    object_index: int
    source: str  # "sam3" | "locateanything" | "grabcut" | "none"
    box: tuple[float, float, float, float] | None = None
    center: tuple[float, float] | None = None
    score: float = 1.0
    label: str | None = None
    mask_path: str | None = None


class MorphSegment(BaseModel):
    from_index: int
    to_index: int
    backend: str
    n_frames: int
    easing: str
    start_frame: int  # index of this segment's first frame in the final video
    frames_dir: str | None = None  # PNG sequence, if written (AE-friendly)


class MotionManifest(BaseModel):
    version: str = MANIFEST_VERSION
    created_utc: str | None = None
    device: str = "cpu"
    fps: int = 30
    size: tuple[int, int] = (0, 0)  # (H, W)
    objects: list[ObjectSpec] = Field(default_factory=list)
    detections: list[DetectionRecord] = Field(default_factory=list)
    segments: list[MorphSegment] = Field(default_factory=list)
    output_video: str = ""
    total_frames: int = 0
    notes: list[str] = Field(default_factory=list)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.model_dump(), indent=2))
        return p

    @classmethod
    def load(cls, path: str | Path) -> MotionManifest:
        return cls.model_validate_json(Path(path).read_text())
