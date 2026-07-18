"""Video writing/reading via imageio's bundled ffmpeg. Frames are RGB uint8."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from motion.io.images import ensure_rgb


def write_video(
    frames: Iterable[np.ndarray],
    path: str | Path,
    fps: int = 30,
    codec: str = "libx264",
    quality: int | None = 8,
) -> Path:
    """Encode RGB frames to a video file. Returns the output path.

    H.264 requires even dimensions; frames are padded by one pixel if needed.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    prepared: list[np.ndarray] = []
    for fr in frames:
        fr = ensure_rgb(fr)
        h, w = fr.shape[:2]
        if h % 2 or w % 2:
            fr = np.pad(fr, ((0, h % 2), (0, w % 2), (0, 0)), mode="edge")
        prepared.append(fr)
    if not prepared:
        raise ValueError("write_video got no frames")

    kwargs: dict = {"fps": fps, "codec": codec}
    if quality is not None:
        kwargs["quality"] = quality
    # macro_block_size=1 keeps the output at the *exact* input dimensions. The
    # default (16) silently resizes to a multiple of 16 — unacceptable when a
    # downstream comp/AE expects the source resolution. We already guarantee even
    # dimensions above, which is what H.264 actually requires.
    kwargs["macro_block_size"] = 1
    iio.imwrite(out, np.stack(prepared), plugin="FFMPEG", **kwargs)
    return out


def read_video_frames(path: str | Path, stride: int = 1) -> Iterator[np.ndarray]:
    """Yield RGB frames from a video, every ``stride``-th frame."""
    for i, frame in enumerate(iio.imiter(path, plugin="FFMPEG")):
        if i % stride == 0:
            yield ensure_rgb(frame)
