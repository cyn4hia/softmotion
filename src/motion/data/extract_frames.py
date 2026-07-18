"""Extract training frames from source videos — uniform sampling or scene-change
sampling. Pure OpenCV, runs anywhere (handy to prep data on the Mac before shipping
to the GPU box)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel
from tqdm import tqdm

from motion.config import load_yaml, resolve_path
from motion.io.images import save_image
from motion.logging import get_logger

log = get_logger(__name__)

_VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


class ExtractConfig(BaseModel):
    input_dir: str = "data/raw"
    output_dir: str = "data/frames"
    fps: float = 2.0  # target samples per second (uniform mode)
    scene_threshold: float = 0.0  # >0 → scene-change mode (mean abs diff, 0..255)
    max_frames_per_video: int = 300
    min_side: int = 384  # skip tiny videos


def _iter_videos(root: Path):
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in _VIDEO_EXT:
            yield p


def _safe_id(rel: Path) -> str:
    """Unique, filesystem-safe id from a path relative to the input root, so two
    videos with the same filename stem in different subdirs never collide."""
    raw = str(rel.with_suffix(""))
    return "".join(c if c.isalnum() else "_" for c in raw).strip("_")


def _extract_one(path: Path, out_dir: Path, cfg: ExtractConfig, prefix: str) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        log.warning("cannot open %s", path)
        return 0
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / cfg.fps))) if cfg.fps > 0 else 1

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    idx = 0
    prev_gray: np.ndarray | None = None
    with tqdm(desc=path.name, leave=False) as bar:
        while saved < cfg.max_frames_per_video:
            ok, frame = cap.read()
            if not ok:
                break
            bar.update(1)
            h, w = frame.shape[:2]
            if min(h, w) < cfg.min_side:
                break

            take = False
            if cfg.scene_threshold > 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is None:
                    take = True
                else:
                    diff = float(np.mean(cv2.absdiff(gray, prev_gray)))
                    take = diff >= cfg.scene_threshold
                prev_gray = gray
            else:
                take = idx % step == 0
            idx += 1

            if take:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                save_image(out_dir / f"{prefix}_{saved:05d}.jpg", rgb)
                saved += 1
    cap.release()
    return saved


def run_extract(config_path: str | Path) -> None:
    cfg = ExtractConfig.model_validate(load_yaml(config_path))
    in_dir = resolve_path(cfg.input_dir)
    out_dir = resolve_path(cfg.output_dir)
    total = 0
    for video in _iter_videos(in_dir):
        rel = video.relative_to(in_dir)
        prefix = _safe_id(rel)  # unique across subdirs → no frame/mask collisions
        # Mirror the input tree so output folders can't collide either.
        n = _extract_one(video, out_dir / rel.parent / video.stem, cfg, prefix)
        total += n
        log.info("%s → %d frames", rel, n)
    log.info("extracted %d frames to %s", total, out_dir)
