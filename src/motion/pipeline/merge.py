"""The end-to-end merge pipeline: 2 (or N) images → one seamless transition video
plus a manifest.

Design goals:
- **Runs on a Mac today.** Perception is optional; if SAM 3 / LocateAnything aren't
  present it falls back (LocateAnything-seeded GrabCut → plain GrabCut → mask-free
  optical flow) so you always get a video.
- **Scales to N objects.** `merge_sequence` chains pairwise morphs A→B→C…, which is
  the same primitive the future AE script will call.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from motion.correspondence.builder import correspondence_from_masks
from motion.device import resolve_device
from motion.io.images import common_canvas, load_image, resize_to_match, save_image
from motion.io.masks import load_mask, save_mask
from motion.io.video import write_video
from motion.logging import get_logger
from motion.morph.registry import get_morph_backend
from motion.perception.base import Localizer, Segmenter
from motion.pipeline.manifest import (
    DetectionRecord,
    MorphSegment,
    MotionManifest,
    ObjectSpec,
)

log = get_logger(__name__)


class MergeConfig(BaseModel):
    backend: str = "auto"  # "auto" | "classical" | "flow"
    n_frames: int = 48
    fps: int = 30
    easing: str = "ease_in_out"
    max_side: int = 1024  # downscale longest side for speed; 0 = no limit
    device: str | None = None
    n_contour: int = 64
    allow_grabcut: bool = True  # model-free fallback segmentation
    write_frames: bool = False  # also emit an AE-friendly PNG sequence


class _ObjectPercept(BaseModel, arbitrary_types_allowed=True):
    mask: np.ndarray | None = None
    source: str = "none"
    box: tuple[float, float, float, float] | None = None
    score: float = 1.0


class MergePipeline:
    def __init__(
        self,
        config: MergeConfig | None = None,
        segmenter: Segmenter | None = None,
        localizer: Localizer | None = None,
    ) -> None:
        self.cfg = config or MergeConfig()
        self.segmenter = segmenter
        self.localizer = localizer
        self.device = resolve_device(self.cfg.device)

    # -- perception -------------------------------------------------------
    def _perceive(
        self, image: np.ndarray, prompt: str | None, mask_path: str | None
    ) -> _ObjectPercept:
        if mask_path:
            m = resize_to_match(load_mask(mask_path).astype(np.uint8), image.shape[:2]).astype(bool)
            return _ObjectPercept(mask=m, source="user")

        if self.segmenter is not None and prompt:
            det = self.segmenter.best(image, prompt)
            if det is not None and det.mask is not None:
                return _ObjectPercept(mask=det.mask, source="sam3", box=det.box, score=det.score)

        box = None
        if self.localizer is not None and prompt:
            ld = self.localizer.best(image, prompt)
            if ld is not None:
                box = ld.box

        if self.cfg.allow_grabcut:
            from motion.perception.heuristic import grabcut_segment

            rect = None
            if box is not None:
                x1, y1, x2, y2 = (int(v) for v in box)
                rect = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))
            mask = grabcut_segment(image, rect=rect)
            src = "grabcut+locate" if box is not None else "grabcut"
            return _ObjectPercept(mask=mask, source=src, box=box)

        return _ObjectPercept(source="none", box=box)

    # -- helpers ----------------------------------------------------------
    def _prep(self, image: np.ndarray) -> np.ndarray:
        if not self.cfg.max_side:
            return image
        h, w = image.shape[:2]
        longest = max(h, w)
        if longest <= self.cfg.max_side:
            return image
        scale = self.cfg.max_side / longest
        return resize_to_match(image, (int(round(h * scale)), int(round(w * scale))))

    def _choose_backend(self, has_a: bool, has_b: bool) -> str:
        if self.cfg.backend != "auto":
            if self.cfg.backend == "classical" and not (has_a and has_b):
                log.warning("classical morph needs masks for both objects; falling back to 'flow'.")
                return "flow"
            return self.cfg.backend
        return "classical" if (has_a and has_b) else "flow"

    # -- public API -------------------------------------------------------
    def merge_pair(
        self,
        image_a: str | Path,
        image_b: str | Path,
        prompt_a: str | None = None,
        prompt_b: str | None = None,
        out_video: str | Path = "outputs/merge.mp4",
        mask_a: str | None = None,
        mask_b: str | None = None,
        out_manifest: str | Path | None = None,
    ) -> MotionManifest:
        return self.merge_sequence(
            images=[str(image_a), str(image_b)],
            prompts=[prompt_a, prompt_b],
            masks=[mask_a, mask_b],
            out_video=out_video,
            out_manifest=out_manifest,
        )

    def merge_sequence(
        self,
        images: list[str],
        prompts: list[str | None] | None = None,
        masks: list[str | None] | None = None,
        out_video: str | Path = "outputs/merge.mp4",
        out_manifest: str | Path | None = None,
    ) -> MotionManifest:
        if len(images) < 2:
            raise ValueError("need at least two images to merge")
        prompts = prompts or [None] * len(images)
        masks = masks or [None] * len(images)

        # Load, downscale, and put everything on the first image's canvas so the
        # whole chain shares one geometry.
        loaded = [self._prep(load_image(p)) for p in images]
        base_size = loaded[0].shape[:2]
        loaded = [loaded[0]] + [resize_to_match(im, base_size) for im in loaded[1:]]

        percepts = [
            self._perceive(im, pr, mk) for im, pr, mk in zip(loaded, prompts, masks, strict=True)
        ]

        out_video = Path(out_video)
        frames_root = out_video.with_suffix("")  # e.g. outputs/merge/
        all_frames: list[np.ndarray] = []
        segments: list[MorphSegment] = []
        detections: list[DetectionRecord] = []

        for i, (_im, pc) in enumerate(zip(loaded, percepts, strict=True)):
            mask_path = None
            if pc.mask is not None and self.cfg.write_frames:
                mask_path = str(frames_root / f"mask_{i:02d}.png")
                save_mask(mask_path, pc.mask)
            detections.append(
                DetectionRecord(
                    object_index=i,
                    source=pc.source,
                    box=pc.box,
                    center=None
                    if pc.box is None
                    else ((pc.box[0] + pc.box[2]) / 2, (pc.box[1] + pc.box[3]) / 2),
                    score=pc.score,
                    label=prompts[i],
                    mask_path=mask_path,
                )
            )

        for i in range(len(loaded) - 1):
            a, b = common_canvas(loaded[i], loaded[i + 1])
            pa, pb = percepts[i], percepts[i + 1]
            backend_name = self._choose_backend(pa.mask is not None, pb.mask is not None)
            backend = get_morph_backend(backend_name)

            corr = None
            if backend.requires_correspondence:
                corr = correspondence_from_masks(
                    pa.mask, pb.mask, a.shape[:2], n_contour=self.cfg.n_contour
                )
            log.info(
                "segment %d→%d via '%s' (%d frames)", i, i + 1, backend_name, self.cfg.n_frames
            )
            frames = backend.render(
                a, b, correspondence=corr, n_frames=self.cfg.n_frames, easing=self.cfg.easing
            )
            # Drop the duplicated boundary frame between consecutive segments.
            start = len(all_frames)
            if all_frames:
                frames = frames[1:]
            all_frames.extend(frames)

            seg_frames_dir = None
            if self.cfg.write_frames:
                seg_frames_dir = str(frames_root / f"seg_{i:02d}")
                for j, fr in enumerate(frames):
                    save_image(Path(seg_frames_dir) / f"{j:04d}.png", fr)

            segments.append(
                MorphSegment(
                    from_index=i,
                    to_index=i + 1,
                    backend=backend_name,
                    n_frames=len(frames),
                    easing=self.cfg.easing,
                    start_frame=start,
                    frames_dir=seg_frames_dir,
                )
            )

        write_video(all_frames, out_video, fps=self.cfg.fps)
        log.info("wrote %s (%d frames)", out_video, len(all_frames))

        manifest = MotionManifest(
            created_utc=datetime.now(UTC).isoformat(),
            device=self.device,
            fps=self.cfg.fps,
            size=base_size,
            objects=[
                ObjectSpec(image=images[i], prompt=prompts[i], mask=masks[i])
                for i in range(len(images))
            ],
            detections=detections,
            segments=segments,
            output_video=str(out_video),
            total_frames=len(all_frames),
            notes=[f"morph backend policy: {self.cfg.backend}"],
        )
        if out_manifest is None:
            out_manifest = out_video.with_suffix(".json")
        manifest.save(out_manifest)
        log.info("wrote manifest %s", out_manifest)
        return manifest
