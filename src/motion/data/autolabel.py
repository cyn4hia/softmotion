"""Weakly label extracted frames with the *base* models to seed fine-tuning.

Produces a neutral, model-agnostic annotation file (`annotations.json`) plus mask
PNGs. `dataset_sam3` / `dataset_la` convert this into each trainer's format. Run a
human review pass over these weak labels before training for best results.

Best on the GPU box (SAM 3 + LocateAnything in PyTorch), but degrades: without
SAM 3 it uses LocateAnything(ggml)-seeded GrabCut so you can prototype on a Mac.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel
from tqdm import tqdm

from motion.config import load_yaml, resolve_path
from motion.io.images import load_image
from motion.io.masks import save_mask
from motion.logging import get_logger

log = get_logger(__name__)

_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class AutolabelConfig(BaseModel):
    frames_dir: str = "data/frames"
    out_dir: str = "data/annotations"
    prompts: list[str] = ["object"]
    device: str | None = None
    sam3_score_threshold: float = 0.5
    localizer_prefer: str | None = None  # "cpp" | "pytorch" | None
    max_images: int = 0  # 0 = all


def run_autolabel(config_path: str | Path) -> None:
    from motion.perception.heuristic import grabcut_segment
    from motion.perception.registry import build_localizer, build_segmenter

    cfg = AutolabelConfig.model_validate(load_yaml(config_path))
    frames_dir = resolve_path(cfg.frames_dir)
    out_dir = resolve_path(cfg.out_dir)
    masks_dir = out_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    seg = build_segmenter(device=cfg.device)
    loc = build_localizer(device=cfg.device, prefer=cfg.localizer_prefer)
    if seg is None and loc is None:
        raise RuntimeError("No perception backend available; cannot autolabel.")
    log.info("autolabel with segmenter=%s localizer=%s", type(seg).__name__, type(loc).__name__)

    images_meta: list[dict] = []
    annotations: list[dict] = []
    ann_id = 0

    files = [p for p in sorted(frames_dir.rglob("*")) if p.suffix.lower() in _IMG_EXT]
    if cfg.max_images:
        files = files[: cfg.max_images]

    for img_id, path in enumerate(tqdm(files, desc="autolabel")):
        image = load_image(path)
        h, w = image.shape[:2]
        images_meta.append({"id": img_id, "file_name": str(path), "width": w, "height": h})

        for prompt in cfg.prompts:
            box = None
            score = 1.0
            mask = None
            if seg is not None:
                det = seg.best(image, prompt)
                if det is not None:
                    box, score, mask = det.box, det.score, det.mask
            if box is None and loc is not None:
                ld = loc.best(image, prompt)
                if ld is not None:
                    box, score = ld.box, ld.score
            if mask is None and box is not None:
                x1, y1, x2, y2 = (int(v) for v in box)
                mask = grabcut_segment(image, rect=(x1, y1, max(1, x2 - x1), max(1, y2 - y1)))
            if box is None and mask is None:
                continue

            mask_file = None
            if mask is not None:
                # Key on the globally-unique annotation id so masks can never
                # collide across frames-with-same-stem or prompts-with-same-slug.
                mask_file = f"masks/{ann_id:07d}_{img_id:06d}_{_slug(prompt)}.png"
                save_mask(out_dir / mask_file, mask)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "concept": prompt,
                    "box": list(box) if box else None,
                    "score": float(score),
                    "mask_file": mask_file,
                }
            )
            ann_id += 1

    payload = {
        "concepts": cfg.prompts,
        "images": images_meta,
        "annotations": annotations,
    }
    (out_dir / "annotations.json").write_text(json.dumps(payload, indent=2))
    log.info(
        "wrote %d annotations over %d images → %s", len(annotations), len(images_meta), out_dir
    )


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in text.lower()).strip("-")
