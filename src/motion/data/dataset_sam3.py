"""Convert neutral autolabels → a SAM 3 fine-tuning dataset.

SAM 3 trains on concept-grounded masks (the SA-Co format: images + noun phrases +
per-phrase masks). This writer emits that logical structure (images + a JSONL of
`{image, phrase, mask, box}` records with RLE masks).

⚠️ VERIFY-ON-GPU: the *exact* on-disk schema SAM 3's trainer expects lives in
`README_TRAIN.md` inside facebookresearch/sam3 (not runnable here). Keep the record
shape below as the single adapter point: if the trainer wants polygons instead of
RLE, or a different key layout, change only :func:`_encode_record`. Everything
upstream (autolabel, review) is trainer-agnostic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from motion.config import resolve_path
from motion.io.masks import load_mask
from motion.logging import get_logger

log = get_logger(__name__)


def _rle_encode(mask: np.ndarray) -> dict:
    """COCO-style uncompressed RLE (column-major), which SAM tooling understands.

    COCO RLE always begins with a run of background (0) pixels. We seed ``prev=0``
    so a mask that starts with foreground automatically emits a leading 0-count —
    no special-casing (adding one manually would double it).
    """
    m = np.asarray(mask, dtype=np.uint8, order="F")
    flat = m.flatten(order="F")
    counts: list[int] = []
    prev = 0
    run = 0
    for v in flat:
        if v == prev:
            run += 1
        else:
            counts.append(int(run))
            prev = int(v)
            run = 1
    counts.append(int(run))
    return {"size": [int(m.shape[0]), int(m.shape[1])], "counts": counts}


def _encode_record(image_path: str, phrase: str, mask: np.ndarray, box) -> dict:
    """SINGLE ADAPTER POINT — align this dict with sam3/README_TRAIN.md."""
    return {
        "image": image_path,
        "phrase": phrase,
        "box": list(box) if box else None,
        "segmentation": _rle_encode(mask),
    }


def build_sam3_dataset(annotations_json: str | Path, out_dir: str | Path) -> Path:
    ann_path = resolve_path(annotations_json)
    base = ann_path.parent
    data = json.loads(ann_path.read_text())
    images = {im["id"]: im for im in data["images"]}

    out = resolve_path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "sam3_train.jsonl"
    n = 0
    with jsonl.open("w") as f:
        for a in data["annotations"]:
            if not a.get("mask_file"):
                continue
            im = images[a["image_id"]]
            mask = load_mask(base / a["mask_file"])
            rec = _encode_record(im["file_name"], a["concept"], mask, a.get("box"))
            f.write(json.dumps(rec) + "\n")
            n += 1
    log.info("wrote %d SAM 3 records → %s", n, jsonl)
    return jsonl
