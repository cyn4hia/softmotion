"""Convert neutral autolabels → LocateAnything supervised fine-tuning samples.

Each sample is a chat turn pairing the detection instruction with the target box
string in LocateAnything's own format (`<box>x1 y1 x2 y2</box>`, coords normalized
to [0, 1000]). Emitted as JSONL, consumed by
:mod:`motion.train.finetune_locateanything`.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from motion.config import resolve_path
from motion.logging import get_logger
from motion.perception.locateanything_localizer import DETECT_TMPL

log = get_logger(__name__)


def _box_to_tokens(box: list[float], w: int, h: int) -> str:
    x1, y1, x2, y2 = box
    n = [
        round(x1 / w * 1000),
        round(y1 / h * 1000),
        round(x2 / w * 1000),
        round(y2 / h * 1000),
    ]
    n = [max(0, min(1000, v)) for v in n]
    return f"<box>{n[0]} {n[1]} {n[2]} {n[3]}</box>"


def build_la_dataset(annotations_json: str | Path, out_jsonl: str | Path) -> Path:
    ann_path = resolve_path(annotations_json)
    data = json.loads(ann_path.read_text())
    images = {im["id"]: im for im in data["images"]}

    by_image_concept: dict[tuple[int, str], list[list[float]]] = defaultdict(list)
    for a in data["annotations"]:
        if a.get("box"):
            by_image_concept[(a["image_id"], a["concept"])].append(a["box"])

    out = resolve_path(out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as f:
        for (img_id, concept), boxes in by_image_concept.items():
            im = images[img_id]
            target = "".join(_box_to_tokens(b, im["width"], im["height"]) for b in boxes)
            sample = {
                "image": im["file_name"],
                "conversations": [
                    {"role": "user", "content": DETECT_TMPL.format(q=concept)},
                    {"role": "assistant", "content": target},
                ],
            }
            f.write(json.dumps(sample) + "\n")
            n += 1
    log.info("wrote %d LocateAnything SFT samples → %s", n, out)
    return out
