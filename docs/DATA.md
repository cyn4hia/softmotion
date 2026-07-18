# The fine-tuning data engine

Turns raw "motion" media into labeled datasets for SAM 3 and LocateAnything.

## Stages

1. **scrape** (`motion.data.scrape`) — download source videos from a URL list via
   yt-dlp. Config: `configs/data/scrape.yaml`.
2. **extract-frames** (`motion data extract-frames`) — uniform or scene-change frame
   sampling. Config: `configs/data/extract.yaml`.
3. **autolabel** (`motion data autolabel`) — run base SAM 3 + LocateAnything to emit
   weak labels (`data/annotations/annotations.json` + mask PNGs). Config:
   `configs/data/autolabel.yaml`.
4. **review** (human) — fix/prune weak labels. This is the highest-leverage step for
   final accuracy.
5. **build datasets** — `dataset_la.build_la_dataset` and
   `dataset_sam3.build_sam3_dataset` convert the reviewed annotations into each
   trainer's format.

## The neutral annotation schema

`autolabel` writes a trainer-agnostic file so review and dataset-building don't care
which trainer comes next:

```json
{
  "concepts": ["the main object", "person"],
  "images": [{"id": 0, "file_name": "data/frames/clip_00001.jpg", "width": 1920, "height": 1080}],
  "annotations": [
    {"id": 0, "image_id": 0, "concept": "person", "box": [x1,y1,x2,y2], "score": 0.94, "mask_file": "masks/clip_00001__person.png"}
  ]
}
```

- `box` is pixel `xyxy`. `mask_file` is a PNG relative to the annotations dir.
- LocateAnything dataset: `box` → `<box>` tokens normalized to [0,1000].
- SAM 3 dataset: `mask_file` → RLE via `_encode_record` (the one place to match SAM
  3's exact schema from its `README_TRAIN.md`).

## Directory layout (gitignored)

```
data/
  raw/                     downloaded source videos
  frames/<clip>/*.jpg      extracted frames
  annotations/
    annotations.json       neutral labels
    masks/*.png            weak masks
  datasets/
    la_train.jsonl         LocateAnything SFT
    sam3/sam3_train.jsonl  SAM 3 records
```

## Legal & ethics

Only collect and train on media you have the right to use. Respect each platform's
Terms of Service and creators' copyright/licences. `scrape.py` is a convenience for
lawful sources; curating that list is your responsibility.

## Tips for good "motion" data

- Prefer clips with a single, clearly isolated hero object per shot.
- Sample near transition boundaries (scene-change mode, `scene_threshold > 0`) to
  capture the object at extremes — that's where morph correspondence matters most.
- Balance concepts so the model doesn't overfit to one object category.
