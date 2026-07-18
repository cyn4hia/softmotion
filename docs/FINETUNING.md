# Fine-tuning SAM 3 & LocateAnything

Goal: make both models accurate on the *domain* of "motion" media (seamless
object-merge / morph transitions that already exist online) so downstream renders
segment and localize objects reliably.

## The pipeline

```
scrape ──▶ extract-frames ──▶ autolabel ──▶ [human review] ──▶ build dataset ──▶ fine-tune ──▶ (LA) convert to GGUF
 (any)        (any)          (GPU best)                        (helper)          (GPU)              (for the Mac)
```

Everything is config-driven; you edit YAML in `configs/`, never code, to launch or
tune a run.

### 1. Collect & prepare data (can run on the Mac)

```bash
motion data extract-frames --config configs/data/extract.yaml
```
(If you scrape: fill `configs/data/sources.example.txt`, then run
`python -c "from motion.data.scrape import run_scrape; run_scrape('configs/data/scrape.yaml')"`.
Only use content you're permitted to train on — see [DATA.md](DATA.md).)

### 2. Weak labels from the base models (GPU best)

```bash
motion data autolabel --config configs/data/autolabel.yaml
```
Writes `data/annotations/annotations.json` + mask PNGs. **Review these** (fix bad
masks/boxes, drop junk) before training — weak labels are a starting point.

### 3. Build trainer-specific datasets

```python
from motion.data.dataset_la import build_la_dataset
from motion.data.dataset_sam3 import build_sam3_dataset
build_la_dataset("data/annotations/annotations.json", "data/datasets/la_train.jsonl")
build_sam3_dataset("data/annotations/annotations.json", "data/datasets/sam3")
```

### 4a. Fine-tune LocateAnything (LoRA)

```bash
# First: set smoke_steps: 10 in the config, run once, confirm it steps + saves.
motion finetune locateanything --config configs/finetune/locateanything_finetune.yaml
```
NVIDIA didn't publish training code, so this is a standard PEFT/LoRA SFT over the
model's chat format. **Two adapter points** are marked `# ADAPT:` in
`train/finetune_locateanything.py` — (1) the chat-template construction and (2) the
processor call signature — both depend on LocateAnything's `trust_remote_code`
modeling. The collator also masks the prompt so loss falls only on the answer boxes;
verify the tokenizer's `padding_side` on the box (the offset assumes right-padding).
Confirm all of this against the model card's `LocateAnythingWorker` example, using
the smoke run.

### 4b. Fine-tune SAM 3

```bash
# Paste the exact command from third_party/sam3/README_TRAIN.md into the config:
#   train_command: "python third_party/sam3/training/train.py --data {dataset} --output {output_dir} ..."
motion finetune sam3 --config configs/finetune/sam3_finetune.yaml
```
We deliberately drive Meta's *official* trainer via a command template rather than
reimplementing it — `{dataset}`, `{output_dir}`, `{config}` are substituted. The
dataset writer's `_encode_record` in `data/dataset_sam3.py` is the single place to
match SAM 3's exact on-disk schema (RLE vs polygon, key names).

### 5. Bring LocateAnything back to the Mac

```bash
motion convert-gguf --ckpt checkpoints/locateanything-ft --out checkpoints/gguf --quant q8_0
```
Merges the LoRA, converts to GGUF via `locate-anything.cpp`, and quantizes. Copy the
`.gguf` to the Mac and point `LOCATE_ANYTHING_GGUF` at it. (SAM 3 fine-tuned weights
stay on the GPU box; on the Mac SAM 3 is optional — supply masks or use GrabCut.)

## Why fine-tune at all

Both models are strong zero-shot but improve markedly on niche domains when tuned on
in-domain labels. "Motion" content has a consistent visual grammar (isolated hero
objects, clean transitions); tuning on it tightens masks/boxes exactly where the
morph quality is most sensitive — the object boundary.

## Verified vs. verify-on-GPU

- ✅ Verified on CPU here: extract-frames, dataset writers, the full render path.
- ⚠️ `VERIFY-ON-GPU` (marked in code): SAM 3 inference/training API, LocateAnything
  PyTorch inference + LoRA collator, GGUF conversion flags. These need the GPU box
  and gated weights; the code is written to the documented APIs with the uncertain
  spots isolated and commented.
