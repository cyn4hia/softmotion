# GPU box setup

For fine-tuning SAM 3 and LocateAnything. You do this once when you get on a CUDA
machine.

## Requirements (from the upstream projects)

- **SAM 3**: Python ≥ 3.12, PyTorch ≥ 2.7, a CUDA GPU (CUDA ≥ 12.6). Checkpoints are
  **gated** on Hugging Face.
- **LocateAnything-3B**: CUDA GPU (Ampere/Hopper/Ada/Blackwell — A100/H100/L40/RTX
  40xx). Runs via `transformers` + `trust_remote_code`.

Recommended: a single A100/H100 (40–80 GB) or an RTX 4090 (24 GB) with LoRA + 4/8-bit
for LocateAnything; SAM 3 fine-tuning follows Meta's own memory guidance.

## One-shot provisioning

```bash
git clone <your motion remote> && cd motion
cp .env.example .env         # put your HF_TOKEN in it
export $(grep -v '^#' .env | xargs)   # or use direnv
bash scripts/setup_gpu.sh    # venv, torch+cu128, motion[gpu,train,data,dev], clones SAM 3, HF login
motion doctor                # should now report cuda + sam3 installed
```

`setup_gpu.sh` clones `facebookresearch/sam3` into `third_party/sam3` and installs it
editable, so its `README_TRAIN.md` is right there for the exact training command.

## CUDA wheel selection

`setup_gpu.sh` defaults to `cu128`. Override to match your driver:

```bash
CUDA_WHL=cu124 PY_VER=3.12 bash scripts/setup_gpu.sh
```

## Gated SAM 3 checkpoints

1. Request access: <https://huggingface.co/facebook/sam3> (and `facebook/sam3.1`).
2. `huggingface-cli login` (uses `HF_TOKEN`).
3. First `Sam3Segmenter` use downloads them, or pre-fetch:
   `huggingface-cli download facebook/sam3 --local-dir checkpoints/sam3`.

## Sanity check before a long run

```bash
make smoke                                   # CPU render still works
motion data extract-frames --config configs/data/extract.yaml
motion data autolabel      --config configs/data/autolabel.yaml
# set smoke_steps: 10 in the LA config first:
motion finetune locateanything --config configs/finetune/locateanything_finetune.yaml
```

Then continue with [FINETUNING.md](FINETUNING.md).
