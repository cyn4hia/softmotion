# motion

**Take two completely separate objects and generate a single seamless motion that merges them into one.**

Given two images (each containing an object), `motion` segments and localizes each
object, establishes a correspondence between them, and renders a continuous
transition video that morphs object A into object B. The architecture generalizes
to *N* objects (a chain of morph segments) and is designed to eventually be driven
as a script from Adobe After Effects.

```
image_a ─┐
         ├─▶ perceive ─▶ correspond ─▶ morph ─▶ compose ─▶ transition.mp4 + manifest.json
image_b ─┘   (SAM 3 +     (mask/          (CPU        (blend,
              LocateAny)   keypoints)      classical    ffmpeg)
                                           or flow)
```

## Two runtimes, one codebase

| | **Mac mini (CPU / Apple Silicon)** | **GPU box (CUDA)** |
|---|---|---|
| Purpose | Render the transition videos | Fine-tune SAM 3 & LocateAnything |
| Install | `pip install -e .` | `pip install -e ".[gpu,train]"` + `scripts/setup_gpu.sh` |
| Segmentation | SAM 3 on MPS/CPU *(slow, optional)* or supply masks | SAM 3 on CUDA |
| Localization | LocateAnything via `locate-anything.cpp` (ggml, Metal/CPU) | LocateAnything in PyTorch (CUDA) |
| Morph engine | ✅ classical mesh-warp + optical-flow (pure NumPy/OpenCV) | ✅ same, or optional diffusion backend |

The **morph engine has zero GPU dependencies** — it runs today on the Mac. The
**fine-tuning code is written to the models' documented APIs and gated behind
optional extras**, so on a GPU machine you `git pull`, run one setup script, edit a
YAML, and launch training. Nothing needs rewriting to "swap over."

## Quick start (Mac / CPU, works now)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
motion doctor                      # shows which backends are available
motion merge examples/a.png examples/b.png -o outputs/merge.mp4 \
    --prompt-a "a red apple" --prompt-b "a green pear"
```

If SAM 3 weights aren't present, `merge` falls back to mask-free correspondence
(optical flow or GrabCut auto-segmentation) so you always get a video.

## Fine-tuning (GPU box)

```bash
ssh gpu-box && cd motion
bash scripts/setup_gpu.sh          # torch+cu128, clone SAM 3, HF login
motion data extract-frames --config configs/data/extract.yaml
motion data autolabel   --config configs/data/autolabel.yaml   # weak labels from base models
# review labels, then build datasets (see docs/FINETUNING.md), then:
motion finetune sam3            --config configs/finetune/sam3_finetune.yaml
motion finetune locateanything  --config configs/finetune/locateanything_finetune.yaml
motion convert-gguf --ckpt checkpoints/locateanything-ft --out checkpoints/gguf   # back to Mac
```

## Layout

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Highlights:

- `src/motion/perception/` — SAM 3 + LocateAnything wrappers behind swappable ABCs
- `src/motion/correspondence/` — mask→contour→triangulation matching
- `src/motion/morph/` — the CPU morph backends (classical, optical-flow) + compositor
- `src/motion/pipeline/` — end-to-end `merge` + the JSON `manifest` contract for AE
- `src/motion/data/` + `src/motion/train/` — the GPU fine-tuning path
- `ae/` — After Effects bridge (ExtendScript → CLI → manifest)

## ⚠️ Licensing (read before shipping)

- **SAM 3** — Meta "SAM License": commercial use *permitted* (with restrictions).
- **LocateAnything-3B** — NVIDIA license: **non-commercial / research only.**
  The `Localizer` interface is deliberately swappable so you can drop in a
  commercially-licensed grounding model (e.g. GroundingDINO) before any
  commercial/AE release. See [docs/LICENSING.md](docs/LICENSING.md).
