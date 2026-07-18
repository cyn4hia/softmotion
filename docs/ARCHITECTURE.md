# Architecture

## The idea

Two images, each with an object. `motion` **perceives** each object, finds a
**correspondence** between them, **morphs** A→B into a frame sequence, **composes**
it into a video, and emits a **manifest** describing the whole thing. The manifest's
object/segment lists are a *chain*, so N objects (A→B→C…) is the same code path with
more entries — no redesign for the "more objects later" goal.

```
                 ┌─────────────┐
image_a ────────▶│  perceive   │  SAM 3 (mask) + LocateAnything (box/point)
image_b ────────▶│ (optional)  │  → falls back to GrabCut / mask-free
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │ correspond  │  mask → contour → align → +frame → Delaunay
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │   morph     │  classical mesh-warp  |  optical-flow
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │  compose    │  cross-dissolve / feather / Poisson  → ffmpeg
                 └──────┬──────┘
                        ▼
              transition.mp4  +  manifest.json  (+ PNG sequence for AE)
```

## Layers (dependency direction points downward — no cycles)

| Package | Role | GPU? |
|---|---|---|
| `motion.device` | one place that resolves cuda/mps/cpu | detects, never requires |
| `motion.io` | image / video / mask I/O | no |
| `motion.correspondence` | contour matching + triangulation | no |
| `motion.morph` | the render engines + compositor | no |
| `motion.perception` | SAM 3 + LocateAnything behind ABCs | lazy (only if used) |
| `motion.pipeline` | end-to-end orchestration + manifest | no |
| `motion.data` | fine-tuning data engine | partial |
| `motion.train` | GPU fine-tuning + GGUF export | yes (lazy) |
| `motion.cli` | `motion …` entrypoint | lazy per-command |

## The two-runtime design ("just swap to a GPU")

The single most important rule: **importing `motion` never imports torch.** Every
heavy dependency (torch, transformers, sam3, peft) is imported *inside the function
that needs it*, guarded by optional extras. Consequences:

- On the **Mac**, `pip install -e .` gives a full render pipeline with zero GPU deps.
- On the **GPU box**, `pip install -e ".[gpu,train]"` + `scripts/setup_gpu.sh` adds
  the models and training stack. Nothing in the shared code changes.
- `motion.device.resolve_device()` and the `perception.registry` factories mean the
  same command auto-selects CUDA, Apple MPS, or CPU — and degrade gracefully to a
  mask-free morph when no model is present.

## Swappability

- **Morph backends** register by name (`morph.registry`). Add a GPU diffusion morph
  by implementing `MorphBackend.render` and `@register_backend` — the pipeline, CLI,
  and manifest pick it up for free.
- **Perception backends** sit behind `Segmenter` / `Localizer` ABCs. This is how we
  contain the LocateAnything license problem: drop in a commercially-licensed
  grounding model without touching the pipeline (see [LICENSING.md](LICENSING.md)).

## The manifest is the API

`pipeline/manifest.py` is a versioned pydantic schema. After Effects (and any future
host) drives a render through the CLI and reads the manifest back. Treat it like a
public API: additive changes only; bump `version` on a break.

## Directory map

```
configs/        YAML for every stage (edit, don't code, to tune/launch)
src/motion/     the package (layers above)
scripts/        setup_gpu.sh, setup_mac.sh, download_models.py, smoke_test.py
ae/             After Effects bridge (ExtendScript → CLI → manifest)
tests/          CPU-path tests (run today)
docs/           this folder
data/ checkpoints/ outputs/ runs/ third_party/   gitignored artifacts
```
