# Rendering on a Mac (CPU / Apple Silicon)

The render pipeline runs entirely on CPU. LocateAnything — despite being an NVIDIA
CUDA project — *is* usable on the Mac through the community **ggml port**
(`mudler/locate-anything.cpp`), which runs on CPU and Apple-Silicon Metal. SAM 3 has
no official CPU/MPS path, so on the Mac you either provide masks, let SAM 3 run on
MPS (slow, best-effort), or fall back to GrabCut.

## What runs where on the Mac

| Component | Mac support |
|---|---|
| Classical mesh-warp morph | ✅ native (NumPy/OpenCV) |
| Optical-flow morph | ✅ native |
| LocateAnything (grounding) | ✅ via `locate-anything.cpp` (Metal/CPU) |
| SAM 3 (masks) | ⚠️ optional — supply masks, GrabCut fallback, or MPS best-effort |

## Setup

```bash
bash scripts/setup_mac.sh
# then add the printed lines to .env:
#   LOCATE_ANYTHING_CLI=.../third_party/locate-anything.cpp/build/bin/locate-anything-cli
#   LOCATE_ANYTHING_GGUF=.../checkpoints/gguf/locate-anything-q8_0.gguf
motion doctor
```

## Rendering

```bash
# Fully model-free (works with nothing installed but the CPU core):
motion merge a.png b.png -o outputs/merge.mp4 --backend flow

# With LocateAnything grounding (ggml) seeding GrabCut masks + mesh morph:
motion merge a.png b.png -o outputs/merge.mp4 --prompt-a "sneaker" --prompt-b "boot"

# Supply your own masks (highest control, no models needed):
motion merge a.png b.png -o outputs/merge.mp4 --backend classical \
    --mask-a a_mask.png --mask-b b_mask.png

# Emit an After Effects PNG sequence too:
motion merge a.png b.png -o outputs/merge.mp4 --write-frames
```

## Backend choice

- `classical` — mesh warp; needs a mask per object; best for shape-to-shape morphs.
- `flow` — optical flow; no masks; best when the two frames are visually related.
- `auto` (default) — `classical` when both masks are available, else `flow`.

## Performance tips

- `--max-side 768` (or lower) speeds up large inputs substantially.
- `q8_0` GGUF is the recommended LocateAnything quant (box-identical to f32, ~6.3 GB).
  Use `q5_k`/`q4_k` for smaller/faster at some accuracy cost.
- The GGML CLI supports `--mode fast|hybrid|slow`; `hybrid` (default) balances speed
  and quality.
