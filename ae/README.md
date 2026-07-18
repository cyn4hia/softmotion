# After Effects integration

`motion` is designed so AE never needs to know how the morph works — it only speaks
the **manifest contract** ([`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md)).

## How it works

```
AE (ExtendScript / UXP)
   │  system.callSystem("motion merge a.png b.png -o out.mp4 --write-frames")
   ▼
motion CLI  ──▶  out.mp4  +  out.json (manifest)  +  out/seg_00/*.png (PNG sequence)
   │
   ▼
AE imports the video (or the PNG sequence) and reads out.json to place layers,
set frame ranges, and label objects.
```

Because the CLI is the only coupling point, the same call works whether the morph
ran on this Mac or was rendered on a farm — and the manifest's object/segment list
already supports N-object merges.

## Files

- `motion_bridge.jsx` — ExtendScript proof-of-concept: runs the CLI, imports the
  result, and reads the manifest. Run via **File ▸ Scripts ▸ Run Script File…**
  (enable *Allow Scripts to Write Files and Access Network* in preferences).

## Recommended output for AE

Render with `--write-frames` so you get a lossless PNG sequence per segment
alongside the mp4; import whichever suits your comp. The manifest gives you
`segments[].start_frame` and `frames_dir` to drive time remapping and layer
placement.

## Path notes

`motion_bridge.jsx` expects the `motion` CLI on PATH, or set `MOTION_BIN` at the top
of the script to the venv's `motion` (e.g. `/Users/you/coding/motion/.venv/bin/motion`).
A future UXP/CEP panel can wrap the same calls with a GUI.
