# Licensing — read before shipping commercially

`motion` itself is MIT (see `LICENSE`). The **models** you fine-tune and ship have
their own terms, and they differ in a way that matters for a future commercial /
After Effects product.

| Model | License | Commercial use? |
|---|---|---|
| **SAM 3** (Meta) | "SAM License" (custom) | ✅ **Permitted**, with restrictions (no military/warfare, no ITAR/trade-controlled uses; must pass the license along with any redistribution of the materials or derivatives). |
| **LocateAnything-3B** (NVIDIA) | NVIDIA non-commercial license | ❌ **Research / academic / non-profit only.** Commercial use is *not* permitted (except by NVIDIA & affiliates). |
| **locate-anything.cpp** (ggml port) | its own repo license (permissive) | ✅ code — but the **model weights it runs are still LocateAnything's** and carry NVIDIA's terms. |

## What this means

- Fine-tuning LocateAnything and using it for **research/prototyping** is fine.
- Any **commercial** release of `motion` (paid product, client work, a sold AE
  plugin) **cannot ship LocateAnything** or its fine-tuned derivatives.
- SAM 3 is fine to ship commercially as long as you honor its restrictions and
  redistribution clause. **Verify the current SAM License text yourself** before
  relying on this — licenses change.

## How the codebase protects you

Localization sits behind the `Localizer` ABC (`perception/base.py`) with a registry
(`perception/registry.py`). To go commercial, implement one new class — e.g. a
`GroundingDinoLocalizer` (Apache-2.0) or another permissively-licensed
open-vocabulary detector — register it, and every downstream stage keeps working. No
pipeline changes.

Recommended commercially-friendly swap-ins to evaluate:
- **GroundingDINO** (Apache-2.0) — open-vocab detection, closest drop-in.
- **OWLv2** (Apache-2.0) — open-vocab detection.
- **YOLO-World** (check the specific weight's license) — fast open-vocab detection.

Keep LocateAnything strictly for the research/fine-tuning track; treat the
`Localizer` interface as the seam where you switch models for production.

> This document is engineering guidance, not legal advice. Confirm each model's
> current license before any commercial use.
