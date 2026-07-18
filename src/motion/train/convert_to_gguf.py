"""Convert a fine-tuned LocateAnything checkpoint → GGUF so it runs on the Mac via
locate-anything.cpp (ggml). Wraps the upstream conversion script + the CLI's
quantizer.

Pipeline:
  fine-tuned HF checkpoint  ──(merge LoRA)──▶  merged HF model
                            ──(convert)────▶  <out>/locate-anything-f32.gguf
                            ──(quantize)───▶  <out>/locate-anything-<quant>.gguf

Set LOCATE_ANYTHING_CPP_DIR to your clone of mudler/locate-anything.cpp
(scripts/setup_mac.sh clones it to third_party/). ⚠️ VERIFY the convert script's
exact flag names against that repo — they're isolated in `_convert_cmd` below.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from motion.config import resolve_path
from motion.logging import get_logger

log = get_logger(__name__)


def _cpp_dir() -> Path:
    d = os.getenv("LOCATE_ANYTHING_CPP_DIR", "third_party/locate-anything.cpp")
    path = resolve_path(d)
    if not path.exists():
        raise RuntimeError(
            f"locate-anything.cpp not found at {path}. Clone it (scripts/setup_mac.sh) "
            "or set LOCATE_ANYTHING_CPP_DIR."
        )
    return path


def _maybe_merge_lora(ckpt: Path) -> Path:
    """If ``ckpt`` is a PEFT adapter, merge it into the base model first."""
    if not (ckpt / "adapter_config.json").exists():
        return ckpt
    try:
        import torch  # noqa: F401
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoProcessor
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "peft/transformers needed to merge LoRA. pip install -e '.[train]'"
        ) from e

    merged = ckpt.parent / (ckpt.name + "-merged")
    log.info("merging LoRA adapter → %s", merged)
    model = AutoPeftModelForCausalLM.from_pretrained(str(ckpt), trust_remote_code=True)
    model = model.merge_and_unload()
    model.save_pretrained(str(merged))
    try:
        AutoProcessor.from_pretrained(str(ckpt), trust_remote_code=True).save_pretrained(
            str(merged)
        )
    except Exception:
        log.warning("could not copy processor; copy it into %s manually if needed", merged)
    return merged


def _convert_cmd(cpp: Path, model_dir: Path, out_f32: Path) -> list[str]:
    # ADAPT: confirm script name/flags against locate-anything.cpp/scripts/.
    return [
        "python",
        str(cpp / "scripts" / "convert_locateanything_to_gguf.py"),
        "--model",
        str(model_dir),
        "--outfile",
        str(out_f32),
    ]


def convert(ckpt: str | Path, out_dir: str | Path, quant: str = "q8_0") -> Path:
    cpp = _cpp_dir()
    model_dir = _maybe_merge_lora(resolve_path(ckpt))
    out = resolve_path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_f32 = out / "locate-anything-f32.gguf"

    log.info("converting %s → %s", model_dir, out_f32)
    subprocess.run(_convert_cmd(cpp, model_dir, out_f32), check=True)

    if quant in ("f32", "f16"):
        final = out_f32 if quant == "f32" else out / "locate-anything-f16.gguf"
        if quant == "f16":
            _quantize(out_f32, final, "f16")
        return final

    final = out / f"locate-anything-{quant}.gguf"
    _quantize(out_f32, final, quant)
    return final


def _quantize(src: Path, dst: Path, quant: str) -> None:
    cli = os.getenv("LOCATE_ANYTHING_CLI", "locate-anything-cli")
    log.info("quantizing → %s (%s)", dst, quant)
    subprocess.run([cli, "quantize", str(src), str(dst), quant], check=True)
