"""`motion` command-line interface — the single entrypoint for rendering and
training. Heavy backends are imported lazily inside each command so `motion doctor`
and `motion merge` stay fast and torch-free on the Mac."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="Seamless object-merge motion.")
data_app = typer.Typer(help="Fine-tuning data engine (scrape / extract / autolabel).")
finetune_app = typer.Typer(help="Fine-tune SAM 3 / LocateAnything (GPU box).")
app.add_typer(data_app, name="data")
app.add_typer(finetune_app, name="finetune")


@app.command()
def doctor() -> None:
    """Report device + which model/morph backends are available here."""
    from motion.device import describe
    from motion.morph.registry import available_backends
    from motion.perception.locateanything_cpp import LocateAnythingCpp

    typer.echo("motion doctor\n" + "-" * 40)
    typer.echo(describe())
    typer.echo(f"morph backends  : {', '.join(available_backends())}")
    typer.echo(
        f"locate-anything : {'cpp CLI found' if LocateAnythingCpp.is_available() else 'not found'}"
    )
    try:
        import sam3  # noqa: F401

        typer.echo("sam3            : installed")
    except Exception:
        typer.echo("sam3            : not installed (render-only mode)")


@app.command()
def merge(
    image_a: Path = typer.Argument(..., exists=True),
    image_b: Path = typer.Argument(..., exists=True),
    out: Path = typer.Option("outputs/merge.mp4", "-o", "--out"),
    prompt_a: str | None = typer.Option(None, "--prompt-a"),
    prompt_b: str | None = typer.Option(None, "--prompt-b"),
    mask_a: str | None = typer.Option(None, "--mask-a"),
    mask_b: str | None = typer.Option(None, "--mask-b"),
    backend: str = typer.Option("auto", help="auto | classical | flow"),
    frames: int = typer.Option(48, help="frames per transition"),
    fps: int = typer.Option(30),
    easing: str = typer.Option("ease_in_out"),
    max_side: int = typer.Option(1024, help="downscale longest side (0 = off)"),
    perception: bool = typer.Option(True, help="use SAM 3 / LocateAnything if available"),
    write_frames: bool = typer.Option(False, help="also emit PNG sequence for AE"),
    device: str | None = typer.Option(None, help="cuda | mps | cpu | auto"),
) -> None:
    """Merge two images into one seamless transition video (+ manifest.json)."""
    from motion.perception.registry import build_localizer, build_segmenter
    from motion.pipeline.merge import MergeConfig, MergePipeline

    cfg = MergeConfig(
        backend=backend,
        n_frames=frames,
        fps=fps,
        easing=easing,
        max_side=max_side,
        device=device,
        write_frames=write_frames,
    )
    seg = build_segmenter(device=device) if perception else None
    loc = build_localizer(device=device) if perception else None
    pipe = MergePipeline(cfg, segmenter=seg, localizer=loc)
    manifest = pipe.merge_pair(
        image_a,
        image_b,
        prompt_a=prompt_a,
        prompt_b=prompt_b,
        mask_a=mask_a,
        mask_b=mask_b,
        out_video=out,
    )
    typer.echo(f"✓ {manifest.output_video}  ({manifest.total_frames} frames)")


@data_app.command("extract-frames")
def data_extract_frames(config: Path = typer.Option(..., "--config", exists=True)) -> None:
    """Extract & sample frames from scraped motion media."""
    from motion.data.extract_frames import run_extract

    run_extract(config)


@data_app.command("autolabel")
def data_autolabel(config: Path = typer.Option(..., "--config", exists=True)) -> None:
    """Weakly label frames with base SAM 3 + LocateAnything (GPU)."""
    from motion.data.autolabel import run_autolabel

    run_autolabel(config)


@finetune_app.command("sam3")
def finetune_sam3(config: Path = typer.Option(..., "--config", exists=True)) -> None:
    """Fine-tune SAM 3 (GPU box). See docs/FINETUNING.md."""
    from motion.train.finetune_sam3 import run as run_sam3

    run_sam3(config)


@finetune_app.command("locateanything")
def finetune_locateanything(config: Path = typer.Option(..., "--config", exists=True)) -> None:
    """LoRA fine-tune LocateAnything (GPU box). See docs/FINETUNING.md."""
    from motion.train.finetune_locateanything import run as run_la

    run_la(config)


@app.command("convert-gguf")
def convert_gguf(
    ckpt: Path = typer.Option(..., "--ckpt", help="fine-tuned LocateAnything checkpoint dir"),
    out: Path = typer.Option("checkpoints/gguf", "--out"),
    quant: str = typer.Option("q8_0", help="f16 | q8_0 | q6_k | q5_k | q4_k"),
) -> None:
    """Convert a fine-tuned LocateAnything checkpoint to GGUF for the Mac (ggml)."""
    from motion.train.convert_to_gguf import convert

    path = convert(ckpt, out, quant=quant)
    typer.echo(f"✓ {path}")


if __name__ == "__main__":  # pragma: no cover
    app()
