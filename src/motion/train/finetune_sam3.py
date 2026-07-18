"""Fine-tune SAM 3 (GPU box).

SAM 3 ships its own training pipeline (see `README_TRAIN.md` in
facebookresearch/sam3, installed via `pip install -e '.[train,dev]'`). Rather than
reimplement Meta's trainer, this driver:

  1. builds the dataset (neutral autolabels → SAM 3 format) if needed,
  2. invokes the official training entrypoint via a **command template** you fill
     in from `README_TRAIN.md` (so we track the exact upstream CLI without guessing).

The command template + hyperparameters live entirely in the YAML, so launching or
tuning a run never touches code.

⚠️ VERIFY-ON-GPU: set `train_command` to the exact command from SAM 3's
`README_TRAIN.md`. Placeholders `{dataset}`, `{output_dir}`, `{config}` are filled.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from pydantic import BaseModel

from motion.config import load_yaml, resolve_path
from motion.logging import get_logger

log = get_logger(__name__)


class Sam3FinetuneConfig(BaseModel):
    annotations_json: str = "data/annotations/annotations.json"
    dataset_dir: str = "data/datasets/sam3"
    output_dir: str = "checkpoints/sam3-ft"
    build_dataset: bool = True
    # Exact upstream training invocation from sam3/README_TRAIN.md. Example shape:
    #   "python training/train.py --config {config} --dataset {dataset} --output {output_dir}"
    train_command: str | None = None
    train_config: str | None = None  # path to SAM 3's own training config, if any
    extra_env: dict[str, str] = {}


def run(config_path: str | Path) -> None:
    cfg = Sam3FinetuneConfig.model_validate(load_yaml(config_path))

    dataset_path: Path
    if cfg.build_dataset:
        from motion.data.dataset_sam3 import build_sam3_dataset

        dataset_path = build_sam3_dataset(cfg.annotations_json, cfg.dataset_dir)
    else:
        dataset_path = resolve_path(cfg.dataset_dir)
    output_dir = resolve_path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.train_command:
        raise RuntimeError(
            "Set `train_command` in the config to SAM 3's training CLI from its "
            "README_TRAIN.md. Dataset is ready at: "
            f"{dataset_path}\nOutput dir: {output_dir}"
        )

    import os

    cmd = cfg.train_command.format(
        dataset=str(dataset_path),
        output_dir=str(output_dir),
        config=str(resolve_path(cfg.train_config)) if cfg.train_config else "",
    )
    env = {**os.environ, **cfg.extra_env}
    log.info("SAM 3 training: %s", cmd)
    proc = subprocess.run(cmd, shell=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"SAM 3 training exited with code {proc.returncode}")
    log.info("SAM 3 fine-tune complete → %s", output_dir)
