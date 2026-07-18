"""Config loading. All tunables live in ``configs/*.yaml`` so the GPU box only
needs a YAML edit — never a code change — to launch or adjust training.

`load_config` reads a YAML file, applies ``${ENV_VAR}`` / ``${ENV:default}``
substitution, and (optionally) validates it against a pydantic model.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[2]

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")

T = TypeVar("T", bound=BaseModel)


def _expand(value: Any) -> Any:
    if isinstance(value, str):

        def sub(m: re.Match[str]) -> str:
            var, default = m.group(1), m.group(2)
            if var in os.environ:
                return os.environ[var]
            if default is not None:  # `${VAR:default}` — empty default allowed
                return default
            raise KeyError(
                f"config references ${{{var}}} but it is unset and has no default. "
                "Set it in the environment, or give a default like ${VAR:fallback} in the YAML."
            )

        return _ENV_RE.sub(sub, value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _expand(raw)


def load_config(path: str | Path, model: type[T]) -> T:
    """Load YAML and validate into a pydantic model."""
    return model.model_validate(load_yaml(path))


def resolve_path(value: str | Path) -> Path:
    """Resolve a possibly-relative config path against the repo root."""
    p = Path(value)
    return p if p.is_absolute() else REPO_ROOT / p
