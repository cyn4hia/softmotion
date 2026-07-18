#!/usr/bin/env python3
"""Create the gitignored artifact directory skeleton with .gitkeep files."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DIRS = [
    "checkpoints/gguf",
    "data/raw",
    "data/frames",
    "data/annotations/masks",
    "data/datasets",
    "outputs",
    "runs",
    "third_party",
]


def main() -> None:
    for d in DIRS:
        p = ROOT / d
        p.mkdir(parents=True, exist_ok=True)
        (p / ".gitkeep").touch()
        print(f"  {d}/")
    print("done.")


if __name__ == "__main__":
    main()
