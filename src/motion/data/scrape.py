"""Collect existing 'motion' media (seamless object-merge / morph transitions) to
fine-tune on. Config-driven list of source URLs, downloaded via yt-dlp.

⚠️ LEGAL: only download content you have the right to use. Respect each platform's
Terms of Service and creators' copyright/licences. This tool is a convenience for
material you are permitted to train on; curating a lawful source list is on you.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from motion.config import load_yaml, resolve_path
from motion.logging import get_logger

log = get_logger(__name__)


class ScrapeConfig(BaseModel):
    sources: list[str] = []  # URLs or a path to a newline file of URLs
    out_dir: str = "data/raw"
    format: str = "bestvideo[height<=1080]+bestaudio/best[height<=1080]"
    limit: int | None = None


def _expand_sources(sources: list[str]) -> list[str]:
    urls: list[str] = []
    for s in sources:
        p = Path(s)
        if p.exists() and p.suffix in (".txt", ".list"):
            urls += [
                ln.strip()
                for ln in p.read_text().splitlines()
                if ln.strip() and not ln.startswith("#")
            ]
        else:
            urls.append(s)
    return urls


def run_scrape(config_path: str | Path) -> None:
    cfg = ScrapeConfig.model_validate(load_yaml(config_path))
    try:
        import yt_dlp  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise RuntimeError("yt-dlp not installed. `pip install -e '.[data]'`.") from e

    out = resolve_path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    urls = _expand_sources(cfg.sources)
    if cfg.limit:
        urls = urls[: cfg.limit]
    log.info("Downloading %d source(s) to %s", len(urls), out)

    opts = {"format": cfg.format, "outtmpl": str(out / "%(id)s.%(ext)s"), "ignoreerrors": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download(urls)
    log.info("scrape complete")
