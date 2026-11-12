"""Polite, resumable HTTP: every page is cached on disk, a re-run only fetches what is missing."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

from . import config

log = logging.getLogger("hfzakon.fetch")

_session: requests.Session | None = None
_robots_disallow: list[str] | None = None
_last_request = 0.0


def session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers["User-Agent"] = config.USER_AGENT
    return _session


def nn_disallowed(url: str) -> bool:
    """True when narodne-novine robots.txt disallows the path (old blocked documents, oglasi)."""
    global _robots_disallow
    if _robots_disallow is None:
        try:
            txt = session().get(config.NN_BASE + "/robots.txt", timeout=config.TIMEOUT_S).text
            _robots_disallow = [line.split(":", 1)[1].strip()
                                for line in txt.splitlines()
                                if line.lower().startswith("disallow:") and line.split(":", 1)[1].strip()]
        except requests.RequestException:
            _robots_disallow = ["/clanci/oglasi/"]
        _robots_disallow = [d for d in _robots_disallow if "*" not in d]  # literal prefixes only
    path = url.replace(config.NN_BASE, "")
    return any(path.startswith(d) for d in _robots_disallow)


def get(url: str) -> bytes | None:
    """GET with retry/backoff and a global politeness delay. None on a final 404."""
    global _last_request
    for attempt in range(1, config.RETRIES + 1):
        wait = config.DELAY_S - (time.monotonic() - _last_request)
        if wait > 0:
            time.sleep(wait)
        try:
            _last_request = time.monotonic()
            r = session().get(url, timeout=config.TIMEOUT_S)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except requests.RequestException as exc:
            if attempt == config.RETRIES:
                raise
            log.warning("retry %d for %s (%s)", attempt, url, exc)
            time.sleep(2 * attempt)
    return None  # unreachable


def get_cached(url: str, cache: Path) -> bytes | None:
    """Disk-cached GET. An empty cache file marks a known 404 so re-runs skip it."""
    if cache.exists():
        data = cache.read_bytes()
        return data or None
    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = get(url)
    except requests.RequestException as exc:
        config.STATE_DIR.mkdir(parents=True, exist_ok=True)
        with (config.STATE_DIR / "errors.log").open("a", encoding="utf-8") as f:
            f.write(f"{url}\t{exc}\n")
        return None
    cache.write_bytes(data or b"")
    return data
