"""Shared helpers for the data pipeline: polite cached HTTP, paths, hashing."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import requests

USER_AGENT = "templexai-research/0.1 (academic project; contact: placeholder@example.org)"
MAX_RPS = 5.0

_local = threading.local()
_lock = threading.Lock()
_last_call = 0.0


def add_data_root_arg(parser) -> None:
    """Add the --data-root / --smoke options every stage honours."""
    parser.add_argument("--data-root", default="data", help="data/ or data_smoke/")
    parser.add_argument("--smoke", action="store_true", help="tiny run (≈5 temples)")


def session() -> requests.Session:
    """One Session per thread."""
    if getattr(_local, "s", None) is None:
        _local.s = requests.Session()
        _local.s.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"})
    return _local.s


def _throttle() -> None:
    """Global limiter shared by all threads: request *starts* are >= 1/MAX_RPS apart."""
    global _last_call
    with _lock:
        gap = 1.0 / MAX_RPS
        now = time.time()
        start = max(now, _last_call + gap)
        _last_call = start
    if start > now:
        time.sleep(start - now)


def cache_key(url: str, params: dict[str, Any] | None) -> str:
    raw = url + json.dumps(params or {}, sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()


def get_json(url: str, params: dict[str, Any] | None, cache_dir: Path, *, retries: int = 6,
             timeout: int = 90, headers: dict[str, str] | None = None) -> dict:
    """GET JSON with disk cache, <=5 req/s and exponential backoff."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / (cache_key(url, params) + ".json")
    if f.exists():
        return json.loads(f.read_text())
    delay = 2.0
    last: Exception | None = None
    for _ in range(retries):
        _throttle()
        try:
            r = session().get(url, params=params, timeout=timeout, headers=headers)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{r.status_code}")
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and "error" in data:  # API-level error (e.g. rate limit): retry, never cache
                raise ValueError(str(data["error"])[:120])
            f.write_text(json.dumps(data))
            return data
        except (requests.RequestException, ValueError) as e:  # noqa: PERF203
            last = e
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise RuntimeError(f"GET {url} failed after {retries} tries: {last}")


def download_file(url: str, dest: Path, *, retries: int = 5, timeout: int = 120) -> bool:
    """Download a binary with backoff. Returns False on permanent failure."""
    delay = 2.0
    for _ in range(retries):
        _throttle()
        try:
            r = session().get(url, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(str(r.status_code))
            if r.status_code == 404:
                return False
            r.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(r.content)
            return True
        except requests.RequestException:
            time.sleep(delay)
            delay = min(delay * 2, 60)
    return False


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def today() -> str:
    return time.strftime("%Y-%m-%d")
