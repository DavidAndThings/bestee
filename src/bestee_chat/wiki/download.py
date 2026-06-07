"""Resumable, checksum-verified download of dump files.

Built on the project's existing ``httpx`` dependency. Downloads stream to a
``.part`` file and resume via HTTP ``Range`` if interrupted, then verify the
sha1 published by Wikimedia before being atomically renamed into place. A
polite, identifying ``User-Agent`` is sent, as Wikimedia requests.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

import httpx

from bestee_chat.wiki._util import ProgressCallback
from bestee_chat.wiki.cache import Manifest, get_layout
from bestee_chat.wiki.sources import WikiSource

logger = logging.getLogger(__name__)

_USER_AGENT = "bestee-chat/0.1 (https://github.com/bestee-chat) wiki-dump"
_TIMEOUT = httpx.Timeout(30.0, read=120.0)
_CHUNK = 1 << 20  # 1 MiB


def parse_sha1sums(text: str, filename: str) -> str | None:
    """Return the sha1 for ``filename`` from a ``*-sha1sums.txt`` body."""
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == filename:
            return parts[0].lower()
    return None


def fetch_expected_sha1(source: WikiSource) -> str | None:
    """Fetch and parse the published sha1 for ``source``'s data file."""
    try:
        resp = httpx.get(
            source.checksums_url,
            headers={"User-Agent": _USER_AGENT},
            follow_redirects=True,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
    except httpx.HTTPError:
        logger.warning("could not fetch checksums from %s", source.checksums_url)
        return None
    return parse_sha1sums(resp.text, source.data_filename)


def _sha1_of(path: Path) -> str:
    """Return the hex sha1 digest of the file at ``path``."""
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(
    url: str,
    dest: Path,
    *,
    expected_sha1: str | None = None,
    progress: ProgressCallback | None = None,
    force: bool = False,
) -> Path:
    """Download ``url`` to ``dest``, resuming and verifying when possible.

    If ``dest`` already exists and matches ``expected_sha1`` (or no checksum is
    known), the download is skipped. A partial ``.part`` file is resumed with a
    ranged request. Raises ``RuntimeError`` on a checksum mismatch.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        if expected_sha1 is None or _sha1_of(dest) == expected_sha1:
            logger.info("using cached %s", dest.name)
            return dest

    part = dest.with_suffix(dest.suffix + ".part")
    resume_from = part.stat().st_size if part.exists() and not force else 0
    headers = {"User-Agent": _USER_AGENT}
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"

    mode = "ab" if resume_from else "wb"
    with httpx.stream(
        "GET", url, headers=headers, follow_redirects=True, timeout=_TIMEOUT
    ) as resp:
        if resume_from and resp.status_code == 200:
            # Server ignored the Range request; restart from scratch.
            resume_from = 0
            mode = "wb"
        resp.raise_for_status()
        total = _content_total(resp, resume_from)
        downloaded = resume_from
        with part.open(mode) as handle:
            for chunk in resp.iter_bytes(_CHUNK):
                handle.write(chunk)
                downloaded += len(chunk)
                if progress is not None:
                    progress(downloaded, total)

    if expected_sha1 is not None:
        actual = _sha1_of(part)
        if actual != expected_sha1:
            part.unlink(missing_ok=True)
            raise RuntimeError(
                f"sha1 mismatch for {dest.name}: expected {expected_sha1}, got {actual}"
            )

    part.replace(dest)
    logger.info("downloaded %s", dest.name)
    return dest


def _content_total(resp: httpx.Response, resume_from: int) -> int | None:
    """Best-effort total byte size including any already-downloaded prefix."""
    length = resp.headers.get("Content-Length")
    if length is None:
        return None
    match = re.search(r"/(\d+)$", resp.headers.get("Content-Range", ""))
    if match:
        return int(match.group(1))
    return int(length) + resume_from


def download_dump(
    source: WikiSource,
    *,
    cache_dir: str | None = None,
    refresh: bool = False,
    verify: bool = True,
    progress: ProgressCallback | None = None,
) -> Path:
    """Download ``source``'s articles dump into the cache and record it.

    Returns the path of the verified dump. ``refresh`` forces a re-download
    even if a cached copy exists; ``verify`` toggles sha1 checking.
    """
    layout = get_layout(source, cache_dir)
    layout.ensure_source_dir()

    expected = fetch_expected_sha1(source) if verify else None
    dump = download_file(
        source.data_url,
        layout.dump_path,
        expected_sha1=expected,
        progress=progress,
        force=refresh,
    )

    manifest = Manifest.load(layout.manifest_path)
    manifest.dbname = source.dbname
    manifest.date = source.date_segment
    manifest.dump_sha1 = expected or _sha1_of(dump)
    manifest.dump_bytes = dump.stat().st_size
    manifest.downloaded_at = datetime.now(UTC).isoformat(timespec="seconds")
    manifest.save(layout.manifest_path)
    return dump
