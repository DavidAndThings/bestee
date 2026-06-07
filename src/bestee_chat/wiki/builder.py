"""End-to-end index construction: download, parse, index, (optionally) embed.

This module wires the lower-level pieces together. It is the single place that
knows the full build sequence, keeping :mod:`download`, :mod:`parse`,
:mod:`index` and :mod:`vectors` independent of one another.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from bestee_chat.embeddings import Encoder
from bestee_chat.wiki._util import ProgressCallback, optional_encoder
from bestee_chat.wiki.cache import Manifest, get_layout
from bestee_chat.wiki.download import download_dump
from bestee_chat.wiki.hardware import probe_hardware
from bestee_chat.wiki.index import BuildStats, LexicalIndex
from bestee_chat.wiki.parse import iter_pages
from bestee_chat.wiki.profiles import Profile, resolve_profile
from bestee_chat.wiki.sources import WikiSource
from bestee_chat.wiki.vectors import build_vectors

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BuildResult:
    """Outcome of :func:`build_index`."""

    profile: Profile
    stats: BuildStats
    vectors_built: bool


def build_index(
    source: WikiSource,
    profile: str = "auto",
    *,
    cache_dir: str | None = None,
    encoder: Encoder | None = None,
    download_if_missing: bool = True,
    lexical_progress: ProgressCallback | None = None,
    vector_progress: ProgressCallback | None = None,
) -> BuildResult:
    """Build (or rebuild) the index for ``source`` at the given ``profile``.

    ``profile`` may be a named preset or ``"auto"`` (the default), which probes
    the machine and picks the richest feasible preset. The downloaded dump is
    reused across profiles; only the index layers are (re)built.
    """
    layout = get_layout(source, cache_dir)
    resolved = resolve_profile(profile, probe_hardware(cache_dir))
    logger.info("building index for %s at profile %s", source.slug, resolved.name)

    dump = layout.dump_path
    if not dump.exists():
        if not download_if_missing:
            raise FileNotFoundError(
                f"no dump at {dump}; run a download first or pass "
                "download_if_missing=True"
            )
        dump = download_dump(source, cache_dir=cache_dir)

    layout.ensure_index_dir(resolved.name)
    with LexicalIndex(layout.sqlite_path(resolved.name)) as lexical:
        stats = lexical.build(iter_pages(dump), source)

    vectors_built = False
    if resolved.semantic:
        enc = encoder or optional_encoder()
        if enc is None:
            raise RuntimeError(
                "semantic profile requires torch, which is not available"
            )
        build_vectors(
            layout.sqlite_path(resolved.name),
            layout.vectors_path(resolved.name),
            enc,
            granularity=resolved.granularity,
            progress=vector_progress,
        )
        vectors_built = True

    _record_index(layout.manifest_path, source, resolved, stats, vectors_built)
    return BuildResult(profile=resolved, stats=stats, vectors_built=vectors_built)


def _record_index(
    manifest_path: Path,
    source: WikiSource,
    profile: Profile,
    stats: BuildStats,
    vectors_built: bool,
) -> None:
    """Update the on-disk manifest with the freshly-built index's metadata."""
    manifest = Manifest.load(manifest_path)
    manifest.dbname = source.dbname
    manifest.date = source.date_segment
    manifest.indexes[profile.name] = {
        "profile": profile.name,
        "articles": stats.articles,
        "redirects": stats.redirects,
        "aliases_linked": stats.aliases_linked,
        "semantic": profile.semantic,
        "vectors": vectors_built,
        "model": profile.model if profile.semantic else "",
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    manifest.save(manifest_path)
