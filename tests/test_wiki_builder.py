"""End-to-end build via the builder, using a bz2 dump and a fake encoder."""

from __future__ import annotations

import bz2
from pathlib import Path

from bestee_chat.wiki.builder import build_index
from bestee_chat.wiki.cache import Manifest, get_layout
from bestee_chat.wiki.query import search_wiki
from bestee_chat.wiki.sources import WikiSource


def _seed_dump(tmp_path: Path, sample_dump: Path) -> WikiSource:
    src = WikiSource(lang="simple")
    layout = get_layout(src, str(tmp_path))
    layout.ensure_source_dir()
    layout.dump_path.write_bytes(bz2.compress(sample_dump.read_bytes()))
    return src


def test_build_index_economy(tmp_path, sample_dump, fake_encoder) -> None:
    src = _seed_dump(tmp_path, sample_dump)
    result = build_index(
        src,
        "economy",
        cache_dir=str(tmp_path),
        encoder=fake_encoder,
        download_if_missing=False,
    )
    assert result.profile.name == "economy"
    assert result.stats.articles == 3
    assert result.vectors_built

    layout = get_layout(src, str(tmp_path))
    manifest = Manifest.load(layout.manifest_path)
    assert "economy" in manifest.indexes

    hits = search_wiki(
        "List of English kings",
        source=src,
        profile="economy",
        cache_dir=str(tmp_path),
        encoder=fake_encoder,
    )
    assert hits[0].title == "List of English monarchs"


def test_build_index_lexical_only(tmp_path, sample_dump) -> None:
    src = _seed_dump(tmp_path, sample_dump)
    result = build_index(
        src, "lexical", cache_dir=str(tmp_path), download_if_missing=False
    )
    assert result.profile.name == "lexical"
    assert not result.vectors_built
