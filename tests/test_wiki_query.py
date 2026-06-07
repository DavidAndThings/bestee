"""Hybrid search (BM25 recall + semantic rerank) with a fake encoder."""

from __future__ import annotations

from pathlib import Path

from bestee_chat.wiki.cache import get_layout
from bestee_chat.wiki.index import LexicalIndex
from bestee_chat.wiki.parse import iter_pages
from bestee_chat.wiki.query import search_wiki
from bestee_chat.wiki.sources import WikiSource
from bestee_chat.wiki.vectors import build_vectors


def _prepare(tmp_path: Path, sample_dump: Path, encoder) -> WikiSource:
    src = WikiSource(lang="simple")
    layout = get_layout(src, str(tmp_path))
    layout.ensure_index_dir("economy")
    with LexicalIndex(layout.sqlite_path("economy")) as index:
        index.build(iter_pages(sample_dump), src)
    build_vectors(
        layout.sqlite_path("economy"),
        layout.vectors_path("economy"),
        encoder,
        granularity="lead",
    )
    return src


def test_hybrid_search_finds_monarchs(tmp_path, sample_dump, fake_encoder) -> None:
    src = _prepare(tmp_path, sample_dump, fake_encoder)
    hits = search_wiki(
        "english kings",
        source=src,
        profile="economy",
        cache_dir=str(tmp_path),
        encoder=fake_encoder,
    )
    assert hits
    assert hits[0].title == "List of English monarchs"
    assert hits[0].rank == 1


def test_hybrid_search_semantic_topic(tmp_path, sample_dump, fake_encoder) -> None:
    src = _prepare(tmp_path, sample_dump, fake_encoder)
    hits = search_wiki(
        "car vehicle",
        source=src,
        profile="economy",
        cache_dir=str(tmp_path),
        encoder=fake_encoder,
    )
    assert hits[0].title == "Automobile"


def test_missing_index_raises(tmp_path, fake_encoder) -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        search_wiki(
            "anything",
            source=WikiSource(lang="simple"),
            profile="lexical",
            cache_dir=str(tmp_path),
            encoder=fake_encoder,
        )
