"""Lexical FTS5/BM25 index build + query against the fixture."""

from __future__ import annotations

from pathlib import Path

from bestee_chat.wiki.index import LexicalIndex
from bestee_chat.wiki.parse import iter_pages
from bestee_chat.wiki.sources import WikiSource


def _build(tmp_path: Path, sample_dump: Path) -> Path:
    src = WikiSource(lang="simple")
    db = tmp_path / "index.sqlite"
    with LexicalIndex(db) as index:
        stats = index.build(iter_pages(sample_dump), src)
    assert stats.articles == 3
    assert stats.redirects == 2
    assert stats.aliases_linked == 2  # kings->monarchs, car->automobile
    return db


def test_build_counts(tmp_path: Path, sample_dump: Path) -> None:
    db = _build(tmp_path, sample_dump)
    with LexicalIndex(db) as index:
        assert index.article_count() == 3


def test_redirect_alias_drives_ranking(tmp_path: Path, sample_dump: Path) -> None:
    db = _build(tmp_path, sample_dump)
    with LexicalIndex(db) as index:
        hits = index.search("List of English kings", limit=5)
    assert hits
    # The page is titled "...monarchs"; the "...kings" redirect alias lands it.
    assert hits[0].title == "List of English monarchs"
    assert hits[0].snippet
    assert hits[0].url.endswith("List_of_English_monarchs")


def test_keyword_query(tmp_path: Path, sample_dump: Path) -> None:
    db = _build(tmp_path, sample_dump)
    with LexicalIndex(db) as index:
        hits = index.search("automobile transport", limit=5)
    assert hits[0].title == "Automobile"
