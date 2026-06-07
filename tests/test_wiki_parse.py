"""Streaming dump parser tests against the committed fixture."""

from __future__ import annotations

from pathlib import Path

from bestee_chat.wiki.parse import iter_articles, iter_pages


def test_iter_pages_reads_all(sample_dump: Path) -> None:
    pages = list(iter_pages(sample_dump))
    assert len(pages) == 6
    assert {p.title for p in pages} >= {"Automobile", "Category:English monarchs"}


def test_iter_articles_filters_redirects_and_namespaces(sample_dump: Path) -> None:
    articles = list(iter_articles(sample_dump))
    titles = {a.title for a in articles}
    assert titles == {"List of English monarchs", "Kingdom of England", "Automobile"}
    assert all(a.namespace == 0 and not a.is_redirect for a in articles)


def test_redirect_metadata(sample_dump: Path) -> None:
    by_title = {p.title: p for p in iter_pages(sample_dump)}
    redirect = by_title["List of English kings"]
    assert redirect.is_redirect
    assert redirect.redirect_target == "List of English monarchs"
    assert by_title["Category:English monarchs"].namespace == 14
