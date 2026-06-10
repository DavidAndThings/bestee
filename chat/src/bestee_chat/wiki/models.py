"""Core data types shared across the ``wiki`` subpackage.

These are deliberately small, immutable records with no behaviour, so they can
be passed freely between the download, parse, index and query layers without
creating coupling between them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RawPage:
    """A single ``<page>`` element read from a MediaWiki XML dump.

    Attributes
    ----------
    page_id:
        The numeric MediaWiki page id.
    title:
        The page title (e.g. ``"List of English monarchs"``).
    namespace:
        The MediaWiki namespace number. Articles live in namespace ``0``;
        other namespaces (Talk, Template, Category, ...) are non-article.
    is_redirect:
        ``True`` when the page is a redirect to another title.
    redirect_target:
        The destination title when ``is_redirect`` is ``True``, else ``None``.
    text:
        The raw wikitext of the page's latest revision (may be empty).
    """

    page_id: int
    title: str
    namespace: int
    is_redirect: bool
    redirect_target: str | None
    text: str

    @property
    def is_article(self) -> bool:
        """True for non-redirect pages in the main article namespace."""
        return self.namespace == 0 and not self.is_redirect


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One ranked result returned by a wiki search.

    ``score`` is a relevance value where larger is better (a normalised BM25
    weight, or a semantic cosine, or a blend of both). ``rank`` is the 1-based
    position within a single result set.
    """

    rank: int
    score: float
    title: str
    snippet: str
    url: str
    page_id: int
