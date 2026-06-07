"""Acquire knowledge of various kinds and cache it for the Brain.

Two sources are supported today:

* :func:`learn_person` -- scrape a person's infobox from a live Wikipedia URL.
* :func:`learn_article` -- read the output of a local dump *search* and cache
  the best-matching article.

Each learned record is tagged with a ``__type__`` and stored (via
:func:`bestee_chat.storage.store_mapping`) under its own cache directory
(``<cache_root>/<kind>``), so different kinds of knowledge stay separate.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from bestee_chat import config
from bestee_chat.storage import store_mapping
from bestee_chat.wiki.query import search_wiki
from bestee_chat.wiki.sources import WikiSource
from bestee_chat.wikipedia import learn_about_a_person


def learn_person(url: str) -> Path:
    """Scrape a person's infobox from ``url`` and cache it as a ``person``."""
    return learn_about_a_person(url)


def learn_article(
    query: str,
    *,
    source: WikiSource | None = None,
    profile: str = "auto",
    cache_dir: str | None = None,
) -> Path:
    """Search a local Wikipedia dump and cache the best-matching article.

    This consumes the output of the dump search (a
    :class:`~bestee_chat.wiki.models.SearchHit`) and stores its title, summary
    and URL as an ``article`` record. A built index is required (see
    ``bestee-chat resources index``); ``cache_dir`` points at the *wiki* cache
    used for that search.

    Raises ``LookupError`` if the search returns nothing.
    """
    hits = search_wiki(
        query,
        source=source or WikiSource(),
        profile=profile,
        limit=1,
        cache_dir=cache_dir,
    )
    if not hits:
        raise LookupError(f"no Wikipedia article found for {query!r}")

    hit = hits[0]
    record = {
        "__type__": "article",
        "__url__": hit.url,
        "__id__": str(uuid4()),
        "title": hit.title,
        "summary": hit.snippet,
    }
    return store_mapping(record, config.knowledge_dir("articles"))
