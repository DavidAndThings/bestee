"""Stream pages out of a MediaWiki XML dump with flat memory use.

The dump is a single huge ``<mediawiki>`` element containing millions of
``<page>`` children, optionally bzip2-compressed. We parse it incrementally
with :func:`xml.etree.ElementTree.iterparse` and clear each element as soon as
it is yielded, so memory stays roughly constant regardless of dump size.

The parser is namespace-agnostic: MediaWiki tags are namespaced
(``{http://...}page``), so we compare on the local tag name only.
"""

from __future__ import annotations

import bz2
from collections.abc import Iterator
from pathlib import Path
from typing import IO
from xml.etree.ElementTree import Element, iterparse

from bestee_chat.wiki.models import RawPage


def _local(tag: str) -> str:
    """Return the local name of a possibly namespaced XML ``tag``."""
    return tag.rsplit("}", 1)[-1]


def _open_dump(path: Path) -> IO[bytes]:
    """Open a dump for binary reading, transparently decompressing ``.bz2``."""
    if path.suffix == ".bz2":
        return bz2.open(path, "rb")
    return path.open("rb")


def _find_child(page: Element, name: str) -> Element | None:
    """Return the first *direct* child of ``page`` with local name ``name``."""
    for child in page:
        if _local(child.tag) == name:
            return child
    return None


def _find_descendant_text(page: Element, name: str) -> str:
    """Return the text of the first descendant with local name ``name``."""
    for elem in page.iter():
        if _local(elem.tag) == name:
            return elem.text or ""
    return ""


def _build_page(page: Element) -> RawPage:
    """Construct a :class:`RawPage` from a parsed ``<page>`` element."""
    title_el = _find_child(page, "title")
    ns_el = _find_child(page, "ns")
    id_el = _find_child(page, "id")
    redirect_el = _find_child(page, "redirect")

    try:
        namespace = int((ns_el.text or "0").strip()) if ns_el is not None else 0
    except ValueError:
        namespace = 0
    try:
        page_id = int((id_el.text or "0").strip()) if id_el is not None else 0
    except ValueError:
        page_id = 0

    redirect_target = redirect_el.get("title") if redirect_el is not None else None
    return RawPage(
        page_id=page_id,
        title=(title_el.text or "").strip() if title_el is not None else "",
        namespace=namespace,
        is_redirect=redirect_el is not None,
        redirect_target=redirect_target,
        text=_find_descendant_text(page, "text"),
    )


def iter_pages(path: str | Path) -> Iterator[RawPage]:
    """Yield every ``<page>`` in the dump at ``path`` as a :class:`RawPage`.

    Works on both ``*.xml`` and ``*.xml.bz2`` files. Memory stays flat: each
    page subtree is cleared right after it is produced.
    """
    path = Path(path)
    with _open_dump(path) as stream:
        context = iterparse(stream, events=("start", "end"))
        _, root = next(context)  # first 'start' is the <mediawiki> root
        for event, elem in context:
            if event != "end" or _local(elem.tag) != "page":
                continue
            yield _build_page(elem)
            elem.clear()
            root.clear()  # drop already-processed pages from the root


def iter_articles(path: str | Path) -> Iterator[RawPage]:
    """Yield only non-redirect pages in the main article namespace (ns 0)."""
    for page in iter_pages(path):
        if page.is_article:
            yield page
