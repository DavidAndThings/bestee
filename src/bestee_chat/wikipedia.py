"""Scrape a person's infobox from a Wikipedia page into a plain dict."""

from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4

import httpx
from bs4 import BeautifulSoup, Tag
from dotenv import load_dotenv

from bestee_chat import config
from bestee_chat.storage import store_mapping

_USER_AGENT = "bestee-chat/0.1 (https://github.com/bestee-chat)"
_TIMEOUT = 30.0


def learn_about_a_person(url: str) -> Path:
    """Scrape a person's infobox and store it in the persons cache.

    The infobox is tagged with metadata (``__type__``, ``__url__``, ``__id__``)
    and handed to :func:`bestee_chat.storage.store_mapping`, which spreads
    records across size-bounded JSON files. The output directory
    (``<cache_root>/persons`` by default, overridable with
    ``$BESTEE_PERSONS_DIR``) is created if missing.

    Returns the path of the file the person was written to.
    """
    load_dotenv()
    persons_dir = config.persons_dir()

    infobox = scrape_infobox(url)
    person = {
        "__type__": "person",
        "__url__": url,
        "__id__": str(uuid4()),
        **infobox,
    }
    return store_mapping(person, persons_dir)


def scrape_infobox(url: str) -> dict[str, str]:
    """Fetch a Wikipedia page and return the person infobox as a plain dict.

    Parameters
    ----------
    url:
        Full URL of the Wikipedia page, e.g.
        ``"https://en.wikipedia.org/wiki/Ada_Lovelace"``.

    Returns
    -------
    dict[str, str]
        Keys are the infobox row labels (e.g. ``"Born"``, ``"Nationality"``).
        A ``"name"`` key is added when the infobox has a caption. Returns an
        empty dict if the page has no infobox.

    Raises
    ------
    httpx.HTTPStatusError
        If the HTTP response has a 4xx or 5xx status code.
    """
    response = httpx.get(
        url,
        headers={"User-Agent": _USER_AGENT},
        follow_redirects=True,
        timeout=_TIMEOUT,
    )
    response.raise_for_status()
    return _parse_infobox(response.text)


def _parse_infobox(html: str) -> dict[str, str]:
    """Parse a Wikipedia infobox from raw HTML.

    Kept separate from the HTTP layer so it can be tested without network
    access.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Wikipedia infoboxes are <table class="infobox …">. The class_ kwarg
    # matches tables that *include* "infobox" among their classes.
    infobox = soup.find("table", class_="infobox")
    if not isinstance(infobox, Tag):
        return {}

    result: dict[str, str] = {}

    # The caption, or the title header (<th class="infobox-above">), usually
    # holds the person's display name.
    title = infobox.find("caption") or infobox.find("th", class_="infobox-above")
    if isinstance(title, Tag):
        name = _clean_text(title)
        if name:
            result["name"] = name

    for row in infobox.find_all("tr"):
        if not isinstance(row, Tag):
            continue

        th = row.find("th")
        td = row.find("td")

        # Rows without both a label and a value are either image rows,
        # full-width section dividers, or other non-data rows.
        if not isinstance(th, Tag) or not isinstance(td, Tag):
            continue

        # A <th colspan="…"> with no sibling <td> is an infobox section
        # header (e.g. "Personal life"). Skip it.
        if th.get("colspan"):
            continue

        key = _clean_text(th)
        value = _clean_text(td)
        if key and value:
            result[key] = value

    return result


def _clean_text(tag: Tag) -> str:
    """Return clean plain text from a BeautifulSoup tag.

    * Removes citation superscripts (``<sup>``).
    * Removes hidden elements (``style="display: none"``).
    * Joins multi-line content (list items, ``<br>``-separated values) with
      ``", "``.
    """
    # Parse a fresh copy so mutations don't affect the source tree.
    node = BeautifulSoup(str(tag), "html.parser")

    for sup in node.find_all("sup"):
        sup.decompose()
    for el in node.select("[style]"):
        if re.search(r"display\s*:\s*none", str(el.get("style") or "")):
            el.decompose()

    # Normalise unicode whitespace: non-breaking -> space, drop zero-width.
    text = node.get_text(separator="\n").replace("\u00a0", " ").replace("\u200b", "")

    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    joined = ", ".join(line for line in lines if line)

    # Tidy punctuation artefacts left by empty inline nodes.
    joined = re.sub(r"\s+,", ",", joined)
    joined = re.sub(r"(?:,\s*){2,}", ", ", joined)
    joined = re.sub(r"\(\s*,\s*", "(", joined)
    return joined.strip(" ,")
