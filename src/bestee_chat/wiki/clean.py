"""Convert MediaWiki wikitext into searchable plain text.

This is a pragmatic, dependency-free cleaner: it strips the markup that hurts
full-text search (templates, refs, tables, links, headings, formatting) while
keeping readable prose. It is intentionally lossy -- the goal is good search
text and snippets, not a faithful render. A future upgrade could swap in
``mwparserfromhell`` behind the same interface for higher fidelity.
"""

from __future__ import annotations

import re

_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_REF_SELF = re.compile(r"<ref[^>]*?/>", re.IGNORECASE)
_REF_PAIR = re.compile(r"<ref[^>]*?>.*?</ref>", re.IGNORECASE | re.DOTALL)
_TABLE = re.compile(r"\{\|.*?\|\}", re.DOTALL)
_INNER_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}", re.DOTALL)
_HTML_TAG = re.compile(r"<[^>]+>", re.DOTALL)
_MEDIA_LINK = re.compile(
    r"\[\[\s*(?:File|Image|Category)\s*:[^\[\]]*\]\]", re.IGNORECASE
)
_INNER_LINK = re.compile(r"\[\[(?:[^\[\]|]*\|)?([^\[\]|]*)\]\]")
_EXTERNAL_LABELLED = re.compile(r"\[(?:https?|ftp)://[^\s\]]+\s+([^\]]*)\]")
_EXTERNAL_BARE = re.compile(r"\[(?:https?|ftp)://[^\s\]]+\]")
_BOLD_ITALIC = re.compile(r"'{2,5}")
_HEADING_MARKS = re.compile(r"=+")
_LIST_MARKS = re.compile(r"^[\*#:;]+\s*", re.MULTILINE)
_MULTISPACE = re.compile(r"[ \t]+")
_BLANKLINES = re.compile(r"\n{3,}")


def _drop_nested(pattern: re.Pattern[str], text: str) -> str:
    """Repeatedly apply ``pattern`` until it stops matching (handles nesting)."""
    previous = None
    while previous != text:
        previous = text
        text = pattern.sub("", text)
    return text


def strip_wikitext(text: str) -> str:
    """Return readable plain text for ``text`` (raw wikitext)."""
    text = _COMMENT.sub("", text)
    text = _REF_SELF.sub("", text)
    text = _REF_PAIR.sub("", text)
    text = _drop_nested(_TABLE, text)
    text = _drop_nested(_INNER_TEMPLATE, text)
    text = _drop_nested(_MEDIA_LINK, text)

    # Resolve [[a|b]] -> b, then [[a]] -> a (repeat for nested piping).
    previous = None
    while previous != text:
        previous = text
        text = _INNER_LINK.sub(lambda m: m.group(1), text)

    text = _EXTERNAL_LABELLED.sub(lambda m: m.group(1), text)
    text = _EXTERNAL_BARE.sub("", text)
    text = _HTML_TAG.sub("", text)
    text = _BOLD_ITALIC.sub("", text)
    text = _HEADING_MARKS.sub("", text)
    text = _LIST_MARKS.sub("", text)

    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = _MULTISPACE.sub(" ", text)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _BLANKLINES.sub("\n\n", text)
    return text.strip()


def first_paragraph(plain_text: str, max_chars: int = 320) -> str:
    """Return the lead paragraph of ``plain_text``, truncated to ``max_chars``."""
    for block in plain_text.split("\n\n"):
        block = " ".join(block.split())
        if block:
            if len(block) <= max_chars:
                return block
            return block[:max_chars].rsplit(" ", 1)[0].rstrip() + " ..."
    return ""


def clean_page(text: str) -> tuple[str, str]:
    """Return ``(summary, body)`` plain text for a page's raw wikitext.

    ``body`` is the full stripped text used for full-text search; ``summary``
    is the lead paragraph used for result snippets and lead-only embedding.
    """
    body = strip_wikitext(text)
    return first_paragraph(body), body
