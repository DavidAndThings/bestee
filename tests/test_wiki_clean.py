"""Wikitext cleaning tests."""

from __future__ import annotations

from bestee_chat.wiki.clean import clean_page, first_paragraph, strip_wikitext

_SAMPLE = (
    "{{Infobox person|name=Ada}}\n"
    "'''Ada Lovelace''' was a [[mathematician|mathematician]] known for the "
    "[[Analytical Engine]].<ref>Some citation.</ref>\n\n"
    "== Career ==\n"
    "She worked with [[Charles Babbage]]."
)


def test_strip_removes_markup() -> None:
    out = strip_wikitext(_SAMPLE)
    assert "{{" not in out
    assert "[[" not in out
    assert "<ref" not in out
    assert "'''" not in out
    assert "==" not in out
    assert "mathematician" in out  # piped link text kept
    assert "Charles Babbage" in out


def test_first_paragraph() -> None:
    body = strip_wikitext(_SAMPLE)
    lead = first_paragraph(body)
    assert lead.startswith("Ada Lovelace was a mathematician")


def test_clean_page_returns_summary_and_body() -> None:
    summary, body = clean_page(_SAMPLE)
    assert summary
    assert "Babbage" in body
