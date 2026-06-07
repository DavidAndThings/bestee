"""Extract a person's infobox from MediaWiki wikitext (for offline learning).

The live-page path in :mod:`bestee_chat.wikipedia` parses a *rendered* HTML
``<table class="infobox">``. A downloaded dump only has raw **wikitext**, where
the same data lives in an ``{{Infobox ...}}`` template, so this module pulls
that template apart into the same kind of ``label -> value`` dict. Values are
cleaned with :func:`bestee_chat.wiki.clean.strip_wikitext` so both paths produce
comparable text.

Like the rest of the cleaner, this is deliberately pragmatic rather than a full
wikitext parser: it understands the handful of templates that actually carry
infobox data (dates, simple lists, marriages) and drops the rest. A future
upgrade could swap in ``mwparserfromhell`` behind :func:`parse_infobox`.
"""

from __future__ import annotations

import re
from pathlib import Path

from bestee_chat.wiki.clean import strip_wikitext
from bestee_chat.wiki.models import RawPage
from bestee_chat.wiki.parse import iter_pages

_INFOBOX_START = re.compile(r"\{\{\s*infobox\b", re.IGNORECASE)
_NUM = re.compile(r"\d+")
_MULTI_COMMA = re.compile(r"(?:,\s*){2,}")

#: Presentational/non-data parameters that should never become learned facts.
_SKIP_KEYS = frozenset(
    {
        "image",
        "image_size",
        "imagesize",
        "image_upright",
        "caption",
        "alt",
        "signature",
        "signature_size",
        "signature_alt",
        "border",
        "frame",
        "module",
        "embed",
        "child",
        "header",
        "width",
    }
)

#: Templates whose positional arguments form a list (rendered comma-separated).
_LIST_TEMPLATES = frozenset(
    {
        "plainlist",
        "unbulleted list",
        "ubl",
        "hlist",
        "flatlist",
        "bulleted list",
        "collapsible list",
    }
)

#: Date templates: the first three numeric positionals are year, month, day.
_DATE_TEMPLATES = frozenset(
    {
        "birth date and age",
        "birth date",
        "birth-date",
        "birth date and given age",
        "death date and age",
        "death date",
        "death-date",
        "start date",
        "end date",
    }
)

#: Wrapper templates we unwrap to their (resolved) content.
_PASSTHROUGH_TEMPLATES = frozenset({"nowrap", "nobr", "nobold", "nb", "small"})

# Aliases folded into the friendly field names the answer engine understands
# (see ``bestee_chat.knowledge._QUESTION_TEMPLATES``).
_BORN_DATE_KEYS = ("born", "birth_date", "date_of_birth", "birthdate")
_BORN_PLACE_KEYS = ("birth_place", "birthplace", "place_of_birth")
_DIED_DATE_KEYS = ("died", "death_date", "date_of_death", "deathdate")
_DIED_PLACE_KEYS = ("death_place", "deathplace", "place_of_death")
_EDUCATION_KEYS = ("education", "alma_mater", "almamater")


def parse_infobox(wikitext: str) -> dict[str, str]:
    """Return the first ``{{Infobox ...}}`` template's fields as a plain dict.

    Keys are the (lowercased) parameter names; values are cleaned plain text.
    Returns an empty dict when ``wikitext`` has no infobox.
    """
    match = _INFOBOX_START.search(wikitext)
    if match is None:
        return {}

    inner, _ = _find_template(wikitext, match.start())
    fields: dict[str, str] = {}
    # The first ``|``-delimited part is the template name (e.g. "Infobox
    # person"); the rest are ``key = value`` parameters.
    for part in _split_top_level(inner, "|")[1:]:
        key, raw_value = _split_key_value(part)
        if key is None:
            continue
        name = key.strip().lower()
        if not name or name in _SKIP_KEYS:
            continue
        value = _clean_value(raw_value)
        if value:
            fields[name] = value
    return fields


def extract_person_infobox(
    dump_path: str | Path, title: str
) -> tuple[str, dict[str, str]] | None:
    """Find ``title`` in the dump and return ``(resolved_title, fields)``.

    Streams the dump at ``dump_path`` to locate the article named ``title``
    (following at most a few redirect hops), parses its infobox and folds the
    common person aliases into the friendly field names the answer engine knows
    (``born``, ``died``, ``education``). A ``name`` field is added from the page
    title when the infobox lacks one.

    Returns ``None`` when no such page exists in the dump. Note this performs a
    linear scan of the (possibly bz2-compressed) dump per redirect hop.
    """
    page = _find_article(dump_path, title)
    if page is None:
        return None
    fields = _as_person_fields(parse_infobox(page.text))
    fields.setdefault("name", page.title)
    return page.title, fields


# ---------------------------------------------------------------------------
# Dump lookup
# ---------------------------------------------------------------------------


def _find_article(
    dump_path: str | Path, title: str, *, max_hops: int = 4
) -> RawPage | None:
    """Return the article named ``title``, following redirects (bounded)."""
    wanted = _normalise_title(title)
    visited: set[str] = set()
    while wanted and wanted not in visited and len(visited) <= max_hops:
        visited.add(wanted)
        page = _scan_for_title(dump_path, wanted)
        if page is None:
            return None
        if page.is_redirect and page.redirect_target:
            wanted = _normalise_title(page.redirect_target)
            continue
        return page
    return None


def _scan_for_title(dump_path: str | Path, wanted: str) -> RawPage | None:
    """Return the first article-namespace page whose title equals ``wanted``."""
    for page in iter_pages(dump_path):
        if page.namespace == 0 and _normalise_title(page.title) == wanted:
            return page
    return None


def _normalise_title(title: str) -> str:
    """Normalise a title for comparison (MediaWiki upper-cases the first char)."""
    cleaned = title.replace("_", " ").strip()
    if not cleaned:
        return ""
    return cleaned[0].upper() + cleaned[1:]


# ---------------------------------------------------------------------------
# Person field normalisation
# ---------------------------------------------------------------------------


def _as_person_fields(raw: dict[str, str]) -> dict[str, str]:
    """Fold common person-infobox aliases into friendly, engine-ready keys."""
    fields = dict(raw)
    born = _combine(fields, _BORN_DATE_KEYS, _BORN_PLACE_KEYS)
    if born:
        fields["born"] = born
    died = _combine(fields, _DIED_DATE_KEYS, _DIED_PLACE_KEYS)
    if died:
        fields["died"] = died
    education = _take_first(fields, _EDUCATION_KEYS)
    if education:
        fields["education"] = education
    return fields


def _combine(
    fields: dict[str, str], date_keys: tuple[str, ...], place_keys: tuple[str, ...]
) -> str:
    """Pop the date/place aliases and return them joined (``date, place``)."""
    date = _take_first(fields, date_keys)
    place = _take_first(fields, place_keys)
    return ", ".join(part for part in (date, place) if part)


def _take_first(fields: dict[str, str], keys: tuple[str, ...]) -> str:
    """Remove every alias in ``keys`` from ``fields``, returning the first value."""
    found = ""
    for key in keys:
        value = fields.pop(key, "")
        if value and not found:
            found = value
    return found


# ---------------------------------------------------------------------------
# Wikitext template handling
# ---------------------------------------------------------------------------


def _clean_value(raw: str) -> str:
    """Resolve known templates in ``raw``, strip wikitext, and tidy whitespace."""
    text = strip_wikitext(_resolve_templates(raw))
    lines = [line.strip() for line in text.splitlines()]
    joined = ", ".join(line for line in lines if line)
    return _MULTI_COMMA.sub(", ", joined).strip(" ,")


def _resolve_templates(text: str) -> str:
    """Replace each top-level ``{{...}}`` in ``text`` with its rendered value."""
    out: list[str] = []
    i = 0
    length = len(text)
    while i < length:
        if text.startswith("{{", i):
            inner, end = _find_template(text, i)
            out.append(_render_template(inner))
            i = end
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _render_template(inner: str) -> str:
    """Render a single template body (text between ``{{`` and ``}}``)."""
    args = _split_top_level(inner, "|")
    name = args[0].strip().lower()
    rest = args[1:]
    if name in _DATE_TEMPLATES:
        return _render_date(rest)
    if name in _LIST_TEMPLATES:
        return _render_list(rest)
    if name in {"marriage", "married"}:
        return _render_marriage(rest)
    if name in _PASSTHROUGH_TEMPLATES:
        positionals = _positionals(rest)
        return _resolve_templates(positionals[-1]) if positionals else ""
    # Unknown template: drop it, matching strip_wikitext's behaviour.
    return ""


def _render_date(args: list[str]) -> str:
    """Render a date template as ``YYYY``, ``YYYY-MM`` or ``YYYY-MM-DD``."""
    numbers: list[str] = []
    for arg in _positionals(args):
        match = _NUM.search(arg)
        if match:
            numbers.append(match.group())
        if len(numbers) == 3:
            break
    if not numbers:
        return ""
    if len(numbers) == 1:
        return numbers[0]
    return "-".join([numbers[0].zfill(4), *(n.zfill(2) for n in numbers[1:])])


def _render_list(args: list[str]) -> str:
    """Render a list template as its comma-separated items.

    Items come either as separate positionals (``{{hlist|a|b}}``) or as one
    positional of ``*``-bulleted lines (``{{plainlist|...}}``), so each
    positional is split on newlines and any leading list markers are stripped.
    """
    items: list[str] = []
    for positional in _positionals(args):
        for line in _resolve_templates(positional).splitlines():
            item = line.lstrip("*#:; \t").strip()
            if item:
                items.append(item)
    return ", ".join(items)


def _render_marriage(args: list[str]) -> str:
    """Render ``{{marriage|Spouse|start|end}}`` as ``Spouse (m. start-end)``."""
    positionals = _positionals(args)
    if not positionals:
        return ""
    name = _resolve_templates(positionals[0])
    years = [year for year in positionals[1:3] if _NUM.search(year)]
    if years:
        return f"{name} (m. {'-'.join(years)})"
    return name


# ---------------------------------------------------------------------------
# Brace/bracket-aware scanning primitives
# ---------------------------------------------------------------------------


def _find_template(text: str, start: int) -> tuple[str, int]:
    """Return ``(inner, end)`` for the ``{{...}}`` starting at ``start``.

    ``inner`` is the text between the braces; ``end`` is the index just past the
    closing ``}}``. Brace depth is tracked so nested templates are handled. An
    unterminated template yields everything to the end of ``text``.
    """
    depth = 0
    i = start
    length = len(text)
    while i < length:
        char = text[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + 2 : i - 1], i + 1
        i += 1
    return text[start + 2 :], length


def _split_top_level(text: str, sep: str) -> list[str]:
    """Split ``text`` on ``sep`` only where outside ``{{}}``/``[[]]`` nesting."""
    parts: list[str] = []
    buf: list[str] = []
    brace = 0
    bracket = 0
    for char in text:
        if char == "{":
            brace += 1
        elif char == "}":
            brace -= 1
        elif char == "[":
            bracket += 1
        elif char == "]":
            bracket -= 1
        if char == sep and brace <= 0 and bracket <= 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(char)
    parts.append("".join(buf))
    return parts


def _split_key_value(part: str) -> tuple[str | None, str]:
    """Split ``part`` on its first top-level ``=`` into ``(key, value)``.

    Returns ``(None, part)`` for a positional argument (no top-level ``=``).
    """
    brace = 0
    bracket = 0
    for i, char in enumerate(part):
        if char == "{":
            brace += 1
        elif char == "}":
            brace -= 1
        elif char == "[":
            bracket += 1
        elif char == "]":
            bracket -= 1
        elif char == "=" and brace <= 0 and bracket <= 0:
            return part[:i], part[i + 1 :]
    return None, part


def _positionals(args: list[str]) -> list[str]:
    """Return the non-empty positional (un-keyed) arguments, stripped."""
    result: list[str] = []
    for arg in args:
        key, value = _split_key_value(arg)
        if key is None:
            stripped = value.strip()
            if stripped:
                result.append(stripped)
    return result
