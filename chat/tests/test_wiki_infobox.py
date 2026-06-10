"""Tests for parsing a person's infobox out of MediaWiki wikitext + a dump."""

from __future__ import annotations

import bz2
import json
from pathlib import Path
from textwrap import dedent

import pytest

from bestee_chat import learn
from bestee_chat.wiki.infobox import extract_person_infobox, parse_infobox
from bestee_chat.wiki.sources import WikiSource

# ---------------------------------------------------------------------------
# Wikitext fixtures
# ---------------------------------------------------------------------------

_PERSON_WIKITEXT = dedent(
    """\
    {{Short description|English mathematician}}
    {{Infobox person
    | name        = Ada Lovelace
    | image       = Ada_Lovelace.jpg
    | caption     = Watercolour portrait
    | birth_name  = Augusta Ada Byron
    | birth_date  = {{birth date|1815|12|10|df=y}}
    | birth_place = [[London]], England
    | death_date  = {{death date and age|1852|11|27|1815|12|10|df=y}}
    | death_place = [[Marylebone]], London
    | nationality = British
    | known_for   = {{plainlist|
    * [[Analytical Engine]]
    * The first published [[computer program]]
    }}
    | spouse      = {{marriage|[[William, Earl of Lovelace|William King]]|1835}}
    | children    = 3
    | alma_mater  = privately educated
    | occupation  = {{hlist|Mathematician|Writer}}
    }}
    '''Augusta Ada King, Countess of Lovelace''' was an English mathematician.
    """
)


def _dump_text() -> str:
    """A tiny two-page dump: a person article plus a redirect to it."""
    return f"""<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/"
  version="0.11" xml:lang="en">
  <page>
    <title>Ada Lovelace</title>
    <ns>0</ns>
    <id>1</id>
    <revision>
      <id>11</id>
      <text>{_PERSON_WIKITEXT}</text>
    </revision>
  </page>
  <page>
    <title>Ada Byron</title>
    <ns>0</ns>
    <id>2</id>
    <redirect title="Ada Lovelace" />
    <revision>
      <id>12</id>
      <text>#REDIRECT [[Ada Lovelace]]</text>
    </revision>
  </page>
</mediawiki>
"""


def _write_dump(path: Path) -> Path:
    """Write the sample dump as plain XML (``iter_pages`` reads ``.xml``)."""
    path.write_text(_dump_text(), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# parse_infobox (pure wikitext, no I/O)
# ---------------------------------------------------------------------------


def test_parses_name_and_simple_field() -> None:
    fields = parse_infobox(_PERSON_WIKITEXT)
    assert fields["name"] == "Ada Lovelace"
    assert fields["nationality"] == "British"


def test_skips_presentational_image_keys() -> None:
    fields = parse_infobox(_PERSON_WIKITEXT)
    assert "image" not in fields
    assert "caption" not in fields


def test_resolves_date_templates() -> None:
    fields = parse_infobox(_PERSON_WIKITEXT)
    assert fields["birth_date"] == "1815-12-10"
    # "death date and age" lists the death date first.
    assert fields["death_date"] == "1852-11-27"


def test_resolves_links_inside_values() -> None:
    fields = parse_infobox(_PERSON_WIKITEXT)
    assert fields["birth_place"] == "London, England"


def test_renders_list_templates_comma_separated() -> None:
    fields = parse_infobox(_PERSON_WIKITEXT)
    assert (
        fields["known_for"] == "Analytical Engine, The first published computer program"
    )
    assert fields["occupation"] == "Mathematician, Writer"


def test_renders_marriage_template_with_piped_link() -> None:
    fields = parse_infobox(_PERSON_WIKITEXT)
    assert fields["spouse"] == "William King (m. 1835)"


def test_drops_unknown_template_only_values() -> None:
    fields = parse_infobox("{{Infobox person|name=X|citizenship={{flag|UK}}}}")
    assert fields == {"name": "X"}


def test_returns_empty_dict_without_infobox() -> None:
    assert parse_infobox("Just some ordinary [[wikitext]] with no infobox.") == {}


# ---------------------------------------------------------------------------
# extract_person_infobox (reads a dump file)
# ---------------------------------------------------------------------------


def test_extract_finds_page_and_normalises_person_fields(tmp_path: Path) -> None:
    dump = _write_dump(tmp_path / "dump.xml")

    found = extract_person_infobox(dump, "Ada Lovelace")
    assert found is not None
    resolved_title, fields = found

    assert resolved_title == "Ada Lovelace"
    # birth_date + birth_place are folded into the engine-friendly "born".
    assert fields["born"] == "1815-12-10, London, England"
    assert fields["died"] == "1852-11-27, Marylebone, London"
    # alma_mater is folded into "education".
    assert fields["education"] == "privately educated"
    # The raw aliases are consumed by the fold.
    assert "birth_date" not in fields
    assert "alma_mater" not in fields


def test_extract_follows_redirects(tmp_path: Path) -> None:
    dump = _write_dump(tmp_path / "dump.xml")

    found = extract_person_infobox(dump, "Ada Byron")

    assert found is not None
    resolved_title, fields = found
    assert resolved_title == "Ada Lovelace"
    assert fields["name"] == "Ada Lovelace"


def test_extract_is_title_normalising(tmp_path: Path) -> None:
    dump = _write_dump(tmp_path / "dump.xml")

    # Underscores and a lowercased first letter still resolve.
    found = extract_person_infobox(dump, "ada_Lovelace")

    assert found is not None
    assert found[0] == "Ada Lovelace"


def test_extract_returns_none_for_missing_page(tmp_path: Path) -> None:
    dump = _write_dump(tmp_path / "dump.xml")
    assert extract_person_infobox(dump, "Nobody In Particular") is None


# ---------------------------------------------------------------------------
# learn_person from a local dump (end to end, no network)
# ---------------------------------------------------------------------------


def test_learn_person_from_dump_stores_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "wiki"
    source = WikiSource()
    layout_dir = cache_dir / source.dbname / source.date_segment
    layout_dir.mkdir(parents=True)
    # The real dump is bz2-compressed XML; write one at the expected path.
    dump_path = layout_dir / source.data_filename
    dump_path.write_bytes(bz2.compress(_dump_text().encode("utf-8")))

    monkeypatch.setenv("BESTEE_PERSONS_DIR", str(tmp_path / "persons"))

    path = learn.learn_person("Ada Lovelace", source=source, cache_dir=str(cache_dir))

    (record,) = json.loads(path.read_text(encoding="utf-8"))
    assert record["__type__"] == "person"
    assert record["name"] == "Ada Lovelace"
    assert record["born"].startswith("1815-12-10")
    assert record["__url__"].endswith("Ada_Lovelace")


def test_learn_person_from_dump_missing_dump_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BESTEE_PERSONS_DIR", str(tmp_path / "persons"))
    with pytest.raises(FileNotFoundError):
        learn.learn_person(
            "Ada Lovelace", source=WikiSource(), cache_dir=str(tmp_path / "empty")
        )
