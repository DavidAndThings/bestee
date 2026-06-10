import json
import os
from pathlib import Path

import pytest

from bestee_chat.wikipedia import (
    _parse_infobox,
    learn_about_a_person,
    scrape_infobox,
)

# ---------------------------------------------------------------------------
# Minimal Wikipedia-like infobox HTML fixture
# ---------------------------------------------------------------------------

_FIXTURE_HTML = """
<!DOCTYPE html>
<html>
<body>
<table class="infobox biography vcard">
  <caption>Jane Doe</caption>
  <tbody>
    <tr>
      <td colspan="2" class="infobox-image">
        <img src="photo.jpg" alt="Jane Doe">
      </td>
    </tr>
    <tr>
      <th scope="row">Born</th>
      <td>1 January 1990<span style="display: none"> (age 35)</span>
        <br>London, England</td>
    </tr>
    <tr>
      <th scope="row">Nationality</th>
      <td>British<sup class="reference"><a href="#">[1]</a></sup></td>
    </tr>
    <tr>
      <th colspan="2" class="infobox-header">Personal life</th>
    </tr>
    <tr>
      <th scope="row">Spouse</th>
      <td>John Doe</td>
    </tr>
    <tr>
      <th scope="row">Children</th>
      <td>
        <ul>
          <li>Alice Doe</li>
          <li>Bob Doe</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>
</body>
</html>
"""

_FIXTURE_NO_INFOBOX_HTML = "<html><body><p>No infobox here.</p></body></html>"


# ---------------------------------------------------------------------------
# Unit tests (no network)
# ---------------------------------------------------------------------------


def test_extracts_name_from_caption() -> None:
    result = _parse_infobox(_FIXTURE_HTML)
    assert result["name"] == "Jane Doe"


def test_extracts_simple_row_value() -> None:
    result = _parse_infobox(_FIXTURE_HTML)
    assert result["Spouse"] == "John Doe"


def test_removes_citation_superscripts() -> None:
    result = _parse_infobox(_FIXTURE_HTML)
    assert "[1]" not in result["Nationality"]
    assert result["Nationality"] == "British"


def test_removes_hidden_elements() -> None:
    result = _parse_infobox(_FIXTURE_HTML)
    assert "age" not in result["Born"]


def test_joins_multiline_value_with_comma() -> None:
    result = _parse_infobox(_FIXTURE_HTML)
    assert "London" in result["Born"]
    assert "1 January 1990" in result["Born"]


def test_joins_list_items_with_comma() -> None:
    result = _parse_infobox(_FIXTURE_HTML)
    assert "Alice Doe" in result["Children"]
    assert "Bob Doe" in result["Children"]


def test_skips_image_only_rows() -> None:
    result = _parse_infobox(_FIXTURE_HTML)
    # Image row has no <th>, so it should not appear as a key
    assert not any("img" in k.lower() or "photo" in k.lower() for k in result)


def test_skips_section_header_rows() -> None:
    result = _parse_infobox(_FIXTURE_HTML)
    assert "Personal life" not in result


def test_returns_empty_dict_when_no_infobox() -> None:
    assert _parse_infobox(_FIXTURE_NO_INFOBOX_HTML) == {}


# ---------------------------------------------------------------------------
# Integration test (real network, opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("BESTEE_RUN_INTEGRATION"),
    reason="set BESTEE_RUN_INTEGRATION=1 to run network-backed tests",
)
def test_scrape_real_wikipedia_page() -> None:
    result = scrape_infobox("https://en.wikipedia.org/wiki/Ada_Lovelace")

    assert isinstance(result, dict)
    assert len(result) > 0
    assert "name" in result
    assert "Born" in result
    # Confirm references and hidden spans are stripped from the Born value
    assert "[" not in result["Born"]


@pytest.mark.skipif(
    not os.environ.get("BESTEE_RUN_INTEGRATION"),
    reason="set BESTEE_RUN_INTEGRATION=1 to run network-backed tests",
)
def test_learn_about_a_person_writes_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    persons_dir = tmp_path / "persons"
    monkeypatch.setenv("BESTEE_PERSONS_DIR", str(persons_dir))

    target = learn_about_a_person("https://en.wikipedia.org/wiki/Ada_Lovelace")

    assert target.parent == persons_dir
    (person,) = json.loads(target.read_text(encoding="utf-8"))
    assert person["__type__"] == "person"
    assert person["__url__"].endswith("Ada_Lovelace")
    assert "__id__" in person
    assert "name" in person
