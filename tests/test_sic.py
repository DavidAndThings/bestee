"""Tests for bestee.sic."""

from unittest.mock import MagicMock, patch

from great_tables import GT

from bestee.sic import _load_bundled_cache, _strip_html, get_sic_codes

_SAMPLE_HTML = """
<html><body>
<table>
<tr><th>SIC Code</th><th>Office</th><th>Industry Title</th></tr>
<tr>
  <td>100</td>
  <td>Industrial Applications and Services</td>
  <td>AGRICULTURAL PRODUCTION-CROPS</td>
</tr>
<tr>
  <td>3571</td>
  <td>Office of Technology</td>
  <td>ELECTRONIC COMPUTERS</td>
</tr>
<tr>
  <td>7372</td>
  <td>Office of Technology</td>
  <td>PREPACKAGED SOFTWARE</td>
</tr>
</table>
</body></html>
"""

_CACHED_ROWS = [
    {
        "sic_code": "100",
        "industry_title": "AGRICULTURAL PRODUCTION-CROPS",
        "office": "Industrial Applications and Services",
    },
    {
        "sic_code": "3571",
        "industry_title": "ELECTRONIC COMPUTERS",
        "office": "Office of Technology",
    },
]


class TestGetSicCodes:
    @patch("bestee.sic.httpx.get")
    def test_returns_gt_from_scrape(
        self,
        mock_get: MagicMock,
    ) -> None:
        mock_response = MagicMock()
        mock_response.text = _SAMPLE_HTML
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = get_sic_codes()

        assert isinstance(result, GT)

    @patch("bestee.sic.httpx.get")
    def test_parses_all_rows(
        self,
        mock_get: MagicMock,
    ) -> None:
        mock_response = MagicMock()
        mock_response.text = _SAMPLE_HTML
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        html = get_sic_codes().as_raw_html()

        assert "100" in html
        assert "3571" in html
        assert "7372" in html
        assert "AGRICULTURAL PRODUCTION-CROPS" in html
        assert "ELECTRONIC COMPUTERS" in html

    @patch("bestee.sic.httpx.get")
    def test_sends_user_agent(
        self,
        mock_get: MagicMock,
    ) -> None:
        mock_response = MagicMock()
        mock_response.text = _SAMPLE_HTML
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        get_sic_codes()

        _, kwargs = mock_get.call_args
        assert "User-Agent" in kwargs["headers"]
        assert "bestee" in kwargs["headers"]["User-Agent"]

    @patch("bestee.sic._load_cache", return_value=_CACHED_ROWS)
    @patch("bestee.sic._scrape_sic_codes", side_effect=Exception("network down"))
    def test_falls_back_to_cache_on_scrape_failure(
        self,
        _mock_scrape: MagicMock,
        _mock_cache: MagicMock,
    ) -> None:
        """Should return cached data when scraping fails."""
        result = get_sic_codes()

        assert isinstance(result, GT)
        html = result.as_raw_html()
        assert "ELECTRONIC COMPUTERS" in html

    @patch("bestee.sic._load_cache", return_value=None)
    @patch("bestee.sic._scrape_sic_codes", side_effect=Exception("network down"))
    def test_raises_when_no_cache_and_scrape_fails(
        self,
        _mock_scrape: MagicMock,
        _mock_cache: MagicMock,
    ) -> None:
        """Should raise RuntimeError when both scrape and cache fail."""
        import pytest

        with pytest.raises(RuntimeError, match="Could not obtain SIC codes"):
            get_sic_codes()

    @patch("bestee.sic._save_cache")
    @patch("bestee.sic.httpx.get")
    def test_saves_cache_after_successful_scrape(
        self,
        mock_get: MagicMock,
        mock_save: MagicMock,
    ) -> None:
        mock_response = MagicMock()
        mock_response.text = _SAMPLE_HTML
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        get_sic_codes()

        mock_save.assert_called_once()
        saved_rows = mock_save.call_args[0][0]
        assert len(saved_rows) == 3
        assert saved_rows[0]["sic_code"] == "100"


class TestLoadCache:
    def test_loads_bundled_cache(self) -> None:
        """The bundled sic_codes.json should be loadable via importlib.resources."""
        rows = _load_bundled_cache()

        assert rows is not None
        assert len(rows) > 400
        assert rows[0]["sic_code"] == "100"


class TestStripHtml:
    def test_strips_tags(self) -> None:
        assert _strip_html("<b>bold</b>") == "bold"

    def test_decodes_entities(self) -> None:
        assert _strip_html("OIL &amp; GAS") == "OIL & GAS"

    def test_leaves_plain_text(self) -> None:
        assert _strip_html("plain text") == "plain text"

    def test_handles_nested_tags(self) -> None:
        assert _strip_html("<a href='#'><b>link</b></a>") == "link"
