"""Tests for bestee.sic."""

from unittest.mock import MagicMock, patch

from great_tables import GT

from bestee.sic import _strip_html, get_sic_codes

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


class TestGetSicCodes:
    @patch("bestee.sic.httpx.get")
    def test_returns_gt_object(
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
        assert "PREPACKAGED SOFTWARE" in html

    @patch("bestee.sic.httpx.get")
    def test_includes_office_column(
        self,
        mock_get: MagicMock,
    ) -> None:
        mock_response = MagicMock()
        mock_response.text = _SAMPLE_HTML
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        html = get_sic_codes().as_raw_html()

        assert "Industrial Applications and Services" in html
        assert "Office of Technology" in html

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


class TestStripHtml:
    def test_strips_tags(self) -> None:
        assert _strip_html("<b>bold</b>") == "bold"

    def test_leaves_plain_text(self) -> None:
        assert _strip_html("plain text") == "plain text"

    def test_handles_nested_tags(self) -> None:
        assert _strip_html("<a href='#'><b>link</b></a>") == "link"
