"""Tests for bestee_compute.etf.issuer — ETF-issuer guessing from ticker name."""

from unittest.mock import MagicMock, patch

import pytest
from massive.rest.models import TickerDetails

from bestee_compute.etf.issuer import _match_issuer, guess_etf_issuer

# Patch the cross-module helper that issuer.py pulls in from stocks/.
_LOOKUP_PATCH = "bestee_compute.etf.issuer.get_ticker_detail"


# ── Pure name matcher ────────────────────────────────────────────────


class TestMatchIssuer:
    @pytest.mark.parametrize(
        "name,expected",
        [
            # Real Massive ticker-details names sampled from the live API.
            ("VanEck Semiconductor ETF", "VanEck"),
            ("State Street SPDR S&P 500 ETF Trust", "SPDR"),
            ("Invesco QQQ Trust, Series 1", "Invesco"),
            ("Vanguard S&P 500 ETF", "Vanguard"),
            ("iShares Core S&P 500 ETF", "iShares"),
            ("ARK Innovation ETF", "ARK"),
            ("State Street Technology Select Sector SPDR ETF", "SPDR"),
            ("Schwab U.S. Broad Market ETF", "Schwab"),
            # Other major issuers (synthesized names).
            ("First Trust Capital Strength ETF", "First Trust"),
            ("JPMorgan Equity Premium Income ETF", "JPMorgan"),
            ("ProShares UltraPro QQQ", "ProShares"),
            ("Direxion Daily Semiconductor Bull 3X Shares", "Direxion"),
            ("WisdomTree US Quality Dividend Growth Fund", "WisdomTree"),
            ("Global X Lithium & Battery Tech ETF", "Global X"),
            ("KraneShares CSI China Internet ETF", "KraneShares"),
            ("PIMCO Active Bond ETF", "PIMCO"),
            ("Fidelity MSCI Information Technology ETF", "Fidelity"),
            ("T. Rowe Price Blue Chip Growth ETF", "T. Rowe Price"),
            ("Goldman Sachs ActiveBeta US Large Cap Equity ETF", "Goldman Sachs"),
            ("Dimensional US Core Equity 2 ETF", "Dimensional"),
        ],
    )
    def test_known_issuer_names(self, name: str, expected: str) -> None:
        assert _match_issuer(name) == expected

    def test_unknown_issuer_returns_none(self) -> None:
        assert _match_issuer("Some Obscure Boutique Asset Manager ETF") is None

    def test_non_etf_company_returns_none(self) -> None:
        assert _match_issuer("Apple Inc") is None
        assert _match_issuer("Microsoft Corp") is None

    def test_empty_name_returns_none(self) -> None:
        assert _match_issuer("") is None

    def test_match_is_case_insensitive(self) -> None:
        assert _match_issuer("VANECK SEMICONDUCTOR ETF") == "VanEck"
        assert _match_issuer("vaneck semiconductor etf") == "VanEck"

    def test_word_boundary_prevents_false_match(self) -> None:
        # "ARK" is short and could substring-match "Arkansas", "Markup",
        # "spark", etc.  Word boundaries should prevent these.
        assert _match_issuer("Arkansas Power & Light") is None
        assert _match_issuer("Spark Therapeutics Inc") is None


# ── guess_etf_issuer (API-mocked) ────────────────────────────────────


def _make_details(name: str) -> MagicMock:
    """Build a TickerDetails mock with the given name."""
    details = MagicMock(spec=TickerDetails)
    details.name = name
    return details


class TestGuessEtfIssuer:
    @patch(_LOOKUP_PATCH)
    def test_returns_matched_issuer(self, mock_lookup: MagicMock) -> None:
        mock_lookup.return_value = _make_details("VanEck Semiconductor ETF")
        assert guess_etf_issuer("SMH") == "VanEck"

    @patch(_LOOKUP_PATCH)
    def test_passes_ticker_to_lookup(self, mock_lookup: MagicMock) -> None:
        mock_lookup.return_value = _make_details("Vanguard S&P 500 ETF")
        guess_etf_issuer("VOO")
        assert mock_lookup.call_args.args == ("VOO",)

    @patch(_LOOKUP_PATCH)
    def test_unknown_name_returns_none(self, mock_lookup: MagicMock) -> None:
        mock_lookup.return_value = _make_details("Apple Inc")
        assert guess_etf_issuer("AAPL") is None

    @patch(_LOOKUP_PATCH)
    def test_empty_name_returns_none(self, mock_lookup: MagicMock) -> None:
        mock_lookup.return_value = _make_details("")
        assert guess_etf_issuer("WAT") is None

    @patch(_LOOKUP_PATCH)
    def test_lookup_failure_returns_none(self, mock_lookup: MagicMock) -> None:
        """``get_ticker_detail`` swallows network/type errors and returns
        *None*; issuer should propagate that."""
        mock_lookup.return_value = None
        assert guess_etf_issuer("SMH") is None

    @patch(_LOOKUP_PATCH)
    def test_api_key_passed_through(self, mock_lookup: MagicMock) -> None:
        mock_lookup.return_value = _make_details("VanEck Semiconductor ETF")
        guess_etf_issuer("SMH", api_key="abc123")
        mock_lookup.assert_called_once_with("SMH", api_key="abc123")
