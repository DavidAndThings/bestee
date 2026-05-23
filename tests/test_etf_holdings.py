"""Tests for bestee.etf.holdings — issuer-aware dispatcher."""

from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from great_tables import GT

from bestee.etf.holdings import (
    IssuerNotSupportedError,
    get_holdings,
    get_holdings_table,
    supported_issuers,
)

_GUESS_PATCH = "bestee.etf.holdings.guess_etf_issuer"
_VANECK_PATCH = "bestee.etf.holdings.vaneck"
_SPDR_PATCH = "bestee.etf.holdings.spdr"
_ISHARES_PATCH = "bestee.etf.holdings.ishares"
_ROUNDHILL_PATCH = "bestee.etf.holdings.roundhill"
_INVESCO_PATCH = "bestee.etf.holdings.invesco"


# ── Registry ─────────────────────────────────────────────────────────


def test_supported_issuers_includes_known_providers() -> None:
    issuers = supported_issuers()
    assert "VanEck" in issuers
    assert "SPDR" in issuers
    assert "iShares" in issuers
    assert "Roundhill" in issuers
    assert "Invesco" in issuers


# ── Successful dispatch ──────────────────────────────────────────────


@patch(_SPDR_PATCH)
@patch(_VANECK_PATCH)
@patch(_GUESS_PATCH)
def test_dispatches_to_vaneck_for_vaneck_etf(
    mock_guess: MagicMock,
    mock_vaneck: MagicMock,
    mock_spdr: MagicMock,
) -> None:
    mock_guess.return_value = "VanEck"
    expected = pl.DataFrame({"Ticker": ["NVDA"]})
    mock_vaneck.get_holdings.return_value = expected

    result = get_holdings("SMH")

    mock_vaneck.get_holdings.assert_called_once_with("SMH")
    mock_spdr.get_holdings.assert_not_called()
    assert result is expected


@patch(_SPDR_PATCH)
@patch(_VANECK_PATCH)
@patch(_GUESS_PATCH)
def test_dispatches_to_spdr_for_spdr_etf(
    mock_guess: MagicMock,
    mock_vaneck: MagicMock,
    mock_spdr: MagicMock,
) -> None:
    mock_guess.return_value = "SPDR"
    expected = pl.DataFrame({"Ticker": ["NVDA"]})
    mock_spdr.get_holdings.return_value = expected

    result = get_holdings("SPY")

    mock_spdr.get_holdings.assert_called_once_with("SPY")
    mock_vaneck.get_holdings.assert_not_called()
    assert result is expected


@patch(_SPDR_PATCH)
@patch(_VANECK_PATCH)
@patch(_GUESS_PATCH)
def test_get_holdings_table_dispatches(
    mock_guess: MagicMock,
    mock_vaneck: MagicMock,
    mock_spdr: MagicMock,
) -> None:
    mock_guess.return_value = "SPDR"
    expected = MagicMock(spec=GT)
    mock_spdr.get_holdings_table.return_value = expected

    result = get_holdings_table("SPY")

    mock_spdr.get_holdings_table.assert_called_once_with("SPY")
    mock_vaneck.get_holdings_table.assert_not_called()
    assert result is expected


# ── Error paths ──────────────────────────────────────────────────────


@patch(_GUESS_PATCH)
def test_unidentifiable_issuer_raises(
    mock_guess: MagicMock,
) -> None:
    """When guess_etf_issuer returns None, dispatcher errors out clearly."""
    mock_guess.return_value = None
    with pytest.raises(IssuerNotSupportedError, match="identify the issuer"):
        get_holdings("AAPL")


@patch(_ISHARES_PATCH)
@patch(_GUESS_PATCH)
def test_dispatches_to_ishares_for_ishares_etf(
    mock_guess: MagicMock,
    mock_ishares: MagicMock,
) -> None:
    mock_guess.return_value = "iShares"
    expected = pl.DataFrame({"Ticker": ["NVDA"]})
    mock_ishares.get_holdings.return_value = expected

    result = get_holdings("IVV")

    mock_ishares.get_holdings.assert_called_once_with("IVV")
    assert result is expected


@patch(_ROUNDHILL_PATCH)
@patch(_GUESS_PATCH)
def test_dispatches_to_roundhill_for_roundhill_etf(
    mock_guess: MagicMock,
    mock_roundhill: MagicMock,
) -> None:
    mock_guess.return_value = "Roundhill"
    expected = pl.DataFrame({"StockTicker": ["NVDA"]})
    mock_roundhill.get_holdings.return_value = expected

    result = get_holdings("CHAT")

    mock_roundhill.get_holdings.assert_called_once_with("CHAT")
    assert result is expected


@patch(_INVESCO_PATCH)
@patch(_GUESS_PATCH)
def test_dispatches_to_invesco_for_invesco_etf(
    mock_guess: MagicMock,
    mock_invesco: MagicMock,
) -> None:
    mock_guess.return_value = "Invesco"
    expected = pl.DataFrame({"Ticker": ["NVDA"]})
    mock_invesco.get_holdings.return_value = expected

    result = get_holdings("QQQ")

    mock_invesco.get_holdings.assert_called_once_with("QQQ")
    assert result is expected


@patch(_GUESS_PATCH)
def test_issuer_without_provider_raises(
    mock_guess: MagicMock,
) -> None:
    """Known issuer but no scraper registered → clear error mentioning the
    issuer and the list of supported ones."""
    mock_guess.return_value = "Vanguard"  # known by guess, no provider yet
    with pytest.raises(IssuerNotSupportedError, match="Vanguard"):
        get_holdings("VOO")


@patch(_GUESS_PATCH)
def test_get_holdings_table_uses_same_resolution(
    mock_guess: MagicMock,
) -> None:
    """The GT dispatcher should raise the same error for unknown issuers."""
    mock_guess.return_value = None
    with pytest.raises(IssuerNotSupportedError):
        get_holdings_table("AAPL")
