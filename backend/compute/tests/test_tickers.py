"""Tests for bestee_compute.stocks.tickers -- SIC index + related companies.

The Massive client, ticker listing, and per-ticker detail fetch are mocked, so
these run offline and assert the cache filtering, index building, and the
related-company SIC comparison.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from massive.rest.models import TickerDetails

from bestee_compute.stocks import tickers as tk
from bestee_compute.stocks.tickers import (
    _sic_key,
    build_ticker_sic_index,
    get_related_tickers,
    get_tickers_by_sic_code,
    get_tickers_by_sic_industry_title,
    related_tickers_sharing_sic,
)

_CLIENT_PATCH = "bestee_compute.stocks.tickers.get_client"
_DETAIL_PATCH = "bestee_compute.stocks.tickers._fetch_one_ticker_detail"
_FETCH_PATCH = "bestee_compute.stocks.tickers._fetch_tickers"


def _detail(symbol: str, sic_code: object) -> TickerDetails:
    detail = MagicMock(spec=TickerDetails)
    detail.ticker = symbol
    detail.sic_code = sic_code
    return detail


# ── get_tickers_by_sic_code (cache filter) ──────────────────────────


def test_filters_index_by_sic_code() -> None:
    index = {"AAA": "7372", "BBB": "3826", "CCC": "7372"}
    assert get_tickers_by_sic_code("7372", index=index) == ["AAA", "CCC"]


def test_accepts_int_and_normalizes_zero_padding() -> None:
    index = {"FARM": "100"}  # API may store "0100"; both normalize to "100"
    assert get_tickers_by_sic_code(100, index=index) == ["FARM"]
    assert _sic_key("0100") == _sic_key(100) == "100"


# ── build_ticker_sic_index ──────────────────────────────────────────


@patch(_CLIENT_PATCH)
@patch(_DETAIL_PATCH)
@patch(_FETCH_PATCH)
def test_build_index_omits_tickers_without_sic(
    mock_fetch: MagicMock, mock_detail: MagicMock, mock_get_client: MagicMock
) -> None:
    mock_fetch.return_value = [
        MagicMock(ticker="AAA"),
        MagicMock(ticker="BBB"),
        MagicMock(ticker="ETF"),  # funds carry no SIC code
    ]
    sic = {"AAA": 7372, "BBB": 3826, "ETF": None}
    mock_detail.side_effect = lambda _client, symbol: _detail(symbol, sic[symbol])
    assert build_ticker_sic_index() == {"AAA": "7372", "BBB": "3826"}


# ── related companies ───────────────────────────────────────────────


@patch(_CLIENT_PATCH)
def test_get_related_tickers(mock_get_client: MagicMock) -> None:
    client = mock_get_client.return_value
    client.get_related_companies.return_value = [
        MagicMock(ticker="MSFT"),
        MagicMock(ticker="GOOGL"),
    ]
    assert get_related_tickers("AAPL") == ["MSFT", "GOOGL"]


@patch(_CLIENT_PATCH)
def test_related_tickers_sharing_sic_from_index(mock_get_client: MagicMock) -> None:
    client = mock_get_client.return_value
    client.get_related_companies.return_value = [
        MagicMock(ticker="ADBE"),
        MagicMock(ticker="ORCL"),
        MagicMock(ticker="JPM"),
    ]
    index = {"MSFT": "7372", "ADBE": "7372", "ORCL": "7372", "JPM": "6021"}
    assert related_tickers_sharing_sic("MSFT", index=index) == ["ADBE", "ORCL"]


@patch(_DETAIL_PATCH)
@patch(_CLIENT_PATCH)
def test_related_sharing_sic_falls_back_to_live_details(
    mock_get_client: MagicMock, mock_detail: MagicMock
) -> None:
    client = mock_get_client.return_value
    client.get_related_companies.return_value = [MagicMock(ticker="NEW")]
    # "NEW" is absent from the index, so its SIC is fetched live.
    mock_detail.side_effect = lambda _client, symbol: _detail(symbol, 7372)
    assert related_tickers_sharing_sic("MSFT", index={"MSFT": "7372"}) == ["NEW"]


@patch(_CLIENT_PATCH)
def test_related_sharing_sic_empty_when_seed_sic_unknown(
    mock_get_client: MagicMock,
) -> None:
    client = mock_get_client.return_value
    client.get_related_companies.return_value = [MagicMock(ticker="MSFT")]
    # Seed not in the index and (mock) details return nothing useful.
    with patch(_DETAIL_PATCH, return_value=None):
        assert related_tickers_sharing_sic("UNKNOWN", index={}) == []


# ── cache updates from runtime discovery ──────────────────────────────


def test_load_index_overlays_user_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tk, "_ticker_sic_index", None)  # force a fresh load
    user_cache = tmp_path / "ticker_sic_codes.json"
    user_cache.write_text(json.dumps({"NEWCO": "9999", "AAPL": "1111"}))
    monkeypatch.setattr(tk, "_user_ticker_sic_cache", lambda: user_cache)
    index = tk._load_ticker_sic_index()
    assert index["NEWCO"] == "9999"  # new entry from the user cache
    assert index["AAPL"] == "1111"  # user cache overrides the bundled snapshot
    assert index["MSFT"] == "7372"  # bundled snapshot still present


@patch(_DETAIL_PATCH)
@patch(_CLIENT_PATCH)
def test_discovered_related_tickers_are_persisted(
    mock_get_client: MagicMock,
    mock_detail: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = mock_get_client.return_value
    client.get_related_companies.return_value = [MagicMock(ticker="NEW")]
    mock_detail.side_effect = lambda _client, symbol: _detail(symbol, 7372)
    # Seed cached as 7372; NEW is absent so its SIC is fetched live.
    monkeypatch.setattr(tk, "_ticker_sic_index", {"MSFT": "7372"})
    user_cache = tmp_path / "ticker_sic_codes.json"
    monkeypatch.setattr(tk, "_user_ticker_sic_cache", lambda: user_cache)

    result = related_tickers_sharing_sic("MSFT")  # no index -> default cache

    assert result == ["NEW"]
    assert json.loads(user_cache.read_text()) == {"NEW": "7372"}  # persisted
    in_memory = tk._ticker_sic_index
    assert in_memory is not None and in_memory["NEW"] == "7372"  # index updated


@patch(_DETAIL_PATCH)
@patch(_CLIENT_PATCH)
def test_injected_index_does_not_persist(
    mock_get_client: MagicMock,
    mock_detail: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = mock_get_client.return_value
    client.get_related_companies.return_value = [MagicMock(ticker="NEW")]
    mock_detail.side_effect = lambda _client, symbol: _detail(symbol, 7372)
    user_cache = tmp_path / "ticker_sic_codes.json"
    monkeypatch.setattr(tk, "_user_ticker_sic_cache", lambda: user_cache)
    # A caller-supplied index is the caller's responsibility -- no cache write.
    assert related_tickers_sharing_sic("MSFT", index={"MSFT": "7372"}) == ["NEW"]
    assert not user_cache.exists()


# ── get_tickers_by_sic_industry_title ────────────────────────────────

_SIC_TITLES = pl.DataFrame(
    {
        "SIC Code": ["5045", "5734", "7372", "6021"],
        "Industry Title": [
            "WHOLESALE-COMPUTERS & PERIPHERAL EQUIPMENT & SOFTWARE",
            "RETAIL-COMPUTER & COMPUTER SOFTWARE STORES",
            "SERVICES-PREPACKAGED SOFTWARE",
            "NATIONAL COMMERCIAL BANKS",
        ],
    }
)
_SIC_DF_PATCH = "bestee_compute.stocks.tickers.get_sic_codes_df"


@patch(_SIC_DF_PATCH, return_value=_SIC_TITLES)
def test_industry_title_aggregates_all_matching_codes(_mock_df: MagicMock) -> None:
    index = {"AAA": "7372", "BBB": "5045", "CCC": "6021", "DDD": "5734"}
    # Case-insensitive: "software" matches the three software titles (not banks),
    # and tickers from all three codes are aggregated.
    result = get_tickers_by_sic_industry_title("software", index=index)
    assert result == ["AAA", "BBB", "DDD"]


@patch(_SIC_DF_PATCH, return_value=_SIC_TITLES)
def test_industry_title_no_match_returns_empty(_mock_df: MagicMock) -> None:
    # No title contains this -- must return [] rather than raising IndexError.
    assert get_tickers_by_sic_industry_title("nonesuch", index={"AAA": "7372"}) == []


@patch(_SIC_DF_PATCH, return_value=_SIC_TITLES)
def test_industry_title_treats_name_as_literal(_mock_df: MagicMock) -> None:
    # "&" is a regex metacharacter; as a literal it matches the wholesale title.
    result = get_tickers_by_sic_industry_title(
        "equipment & software", index={"BBB": "5045"}
    )
    assert result == ["BBB"]


# ── resolve_terms_to_tickers / resolve_term_to_ticker ──────────────────

_CATALOG: tuple[frozenset[str], dict[str, tuple[str, ...]]] = (
    frozenset({"AAPL", "GOOG", "GOOGL", "MSFT"}),
    {
        "apple inc.": ("AAPL",),
        "alphabet inc.": ("GOOG", "GOOGL"),
        "microsoft corp": ("MSFT",),
    },
)


def test_resolve_keeps_known_ticker_symbol() -> None:
    # A raw symbol (any case) is kept, upper-cased.
    assert tk.resolve_terms_to_tickers(["aapl"], catalog=_CATALOG) == ["AAPL"]


def test_resolve_company_name_maps_to_share_classes() -> None:
    # A name shared by two listings resolves to both, case-insensitively.
    assert tk.resolve_terms_to_tickers(["Alphabet Inc."], catalog=_CATALOG) == [
        "GOOG",
        "GOOGL",
    ]


@patch(_SIC_DF_PATCH, return_value=_SIC_TITLES)
def test_resolve_industry_title_expands_to_codes(_mock_df: MagicMock) -> None:
    index = {"AAA": "7372", "BBB": "5045", "CCC": "6021", "DDD": "5734"}
    # No symbol/name match, so the term falls through to SIC expansion.
    result = tk.resolve_terms_to_tickers(
        ["SERVICES-PREPACKAGED SOFTWARE"], catalog=_CATALOG, index=index
    )
    assert result == ["AAA"]


@patch(_SIC_DF_PATCH, return_value=_SIC_TITLES)
def test_resolve_mixed_terms_dedup_preserves_order(_mock_df: MagicMock) -> None:
    index = {"AAPL": "3571", "AAA": "7372"}
    result = tk.resolve_terms_to_tickers(
        ["AAPL", "Apple Inc.", "SERVICES-PREPACKAGED SOFTWARE"],
        catalog=_CATALOG,
        index=index,
    )
    # AAPL resolves once (symbol then name both map to it), then the SIC code.
    assert result == ["AAPL", "AAA"]


def test_resolve_unknown_term_is_dropped() -> None:
    assert tk.resolve_terms_to_tickers(["not a real thing"], catalog=_CATALOG) == []


def test_resolve_term_to_single_ticker() -> None:
    assert tk.resolve_term_to_ticker("Apple Inc.", catalog=_CATALOG) == "AAPL"
    assert tk.resolve_term_to_ticker("nope", catalog=_CATALOG) is None


# ── get_ticker_name_map ──────────────────────────────────────


def test_get_ticker_name_map_skips_blank_symbols() -> None:
    tk.get_ticker_name_map.cache_clear()
    df = pl.DataFrame(
        {"Ticker": ["AAPL", "MSFT", ""], "Name": ["Apple Inc.", "Microsoft Corp", "X"]}
    )
    with patch("bestee_compute.stocks.tickers.get_all_tickers_df", return_value=df):
        result = tk.get_ticker_name_map()
    tk.get_ticker_name_map.cache_clear()
    assert result == {"AAPL": "Apple Inc.", "MSFT": "Microsoft Corp"}
