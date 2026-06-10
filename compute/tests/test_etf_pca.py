"""Tests for bestee_compute.etf.pca — cross-ETF PCA on weighted holdings."""

from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from bestee_compute.etf.pca import ETFPCAResult, _join_key, etf_pca

_NORMALIZE_PATCH = "bestee_compute.etf.pca.get_holdings_normalized"


def _canonical(rows: list[dict[str, object]]) -> pl.DataFrame:
    """Build a frame in the canonical normalize schema for testing."""
    return pl.DataFrame(
        rows,
        schema={
            "ticker": pl.Utf8,
            "name": pl.Utf8,
            "weight": pl.Float64,
            "market_value": pl.Float64,
            "issuer": pl.Utf8,
        },
    )


# ── _join_key ────────────────────────────────────────────────────────


class TestJoinKey:
    def test_uses_ticker_when_present(self) -> None:
        df = _canonical(
            [
                {
                    "ticker": "NVDA",
                    "name": "Nvidia Corp",
                    "weight": 0.5,
                    "market_value": None,
                    "issuer": "VanEck",
                }
            ]
        )
        out = _join_key(df)
        assert out["holding"].to_list() == ["NVDA"]
        assert out["weight"].to_list() == [0.5]

    def test_uppercases_and_strips_ticker(self) -> None:
        df = _canonical(
            [
                {
                    "ticker": "  msft ",
                    "name": "Microsoft",
                    "weight": 0.3,
                    "market_value": None,
                    "issuer": "SPDR",
                }
            ]
        )
        assert _join_key(df)["holding"].to_list() == ["MSFT"]

    def test_falls_back_to_name_when_ticker_missing(self) -> None:
        df = _canonical(
            [
                {
                    "ticker": None,
                    "name": "US Treasury 10Y",
                    "weight": 0.1,
                    "market_value": None,
                    "issuer": "iShares",
                }
            ]
        )
        assert _join_key(df)["holding"].to_list() == ["US TREASURY 10Y"]

    def test_drops_rows_with_no_identifier(self) -> None:
        df = _canonical(
            [
                {
                    "ticker": "NVDA",
                    "name": None,
                    "weight": 0.5,
                    "market_value": None,
                    "issuer": "VanEck",
                },
                {
                    "ticker": None,
                    "name": None,
                    "weight": 0.5,
                    "market_value": None,
                    "issuer": "VanEck",
                },
                {
                    "ticker": "  ",
                    "name": "  ",
                    "weight": 0.5,
                    "market_value": None,
                    "issuer": "VanEck",
                },
            ]
        )
        assert _join_key(df)["holding"].to_list() == ["NVDA"]

    def test_drops_rows_with_no_weight(self) -> None:
        df = _canonical(
            [
                {
                    "ticker": "NVDA",
                    "name": "Nvidia",
                    "weight": 0.5,
                    "market_value": None,
                    "issuer": "VanEck",
                },
                {
                    "ticker": "MSFT",
                    "name": "Microsoft",
                    "weight": None,
                    "market_value": None,
                    "issuer": "VanEck",
                },
            ]
        )
        assert _join_key(df)["holding"].to_list() == ["NVDA"]

    def test_sums_duplicates(self) -> None:
        """Some ETFs split a position across rows (e.g. tax lots, swaps)."""
        df = _canonical(
            [
                {
                    "ticker": "NVDA",
                    "name": "Nvidia",
                    "weight": 0.3,
                    "market_value": None,
                    "issuer": "Roundhill",
                },
                {
                    "ticker": "NVDA",
                    "name": "Nvidia Swap",
                    "weight": 0.2,
                    "market_value": None,
                    "issuer": "Roundhill",
                },
            ]
        )
        out = _join_key(df).sort("holding")
        assert out["holding"].to_list() == ["NVDA"]
        assert out["weight"].to_list() == [pytest.approx(0.5)]


# ── etf_pca ──────────────────────────────────────────────────────────


def _stub_provider(holdings_by_ticker: dict[str, pl.DataFrame]):
    """Return a stub for ``get_holdings_normalized`` keyed by ticker."""

    def side_effect(ticker: str) -> pl.DataFrame:
        return holdings_by_ticker[ticker.upper()]

    return side_effect


def _two_etf_holdings() -> dict[str, pl.DataFrame]:
    """Two ETFs with partially overlapping US-equity holdings.

    ETF_A: 60% NVDA, 40% MSFT
    ETF_B: 30% NVDA, 70% AAPL
    """
    return {
        "ETF_A": _canonical(
            [
                {
                    "ticker": "NVDA",
                    "name": "Nvidia Corp",
                    "weight": 0.6,
                    "market_value": None,
                    "issuer": "VanEck",
                },
                {
                    "ticker": "MSFT",
                    "name": "Microsoft",
                    "weight": 0.4,
                    "market_value": None,
                    "issuer": "VanEck",
                },
            ]
        ),
        "ETF_B": _canonical(
            [
                {
                    "ticker": "NVDA",
                    "name": "Nvidia Corp",
                    "weight": 0.3,
                    "market_value": None,
                    "issuer": "SPDR",
                },
                {
                    "ticker": "AAPL",
                    "name": "Apple Inc",
                    "weight": 0.7,
                    "market_value": None,
                    "issuer": "SPDR",
                },
            ]
        ),
    }


@patch(_NORMALIZE_PATCH)
def test_returns_pca_result(mock_norm) -> None:
    mock_norm.side_effect = _stub_provider(_two_etf_holdings())
    result = etf_pca(["ETF_A", "ETF_B"])

    assert isinstance(result, ETFPCAResult)
    assert result.tickers == ["ETF_A", "ETF_B"]
    # Union of holdings is the 3 unique tickers, in sorted order.
    assert result.holdings == ["AAPL", "MSFT", "NVDA"]


@patch(_NORMALIZE_PATCH)
def test_weight_matrix_shape_and_values(mock_norm) -> None:
    """ETF_A holds NVDA/MSFT; ETF_B holds NVDA/AAPL — zeros elsewhere."""
    mock_norm.side_effect = _stub_provider(_two_etf_holdings())
    result = etf_pca(["ETF_A", "ETF_B"])

    assert result.weights.shape == (2, 3)
    # Columns in result.holdings order: AAPL, MSFT, NVDA.
    np.testing.assert_allclose(result.weights[0], [0.0, 0.4, 0.6])
    np.testing.assert_allclose(result.weights[1], [0.7, 0.0, 0.3])


@patch(_NORMALIZE_PATCH)
def test_components_are_orthonormal(mock_norm) -> None:
    """Each PC is a unit vector and they are mutually orthogonal."""
    mock_norm.side_effect = _stub_provider(_two_etf_holdings())
    result = etf_pca(["ETF_A", "ETF_B"])

    n_comp = result.components.shape[0]
    gram = result.components @ result.components.T
    np.testing.assert_allclose(gram, np.eye(n_comp), atol=1e-10)


@patch(_NORMALIZE_PATCH)
def test_explained_variance_sums_to_total_variance(mock_norm) -> None:
    """Sum of explained variances equals total variance of the centered matrix."""
    mock_norm.side_effect = _stub_provider(_two_etf_holdings())
    result = etf_pca(["ETF_A", "ETF_B"])

    centered = result.weights - result.weights.mean(axis=0, keepdims=True)
    expected_total = (centered**2).sum() / max(len(result.tickers) - 1, 1)
    assert result.explained_variance.sum() == pytest.approx(expected_total)
    assert result.explained_variance_ratio.sum() == pytest.approx(1.0)


@patch(_NORMALIZE_PATCH)
def test_two_etfs_yield_one_meaningful_component(mock_norm) -> None:
    """N-1 informative components.  For 2 ETFs, PC1 captures all variance."""
    mock_norm.side_effect = _stub_provider(_two_etf_holdings())
    result = etf_pca(["ETF_A", "ETF_B"])

    assert result.explained_variance_ratio[0] == pytest.approx(1.0)
    # Higher-index components are numerical noise — all ~zero.
    assert np.all(result.explained_variance_ratio[1:] < 1e-9)


@patch(_NORMALIZE_PATCH)
def test_projections_separate_etfs_along_pc1(mock_norm) -> None:
    """The two ETFs should land on opposite sides of PC1 (centering means
    deviations sum to zero across rows)."""
    mock_norm.side_effect = _stub_provider(_two_etf_holdings())
    result = etf_pca(["ETF_A", "ETF_B"])

    pc1 = result.projections[:, 0]
    assert pc1[0] * pc1[1] < 0  # opposite signs
    np.testing.assert_allclose(pc1.sum(), 0.0, atol=1e-10)


@patch(_NORMALIZE_PATCH)
def test_n_components_truncation(mock_norm) -> None:
    mock_norm.side_effect = _stub_provider(_two_etf_holdings())
    result = etf_pca(["ETF_A", "ETF_B"], n_components=1)

    assert result.components.shape == (1, 3)
    assert result.projections.shape == (2, 1)
    assert result.explained_variance.shape == (1,)


@patch(_NORMALIZE_PATCH)
def test_projections_frame_shape(mock_norm) -> None:
    mock_norm.side_effect = _stub_provider(_two_etf_holdings())
    result = etf_pca(["ETF_A", "ETF_B"])

    df = result.projections_frame()
    assert df.columns[0] == "ticker"
    assert df["ticker"].to_list() == ["ETF_A", "ETF_B"]
    # 1 PC per ETF dimension after centering (n_etfs).
    assert df.shape == (2, 1 + result.projections.shape[1])


@patch(_NORMALIZE_PATCH)
def test_components_frame_sorted_by_abs_loading(mock_norm) -> None:
    mock_norm.side_effect = _stub_provider(_two_etf_holdings())
    result = etf_pca(["ETF_A", "ETF_B"])

    df = result.components_frame()
    assert set(df.columns) == {"component", "holding", "loading"}
    # Within PC1, loadings should be sorted by |loading| descending.
    pc1 = df.filter(pl.col("component") == "PC1")
    abs_loadings = [abs(v) for v in pc1["loading"].to_list()]
    assert abs_loadings == sorted(abs_loadings, reverse=True)


# ── Error paths ──────────────────────────────────────────────────────


def test_requires_at_least_two_tickers() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        etf_pca(["SPY"])


@patch(_NORMALIZE_PATCH)
def test_identical_etfs_raise(mock_norm) -> None:
    """Two ETFs with the same holdings produce zero variance everywhere."""
    same = _canonical(
        [
            {
                "ticker": "NVDA",
                "name": "Nvidia",
                "weight": 1.0,
                "market_value": None,
                "issuer": "VanEck",
            }
        ]
    )
    mock_norm.side_effect = _stub_provider({"A": same, "B": same})

    with pytest.raises(ValueError, match="identical weight vectors"):
        etf_pca(["A", "B"])


@patch(_NORMALIZE_PATCH)
def test_all_empty_etfs_raise(mock_norm) -> None:
    """No usable holdings anywhere → clear error."""
    empty = _canonical([])
    mock_norm.side_effect = _stub_provider({"A": empty, "B": empty})

    with pytest.raises(ValueError, match="usable holdings"):
        etf_pca(["A", "B"])
