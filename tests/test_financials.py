"""Tests for bestee.stocks.financials."""

from unittest.mock import MagicMock, patch

from great_tables import GT

from bestee.stocks.financials import (
    _group_metrics,
    _period_sort_key,
    _resolve_ticker,
    build_financials_table,
)
from bestee.stocks.models import FinancialMetric, Metric, StatementType

_CLIENT_PATCH = "bestee.stocks.financials.get_client"

# Shorthand helpers for building FinancialMetric instances.
_REV_Q4_24 = FinancialMetric(Metric.REVENUE, fiscal_year=2024, fiscal_quarter=4)
_REV_Q4_23 = FinancialMetric(Metric.REVENUE, fiscal_year=2023, fiscal_quarter=4)
_EBITDA_Q4_24 = FinancialMetric(Metric.EBITDA, fiscal_year=2024, fiscal_quarter=4)
_ASSETS_Q4_24 = FinancialMetric(Metric.TOTAL_ASSETS, fiscal_year=2024, fiscal_quarter=4)
_PE_Q4_24 = FinancialMetric(Metric.PE_RATIO, fiscal_year=2024, fiscal_quarter=4)
_PE_Q4_23 = FinancialMetric(Metric.PE_RATIO, fiscal_year=2023, fiscal_quarter=4)


# ── Mock helpers ─────────────────────────────────────────────────────


def _make_financial(
    tickers: list[str],
    period_end: str,
    **fields: float,
) -> MagicMock:
    mock = MagicMock(spec=[])  # empty spec — unset attrs raise AttributeError
    mock.tickers = tickers
    mock.period_end = period_end
    for k, v in fields.items():
        setattr(mock, k, v)
    return mock


def _make_ratio(ticker: str, date: str, **fields: float) -> MagicMock:
    mock = MagicMock(spec=[])  # empty spec avoids auto-attributes
    mock.ticker = ticker
    mock.date = date
    for k, v in fields.items():
        setattr(mock, k, v)
    return mock


# ── Unit tests for helpers ───────────────────────────────────────────


class TestGroupMetrics:
    def test_groups_by_statement_and_period(self) -> None:
        metrics = [_REV_Q4_24, _EBITDA_Q4_24, _ASSETS_Q4_24, _PE_Q4_24]
        groups = _group_metrics(metrics)

        # Income statement Q4 2024, balance sheet Q4 2024, ratios (one group)
        assert len(groups) == 3

        income_key = (StatementType.INCOME_STATEMENT, 2024, 4)
        assert income_key in groups
        assert _REV_Q4_24 in groups[income_key]
        assert _EBITDA_Q4_24 in groups[income_key]

    def test_same_metric_different_periods_creates_two_groups(self) -> None:
        metrics = [_REV_Q4_24, _REV_Q4_23]
        groups = _group_metrics(metrics)

        assert len(groups) == 2
        assert (StatementType.INCOME_STATEMENT, 2024, 4) in groups
        assert (StatementType.INCOME_STATEMENT, 2023, 4) in groups

    def test_ratios_collapse_into_single_group(self) -> None:
        """Ratios from different periods should share one group."""
        metrics = [_PE_Q4_24, _PE_Q4_23]
        groups = _group_metrics(metrics)

        # Both ratio metrics go to the single _RATIOS_KEY group.
        assert len(groups) == 1

    def test_empty_list(self) -> None:
        assert _group_metrics([]) == {}


class TestResolveTicker:
    def test_resolves_from_tickers_list(self) -> None:
        result = MagicMock(spec=[])
        result.tickers = ["AAPL", "AAPL"]
        assert _resolve_ticker(result, {"AAPL", "MSFT"}) == "AAPL"

    def test_resolves_from_ticker_singular(self) -> None:
        result = MagicMock(spec=[])
        result.ticker = "GOOG"
        assert _resolve_ticker(result, {"GOOG"}) == "GOOG"

    def test_returns_none_for_unknown(self) -> None:
        result = MagicMock(spec=[])
        result.ticker = "TSLA"
        assert _resolve_ticker(result, {"AAPL"}) is None


class TestPeriodSortKey:
    def test_uses_period_end(self) -> None:
        r = MagicMock(spec=[])
        r.period_end = "2024-12-31"
        assert _period_sort_key(r) == "2024-12-31"

    def test_uses_date_fallback(self) -> None:
        r = MagicMock(spec=[])
        r.date = "2025-01-15"
        assert _period_sort_key(r) == "2025-01-15"

    def test_returns_empty_when_neither(self) -> None:
        r = MagicMock(spec=[])
        assert _period_sort_key(r) == ""


class TestFinancialMetricLabel:
    def test_label_includes_period(self) -> None:
        assert _REV_Q4_24.label == "Revenue (FY2024 Q4)"

    def test_different_periods_produce_different_labels(self) -> None:
        assert _REV_Q4_24.label != _REV_Q4_23.label
        assert "FY2024" in _REV_Q4_24.label
        assert "FY2023" in _REV_Q4_23.label


# ── Integration tests for build_financials_table ─────────────────────


class TestBuildFinancialsTable:
    @patch(_CLIENT_PATCH)
    def test_returns_gt_with_correct_data(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        """Table should contain values from the API response."""
        client = mock_get_client.return_value
        client.list_financials_income_statements.return_value = iter(
            [
                _make_financial(["AAPL"], "2024-09-30", revenue=391e9, ebitda=134e9),
                _make_financial(["MSFT"], "2024-06-30", revenue=245e9, ebitda=125e9),
            ]
        )

        metrics = [_REV_Q4_24, _EBITDA_Q4_24]
        result = build_financials_table(["AAPL", "MSFT"], metrics)

        assert isinstance(result, GT)
        html = result.as_raw_html()
        assert "AAPL" in html
        assert "MSFT" in html

    @patch(_CLIENT_PATCH)
    def test_only_calls_needed_endpoints(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        """Should NOT call endpoints whose statement type has no metrics."""
        client = mock_get_client.return_value
        client.list_financials_income_statements.return_value = iter([])

        build_financials_table(["AAPL"], [_REV_Q4_24])

        client.list_financials_income_statements.assert_called_once()
        client.list_financials_balance_sheets.assert_not_called()
        client.list_financials_cash_flow_statements.assert_not_called()
        client.list_financials_ratios.assert_not_called()

    @patch(_CLIENT_PATCH)
    def test_batches_tickers_with_period_filters(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        """Tickers should be batched; fiscal period should be sent."""
        client = mock_get_client.return_value
        client.list_financials_income_statements.return_value = iter([])

        build_financials_table(["AAPL", "MSFT", "GOOG"], [_REV_Q4_24])

        client.list_financials_income_statements.assert_called_once_with(
            params={"tickers.any_of": "AAPL,MSFT,GOOG"},
            fiscal_year=2024,
            fiscal_quarter=4,
            timeframe="quarterly",
        )

    @patch(_CLIENT_PATCH)
    def test_same_metric_two_periods_makes_two_calls(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        """Revenue for Q4 2024 and Q4 2023 should produce two API calls."""
        client = mock_get_client.return_value
        client.list_financials_income_statements.return_value = iter([])

        build_financials_table(["AAPL"], [_REV_Q4_24, _REV_Q4_23])

        assert client.list_financials_income_statements.call_count == 2

    @patch(_CLIENT_PATCH)
    def test_same_metric_two_periods_produces_two_columns(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        """Each period should have its own column in the output."""
        client = mock_get_client.return_value

        # First call (Q4 2024) → returns 2024 revenue
        # Second call (Q4 2023) → returns 2023 revenue
        client.list_financials_income_statements.side_effect = [
            iter([_make_financial(["AAPL"], "2024-09-30", revenue=391e9)]),
            iter([_make_financial(["AAPL"], "2023-09-30", revenue=383e9)]),
        ]

        result = build_financials_table(["AAPL"], [_REV_Q4_24, _REV_Q4_23])
        html = result.as_raw_html()

        assert "Revenue (FY2024 Q4)" in html
        assert "Revenue (FY2023 Q4)" in html

    @patch(_CLIENT_PATCH)
    def test_multiple_statement_types(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        """Metrics from different statements should each trigger one call."""
        client = mock_get_client.return_value
        client.list_financials_income_statements.return_value = iter(
            [_make_financial(["AAPL"], "2024-09-30", revenue=391e9)]
        )
        client.list_financials_ratios.return_value = iter(
            [_make_ratio("AAPL", "2025-01-15", price_to_earnings=32.5)]
        )

        result = build_financials_table(["AAPL"], [_REV_Q4_24, _PE_Q4_24])

        assert isinstance(result, GT)
        client.list_financials_income_statements.assert_called_once()
        client.list_financials_ratios.assert_called_once()
        client.list_financials_balance_sheets.assert_not_called()

    @patch(_CLIENT_PATCH)
    def test_ratios_skip_period_filters(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        """Ratios endpoint should NOT receive fiscal_year/quarter."""
        client = mock_get_client.return_value
        client.list_financials_ratios.return_value = iter([])

        build_financials_table(["AAPL"], [_PE_Q4_24])

        # Should only pass params (ticker filter), NOT fiscal_year etc.
        client.list_financials_ratios.assert_called_once_with(
            params={"ticker.any_of": "AAPL"},
        )

    @patch(_CLIENT_PATCH)
    def test_missing_data_shows_placeholder(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        """Tickers with no data should show the missing-text placeholder."""
        client = mock_get_client.return_value
        client.list_financials_income_statements.return_value = iter(
            [_make_financial(["AAPL"], "2024-09-30", revenue=391e9)]
        )

        result = build_financials_table(["AAPL", "MSFT"], [_REV_Q4_24])
        html = result.as_raw_html()

        assert "—" in html or "&mdash;" in html
