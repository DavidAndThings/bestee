"""Tests for bestee.stocks.decorators — DSL parser, evaluator, and builder."""

import ast
import importlib.resources
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from great_tables import GT

import bestee.resources
from bestee.stocks import columns as cols
from bestee.stocks.decorators import (
    ComputedMetricDecorator,
    FinancialsDecorator,
    NoUpstreamError,
    ProcessingLevelError,
    SameSICategoryDecorator,
    TableDecorator,
    TickerSummaryDecorator,
    decorator_builder,
)
from bestee.stocks.models import FinancialMetric, Metric

# ── Stub upstream decorator for unit-testing chains ──────────────────


class _StubUpstream(TableDecorator):
    """Returns a fixed DataFrame for tests that exercise downstream decorators."""

    def __init__(self, df: pl.DataFrame):
        super().__init__(None)
        self._df = df

    def _build_df(self) -> pl.DataFrame:
        return self._df

    def build(self) -> GT:
        return GT(self._df)


# ── ComputedMetricDecorator._parse ───────────────────────────────────


class TestComputedMetricParse:
    def test_simple_division_expression(self) -> None:
        label, fm_refs, cm_refs, tree = ComputedMetricDecorator._parse(
            "COMPUTED_METRIC ROA "
            "(FINANCIAL_METRIC NET_INCOME 2024 4) / "
            "(FINANCIAL_METRIC TOTAL_ASSETS 2024 4)"
        )
        assert label == "Roa"
        assert len(fm_refs) == 2
        assert fm_refs[0] == FinancialMetric(Metric.NET_INCOME, 2024, 4)
        assert fm_refs[1] == FinancialMetric(Metric.TOTAL_ASSETS, 2024, 4)
        assert cm_refs == []
        assert isinstance(tree, ast.Expression)

    def test_chained_computed_metric_reference(self) -> None:
        """COMPUTED_METRIC <NAME> with no body references a prior column."""
        label, fm_refs, cm_refs, _ = ComputedMetricDecorator._parse(
            "COMPUTED_METRIC LEVERAGE (COMPUTED_METRIC ROE) / (COMPUTED_METRIC ROA)"
        )
        assert label == "Leverage"
        assert fm_refs == []
        # Capitalize matches how prior decorators name their column.
        assert cm_refs == ["Roe", "Roa"]

    def test_mixed_fm_and_cm_references(self) -> None:
        label, fm_refs, cm_refs, _ = ComputedMetricDecorator._parse(
            "COMPUTED_METRIC MIXED "
            "(COMPUTED_METRIC ROA) * (FINANCIAL_METRIC REVENUE 2024 4)"
        )
        assert label == "Mixed"
        assert fm_refs == [FinancialMetric(Metric.REVENUE, 2024, 4)]
        assert cm_refs == ["Roa"]

    def test_numeric_literals_and_unary(self) -> None:
        _, fm_refs, cm_refs, tree = ComputedMetricDecorator._parse(
            "COMPUTED_METRIC X -((FINANCIAL_METRIC REVENUE 2024 4) * 2.5)"
        )
        assert fm_refs == [FinancialMetric(Metric.REVENUE, 2024, 4)]
        assert cm_refs == []
        assert isinstance(tree, ast.Expression)

    def test_malformed_top_level_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid expression"):
            ComputedMetricDecorator._parse("not a valid expression")

    def test_unknown_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown metric"):
            ComputedMetricDecorator._parse(
                "COMPUTED_METRIC X (FINANCIAL_METRIC NOT_A_METRIC 2024 4)"
            )

    def test_syntactically_invalid_arithmetic_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid arithmetic"):
            ComputedMetricDecorator._parse(
                "COMPUTED_METRIC X (FINANCIAL_METRIC REVENUE 2024 4) +"
            )


# ── ComputedMetricDecorator._evaluate ────────────────────────────────


def _expr(src: str) -> ast.Expression:
    return ast.parse(src, mode="eval")


class TestComputedMetricEvaluate:
    def test_constant(self) -> None:
        assert ComputedMetricDecorator._evaluate(_expr("42"), {}) == 42.0

    def test_addition_subtraction(self) -> None:
        assert ComputedMetricDecorator._evaluate(_expr("3 + 4 - 1"), {}) == 6.0

    def test_division(self) -> None:
        assert ComputedMetricDecorator._evaluate(_expr("10 / 4"), {}) == 2.5

    def test_division_by_zero_returns_none(self) -> None:
        assert ComputedMetricDecorator._evaluate(_expr("10 / 0"), {}) is None

    def test_modulo_by_zero_returns_none(self) -> None:
        assert ComputedMetricDecorator._evaluate(_expr("10 % 0"), {}) is None

    def test_power(self) -> None:
        assert ComputedMetricDecorator._evaluate(_expr("2 ** 3"), {}) == 8.0

    def test_unary_minus(self) -> None:
        assert ComputedMetricDecorator._evaluate(_expr("-5"), {}) == -5.0

    def test_placeholder_lookup(self) -> None:
        env: dict[str, float | None] = {"_v0": 100.0, "_v1": 25.0}
        assert ComputedMetricDecorator._evaluate(_expr("_v0 / _v1"), env) == 4.0

    def test_missing_placeholder_propagates_none(self) -> None:
        # _v0 missing from env → ast.Name returns None → BinOp returns None.
        env: dict[str, float | None] = {"_v1": 25.0}
        assert ComputedMetricDecorator._evaluate(_expr("_v0 / _v1"), env) is None

    def test_explicit_none_propagates(self) -> None:
        env: dict[str, float | None] = {"_v0": None, "_v1": 25.0}
        assert ComputedMetricDecorator._evaluate(_expr("_v0 * _v1"), env) is None

    def test_unsupported_operator_raises(self) -> None:
        # Bitwise-and is not in the allowed operators.
        with pytest.raises(ValueError, match="Unsupported binary operator"):
            ComputedMetricDecorator._evaluate(_expr("5 & 3"), {})

    def test_unsupported_literal_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported literal"):
            ComputedMetricDecorator._evaluate(_expr("'a string'"), {})


# ── decorator_builder dispatch ───────────────────────────────────────


class TestDecoratorBuilder:
    def test_single_stocks_command(self) -> None:
        result = decorator_builder([["STOCKS"]])
        assert isinstance(result, TickerSummaryDecorator)

    def test_stocks_with_same_sic(self) -> None:
        result = decorator_builder([["STOCKS"], ["SAME_SIC_CATEGORY_AS", "AAPL"]])
        assert isinstance(result, SameSICategoryDecorator)
        assert result.ticker == "AAPL"

    def test_financial_metrics_accumulate_into_single_decorator(self) -> None:
        """Two FINANCIAL_METRIC commands share one FinancialsDecorator."""
        result = decorator_builder(
            [
                ["STOCKS"],
                ["FINANCIAL_METRIC", "REVENUE", "2024", "4"],
                ["FINANCIAL_METRIC", "EBITDA", "2024", "4"],
            ]
        )
        assert isinstance(result, FinancialsDecorator)
        assert len(result.metrics) == 2

    def test_computed_metric_chains_on_top(self) -> None:
        result = decorator_builder(
            [
                ["STOCKS"],
                ["FINANCIAL_METRIC", "NET_INCOME", "2024", "4"],
                [
                    "COMPUTED_METRIC",
                    "ROA",
                    "(FINANCIAL_METRIC",
                    "NET_INCOME",
                    "2024",
                    "4)",
                    "/",
                    "(FINANCIAL_METRIC",
                    "TOTAL_ASSETS",
                    "2024",
                    "4)",
                ],
            ]
        )
        assert isinstance(result, ComputedMetricDecorator)
        # Upstream is the FinancialsDecorator from level 1.
        assert isinstance(result._upstream, FinancialsDecorator)

    def test_two_computed_metrics_chain_in_input_order(self) -> None:
        result = decorator_builder(
            [
                ["STOCKS"],
                [
                    "COMPUTED_METRIC",
                    "ROE",
                    "(FINANCIAL_METRIC",
                    "NET_INCOME",
                    "2024",
                    "4)",
                    "/",
                    "(FINANCIAL_METRIC",
                    "TOTAL_EQUITY",
                    "2024",
                    "4)",
                ],
                [
                    "COMPUTED_METRIC",
                    "ROA",
                    "(FINANCIAL_METRIC",
                    "NET_INCOME",
                    "2024",
                    "4)",
                    "/",
                    "(FINANCIAL_METRIC",
                    "TOTAL_ASSETS",
                    "2024",
                    "4)",
                ],
            ]
        )
        # Outer is the last COMPUTED_METRIC seen.
        assert isinstance(result, ComputedMetricDecorator)
        assert result._metric_label == "Roa"
        # Its upstream is the previous COMPUTED_METRIC.
        assert isinstance(result._upstream, ComputedMetricDecorator)
        assert result._upstream._metric_label == "Roe"

    def test_ordering_does_not_matter_within_phases(self) -> None:
        """COMPUTED_METRIC before STOCKS still works — phases are reordered."""
        result = decorator_builder(
            [
                [
                    "COMPUTED_METRIC",
                    "X",
                    "(FINANCIAL_METRIC",
                    "REVENUE",
                    "2024",
                    "4)",
                    "*",
                    "2",
                ],
                ["FINANCIAL_METRIC", "REVENUE", "2024", "4"],
                ["STOCKS"],
            ]
        )
        assert isinstance(result, ComputedMetricDecorator)
        assert isinstance(result._upstream, FinancialsDecorator)

    def test_unknown_command_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="Unrecognized"):
            decorator_builder([["STOCKS"], ["NOT_A_COMMAND"]])

    def test_same_sic_without_upstream_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="SAME_SIC_CATEGORY_AS"):
            decorator_builder([["SAME_SIC_CATEGORY_AS", "AAPL"]])

    def test_financial_metric_without_upstream_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="FINANCIAL_METRIC"):
            decorator_builder([["FINANCIAL_METRIC", "REVENUE", "2024", "4"]])

    def test_computed_metric_without_upstream_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="COMPUTED_METRIC"):
            decorator_builder(
                [
                    [
                        "COMPUTED_METRIC",
                        "X",
                        "(FINANCIAL_METRIC",
                        "REVENUE",
                        "2024",
                        "4)",
                    ]
                ]
            )

    def test_empty_command_list_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="No decorator"):
            decorator_builder([])

    def test_unknown_financial_metric_name_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="Unknown FINANCIAL_METRIC"):
            decorator_builder(
                [
                    ["STOCKS"],
                    ["FINANCIAL_METRIC", "NOT_A_METRIC", "2024", "4"],
                ]
            )


# ── TableDecorator base / NoUpstreamError ────────────────────────────


class TestNoUpstreamError:
    def test_build_upstream_df_raises_when_none(self) -> None:
        class _Stage(TableDecorator):
            def _build_df(self) -> pl.DataFrame:
                return self._build_upstream_df()

            def build(self) -> GT:
                return GT(self._build_df())

        with pytest.raises(NoUpstreamError):
            _Stage().build()


# ── SameSICategoryDecorator integration ──────────────────────────────


class TestSameSICategoryDecorator:
    def test_filters_by_target_sic(self) -> None:
        df = pl.DataFrame(
            {
                cols.TICKER: ["AAPL", "MSFT", "GOOG", "F"],
                cols.SIC_CODE: ["3571", "3571", "3571", "3711"],
            }
        )
        upstream = _StubUpstream(df)
        result_df = SameSICategoryDecorator(
            ticker="AAPL", upstream=upstream
        )._build_df()
        assert sorted(result_df[cols.TICKER].to_list()) == ["AAPL", "GOOG", "MSFT"]

    def test_missing_ticker_raises(self) -> None:
        df = pl.DataFrame({cols.TICKER: ["AAPL"], cols.SIC_CODE: ["3571"]})
        upstream = _StubUpstream(df)
        with pytest.raises(ValueError, match="not found in upstream"):
            SameSICategoryDecorator(ticker="MSFT", upstream=upstream).build()

    def test_null_sic_raises(self) -> None:
        df = pl.DataFrame(
            {cols.TICKER: ["AAPL", "MSFT"], cols.SIC_CODE: [None, "3571"]},
            schema={cols.TICKER: pl.Utf8, cols.SIC_CODE: pl.Utf8},
        )
        upstream = _StubUpstream(df)
        with pytest.raises(ValueError, match="no SIC code"):
            SameSICategoryDecorator(ticker="AAPL", upstream=upstream).build()


# ── ComputedMetricDecorator integration ──────────────────────────────


_FINANCIALS_PATCH = "bestee.stocks.decorators.build_financials_df"


class TestComputedMetricBuild:
    @patch(_FINANCIALS_PATCH)
    def test_computes_per_ticker(self, mock_build: MagicMock) -> None:
        # Upstream has two tickers.
        upstream_df = pl.DataFrame({cols.TICKER: ["AAPL", "MSFT"]})
        # Mock financials returns NET_INCOME and TOTAL_ASSETS.
        ni = FinancialMetric(Metric.NET_INCOME, 2024, 4)
        ta = FinancialMetric(Metric.TOTAL_ASSETS, 2024, 4)
        mock_build.return_value = pl.DataFrame(
            {
                cols.TICKER: ["AAPL", "MSFT"],
                ni.label: [100.0, 200.0],
                ta.label: [400.0, 1000.0],
            }
        )

        deco = ComputedMetricDecorator(
            expression=(
                "COMPUTED_METRIC ROA "
                "(FINANCIAL_METRIC NET_INCOME 2024 4) / "
                "(FINANCIAL_METRIC TOTAL_ASSETS 2024 4)"
            ),
            upstream=_StubUpstream(upstream_df),
        )
        result_df = deco._build_df()
        assert result_df["Roa"].to_list() == [0.25, 0.2]

    @patch(_FINANCIALS_PATCH)
    def test_missing_data_yields_none(self, mock_build: MagicMock) -> None:
        upstream_df = pl.DataFrame({cols.TICKER: ["AAPL", "MSFT"]})
        ni = FinancialMetric(Metric.NET_INCOME, 2024, 4)
        ta = FinancialMetric(Metric.TOTAL_ASSETS, 2024, 4)
        # MSFT has no total_assets data.
        mock_build.return_value = pl.DataFrame(
            {
                cols.TICKER: ["AAPL", "MSFT"],
                ni.label: [100.0, 200.0],
                ta.label: [400.0, None],
            }
        )

        deco = ComputedMetricDecorator(
            expression=(
                "COMPUTED_METRIC ROA "
                "(FINANCIAL_METRIC NET_INCOME 2024 4) / "
                "(FINANCIAL_METRIC TOTAL_ASSETS 2024 4)"
            ),
            upstream=_StubUpstream(upstream_df),
        )
        result_df = deco._build_df()
        assert result_df["Roa"].to_list() == [0.25, None]

    @patch(_FINANCIALS_PATCH)
    def test_division_by_zero_yields_none(self, mock_build: MagicMock) -> None:
        upstream_df = pl.DataFrame({cols.TICKER: ["AAPL"]})
        ni = FinancialMetric(Metric.NET_INCOME, 2024, 4)
        ta = FinancialMetric(Metric.TOTAL_ASSETS, 2024, 4)
        mock_build.return_value = pl.DataFrame(
            {
                cols.TICKER: ["AAPL"],
                ni.label: [100.0],
                ta.label: [0.0],
            }
        )

        deco = ComputedMetricDecorator(
            expression=(
                "COMPUTED_METRIC ROA "
                "(FINANCIAL_METRIC NET_INCOME 2024 4) / "
                "(FINANCIAL_METRIC TOTAL_ASSETS 2024 4)"
            ),
            upstream=_StubUpstream(upstream_df),
        )
        result_df = deco._build_df()
        assert result_df["Roa"].to_list() == [None]

    def test_chained_computed_metric_reads_upstream_column(self) -> None:
        """A second ComputedMetricDecorator can reference the first's column."""
        upstream_df = pl.DataFrame({cols.TICKER: ["AAPL", "MSFT"], "Roa": [0.25, 0.10]})
        deco = ComputedMetricDecorator(
            expression="COMPUTED_METRIC DOUBLED_ROA (COMPUTED_METRIC ROA) * 2",
            upstream=_StubUpstream(upstream_df),
        )
        result_df = deco._build_df()
        assert result_df["Doubled_roa"].to_list() == [0.5, 0.2]

    def test_missing_upstream_reference_raises(self) -> None:
        upstream_df = pl.DataFrame({cols.TICKER: ["AAPL"]})
        deco = ComputedMetricDecorator(
            expression="COMPUTED_METRIC X (COMPUTED_METRIC MISSING) * 2",
            upstream=_StubUpstream(upstream_df),
        )
        with pytest.raises(ValueError, match="not found in upstream"):
            deco._build_df()


# ── Stress-input regression tests ────────────────────────────────────


_STRESS_INPUTS = sorted(
    p.name
    for p in importlib.resources.files(bestee.resources).iterdir()
    if p.name.endswith(".txt")
)


@pytest.mark.parametrize("filename", _STRESS_INPUTS)
def test_stress_input_parses_and_builds(filename: str) -> None:
    """Each bundled DSL script parses and assembles into a decorator chain."""
    text = (
        importlib.resources.files(bestee.resources)
        .joinpath(filename)
        .read_text(encoding="utf-8")
    )
    commands = [line.split() for line in text.splitlines() if line.strip()]
    assert commands, f"{filename} has no commands"

    pipeline = decorator_builder(commands)

    # Every stress input ends with at least one transformation, so the
    # output is never just the bare TickerSummaryDecorator.
    assert isinstance(pipeline, TableDecorator)
