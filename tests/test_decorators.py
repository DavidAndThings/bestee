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
    TimeSeriesCacheDecorator,
    TimeSeriesDerivedDecorator,
    TimeSeriesMetricDecorator,
    decorator_builder,
)
from bestee.stocks.models import (
    FinancialMetric,
    Metric,
    TimeSeriesDef,
    TimeSeriesName,
    TimeSeriesSpan,
)

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


# ── ComputedMetricDecorator parsing / validation ─────────────────────


class TestComputedMetricParse:
    def test_collects_referenced_names(self) -> None:
        deco = ComputedMetricDecorator(
            name="cm1",
            expression="(fm2 * 2) / (fm3 + fm4)",
            upstream=_StubUpstream(pl.DataFrame({cols.TICKER: []})),
        )
        assert deco._referenced_names == ["fm2", "fm3", "fm4"]

    def test_chained_computed_metric_reference(self) -> None:
        """Bare identifiers can reference columns produced upstream."""
        deco = ComputedMetricDecorator(
            name="leverage",
            expression="roe / roa",
            upstream=_StubUpstream(pl.DataFrame({cols.TICKER: []})),
        )
        assert deco._referenced_names == ["roa", "roe"]

    def test_numeric_literals_and_unary_minus(self) -> None:
        deco = ComputedMetricDecorator(
            name="x",
            expression="-(rev * 2.5)",
            upstream=_StubUpstream(pl.DataFrame({cols.TICKER: []})),
        )
        assert deco._referenced_names == ["rev"]

    def test_syntactically_invalid_arithmetic_raises(self) -> None:
        with pytest.raises(ValueError, match="not valid Python syntax"):
            ComputedMetricDecorator(
                name="x",
                expression="rev +",
                upstream=_StubUpstream(pl.DataFrame({cols.TICKER: []})),
            )

    def test_unsupported_construct_raises(self) -> None:
        """Function calls and other non-arithmetic syntax should be rejected."""
        with pytest.raises(ValueError, match="unsupported construct"):
            ComputedMetricDecorator(
                name="x",
                expression="abs(rev)",
                upstream=_StubUpstream(pl.DataFrame({cols.TICKER: []})),
            )

    def test_non_numeric_constant_raises(self) -> None:
        with pytest.raises(ValueError, match="must be numeric"):
            ComputedMetricDecorator(
                name="x",
                expression="rev + 'oops'",
                upstream=_StubUpstream(pl.DataFrame({cols.TICKER: []})),
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
        """Two named FINANCIAL_METRIC lines share one FinancialsDecorator."""
        result = decorator_builder(
            [
                ["STOCKS"],
                ["FINANCIAL_METRIC", "rev", "REVENUE", "2024", "4"],
                ["FINANCIAL_METRIC", "ebitda", "EBITDA", "2024", "4"],
            ]
        )
        assert isinstance(result, FinancialsDecorator)
        assert len(result.metrics) == 2

    def test_financial_metric_wrong_arity_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="requires 4 arguments"):
            decorator_builder(
                [
                    ["STOCKS"],
                    # Missing the leading name.
                    ["FINANCIAL_METRIC", "REVENUE", "2024", "4"],
                ]
            )

    def test_duplicate_financial_metric_name_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="already defined"):
            decorator_builder(
                [
                    ["STOCKS"],
                    ["FINANCIAL_METRIC", "rev", "REVENUE", "2024", "4"],
                    ["FINANCIAL_METRIC", "rev", "EBITDA", "2024", "4"],
                ]
            )

    def test_computed_metric_chains_on_top(self) -> None:
        result = decorator_builder(
            [
                ["STOCKS"],
                ["FINANCIAL_METRIC", "ni", "NET_INCOME", "2024", "4"],
                ["FINANCIAL_METRIC", "assets", "TOTAL_ASSETS", "2024", "4"],
                ["COMPUTED_METRIC", "roa", "ni", "/", "assets"],
            ]
        )
        assert isinstance(result, ComputedMetricDecorator)
        # Upstream is the FinancialsDecorator from level 1.
        assert isinstance(result._upstream, FinancialsDecorator)
        assert result._name == "roa"

    def test_two_computed_metrics_chain_in_input_order(self) -> None:
        result = decorator_builder(
            [
                ["STOCKS"],
                ["FINANCIAL_METRIC", "ni", "NET_INCOME", "2024", "4"],
                ["FINANCIAL_METRIC", "equity", "TOTAL_EQUITY", "2024", "4"],
                ["FINANCIAL_METRIC", "assets", "TOTAL_ASSETS", "2024", "4"],
                ["COMPUTED_METRIC", "roe", "ni", "/", "equity"],
                ["COMPUTED_METRIC", "roa", "ni", "/", "assets"],
            ]
        )
        # Outer is the last COMPUTED_METRIC seen.
        assert isinstance(result, ComputedMetricDecorator)
        assert result._name == "roa"
        # Its upstream is the previous COMPUTED_METRIC.
        assert isinstance(result._upstream, ComputedMetricDecorator)
        assert result._upstream._name == "roe"

    def test_ordering_does_not_matter_within_phases(self) -> None:
        """COMPUTED_METRIC before STOCKS still works — phases are reordered."""
        result = decorator_builder(
            [
                ["COMPUTED_METRIC", "doubled", "rev", "*", "2"],
                ["FINANCIAL_METRIC", "rev", "REVENUE", "2024", "4"],
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
            decorator_builder([["FINANCIAL_METRIC", "rev", "REVENUE", "2024", "4"]])

    def test_computed_metric_without_upstream_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="COMPUTED_METRIC"):
            decorator_builder([["COMPUTED_METRIC", "x", "rev", "*", "2"]])

    def test_computed_metric_wrong_arity_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="at least 2 arguments"):
            decorator_builder(
                [
                    ["STOCKS"],
                    ["COMPUTED_METRIC", "x"],
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
                    ["FINANCIAL_METRIC", "x", "NOT_A_METRIC", "2024", "4"],
                ]
            )

    def test_non_integer_fiscal_period_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="must be integers"):
            decorator_builder(
                [
                    ["STOCKS"],
                    ["FINANCIAL_METRIC", "x", "REVENUE", "twenty24", "4"],
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
    def test_computes_per_ticker_from_named_columns(self) -> None:
        """Bare identifiers in the expression resolve to upstream columns."""
        upstream_df = pl.DataFrame(
            {
                cols.TICKER: ["AAPL", "MSFT"],
                "ni": [100.0, 200.0],
                "assets": [400.0, 1000.0],
            }
        )
        deco = ComputedMetricDecorator(
            name="roa",
            expression="ni / assets",
            upstream=_StubUpstream(upstream_df),
        )
        result_df = deco._build_df()
        assert result_df["roa"].to_list() == [0.25, 0.2]

    def test_complex_expression(self) -> None:
        """The sample-input form: (fm2 * 2) / (fm3 + fm4)."""
        upstream_df = pl.DataFrame(
            {
                cols.TICKER: ["AAPL", "MSFT"],
                "fm2": [10.0, 20.0],
                "fm3": [5.0, 30.0],
                "fm4": [5.0, 10.0],
            }
        )
        deco = ComputedMetricDecorator(
            name="cm1",
            expression="(fm2 * 2) / (fm3 + fm4)",
            upstream=_StubUpstream(upstream_df),
        )
        result_df = deco._build_df()
        # AAPL: (10*2)/(5+5) = 2.0;  MSFT: (20*2)/(30+10) = 1.0.
        assert result_df["cm1"].to_list() == [2.0, 1.0]

    def test_missing_data_yields_none(self) -> None:
        upstream_df = pl.DataFrame(
            {
                cols.TICKER: ["AAPL", "MSFT"],
                "ni": [100.0, 200.0],
                "assets": [400.0, None],
            }
        )
        deco = ComputedMetricDecorator(
            name="roa",
            expression="ni / assets",
            upstream=_StubUpstream(upstream_df),
        )
        result_df = deco._build_df()
        assert result_df["roa"].to_list() == [0.25, None]

    def test_division_by_zero_yields_none(self) -> None:
        upstream_df = pl.DataFrame(
            {
                cols.TICKER: ["AAPL"],
                "ni": [100.0],
                "assets": [0.0],
            }
        )
        deco = ComputedMetricDecorator(
            name="roa",
            expression="ni / assets",
            upstream=_StubUpstream(upstream_df),
        )
        result_df = deco._build_df()
        assert result_df["roa"].to_list() == [None]

    def test_chained_computed_metric_reads_upstream_column(self) -> None:
        """A second ComputedMetricDecorator can reference the first's column."""
        upstream_df = pl.DataFrame({cols.TICKER: ["AAPL", "MSFT"], "roa": [0.25, 0.10]})
        deco = ComputedMetricDecorator(
            name="doubled_roa",
            expression="roa * 2",
            upstream=_StubUpstream(upstream_df),
        )
        result_df = deco._build_df()
        assert result_df["doubled_roa"].to_list() == [0.5, 0.2]

    def test_missing_upstream_reference_raises(self) -> None:
        upstream_df = pl.DataFrame({cols.TICKER: ["AAPL"]})
        deco = ComputedMetricDecorator(
            name="x",
            expression="missing * 2",
            upstream=_StubUpstream(upstream_df),
        )
        with pytest.raises(ValueError, match="unknown name"):
            deco._build_df()


# ── FinancialsDecorator name handling ────────────────────────────────


class TestFinancialsDecoratorNames:
    @patch(_FINANCIALS_PATCH)
    def test_renames_columns_from_label_to_name(self, mock_build: MagicMock) -> None:
        upstream_df = pl.DataFrame({cols.TICKER: ["AAPL", "MSFT"]})
        ni = FinancialMetric(Metric.NET_INCOME, 2024, 4)
        ta = FinancialMetric(Metric.TOTAL_ASSETS, 2024, 4)
        mock_build.return_value = pl.DataFrame(
            {
                cols.TICKER: ["AAPL", "MSFT"],
                ni.label: [100.0, 200.0],
                ta.label: [400.0, 1000.0],
            }
        )

        deco = FinancialsDecorator(upstream=_StubUpstream(upstream_df))
        deco.add_metric(name="fm_ni", metric=ni)
        deco.add_metric(name="fm_ta", metric=ta)
        df = deco._build_df()

        # Columns carry the user-supplied names, not the SDK labels.
        assert "fm_ni" in df.columns
        assert "fm_ta" in df.columns
        assert ni.label not in df.columns
        assert df["fm_ni"].to_list() == [100.0, 200.0]
        assert df["fm_ta"].to_list() == [400.0, 1000.0]

    def test_duplicate_name_raises(self) -> None:
        deco = FinancialsDecorator(upstream=_StubUpstream(pl.DataFrame()))
        deco.add_metric(
            name="rev",
            metric=FinancialMetric(Metric.REVENUE, 2024, 4),
        )
        with pytest.raises(ValueError, match="already defined"):
            deco.add_metric(
                name="rev",
                metric=FinancialMetric(Metric.EBITDA, 2024, 4),
            )

    def test_no_metrics_passes_upstream_through(self) -> None:
        """With no metrics added, no API call should happen."""
        upstream_df = pl.DataFrame({cols.TICKER: ["AAPL"], "Existing": [42.0]})
        deco = FinancialsDecorator(upstream=_StubUpstream(upstream_df))
        df = deco._build_df()
        assert df.columns == [cols.TICKER, "Existing"]


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


# ── TimeSeriesMetricDecorator integration ────────────────────────────


_GET_TS_PATCH = "bestee.stocks.decorators.get_time_series"


def _ts_def(
    name: TimeSeriesName = TimeSeriesName.CLOSE_PRICE,
    span: TimeSeriesSpan = TimeSeriesSpan.DAY,
    start: str = "2025-01-02",
    end: str = "2025-01-08",
    multiplier: int = 1,
) -> TimeSeriesDef:
    return TimeSeriesDef(
        name=name, span=span, start=start, end=end, multiplier=multiplier
    )


class TestTimeSeriesMetricDecorator:
    @patch(_GET_TS_PATCH)
    def test_adds_scalar_column_per_ticker(self, mock_get: MagicMock) -> None:
        """One reducer applied per ticker → one Float64 entry per row.

        Both series here are perfectly linear (constant slope), so R²
        should be 1.0 for both.  The column lands under the
        user-supplied *name*, not the metric name.
        """
        mock_get.side_effect = lambda ticker, _ts_def: {
            "AAPL": [100.0, 101.0, 102.0],
            "MSFT": [200.0, 201.0],
        }[ticker]

        upstream_df = pl.DataFrame({cols.TICKER: ["AAPL", "MSFT"]})
        ts = _ts_def()
        chain = TimeSeriesMetricDecorator(
            ts_def=ts,
            upstream_decorator=TimeSeriesCacheDecorator(
                ts_def=ts, upstream_decorator=_StubUpstream(upstream_df)
            ),
            name="tsm1",
            metric="RSquared",
        )

        result = chain._build_df()
        assert "tsm1" in result.columns
        # Metric name is not used as the column label anymore.
        assert "RSquared" not in result.columns
        assert result.schema["tsm1"] == pl.Float64
        assert result["tsm1"].to_list() == [pytest.approx(1.0), pytest.approx(1.0)]

    @patch(_GET_TS_PATCH)
    def test_cache_serves_repeated_lookups(self, mock_get: MagicMock) -> None:
        """A cache upstream should only hit the network once per ticker
        across multiple downstream consumers."""
        mock_get.return_value = [1.0, 2.0]

        upstream_df = pl.DataFrame({cols.TICKER: ["AAPL"]})
        ts = _ts_def()
        cache = TimeSeriesCacheDecorator(
            ts_def=ts, upstream_decorator=_StubUpstream(upstream_df)
        )
        TimeSeriesMetricDecorator(
            ts_def=ts, upstream_decorator=cache, name="a", metric="RSquared"
        )._build_df()
        TimeSeriesMetricDecorator(
            ts_def=ts, upstream_decorator=cache, name="b", metric="RSquared"
        )._build_df()

        assert mock_get.call_count == 1

    def test_no_producer_raises_not_implemented(self) -> None:
        """Without a cache (or other producer) upstream, the chain walk
        runs out of stages and surfaces the NotImplementedError."""
        upstream_df = pl.DataFrame({cols.TICKER: ["AAPL"]})
        deco = TimeSeriesMetricDecorator(
            ts_def=_ts_def(),
            upstream_decorator=_StubUpstream(upstream_df),
            name="tsm1",
            metric="RSquared",
        )
        with pytest.raises(NotImplementedError):
            deco._build_df()

    @patch(_GET_TS_PATCH)
    def test_build_returns_gt_with_metric_column(self, mock_get: MagicMock) -> None:
        mock_get.return_value = [1.0, 2.0, 3.0]
        upstream_df = pl.DataFrame({cols.TICKER: ["AAPL"]})
        ts = _ts_def()
        deco = TimeSeriesMetricDecorator(
            ts_def=ts,
            upstream_decorator=TimeSeriesCacheDecorator(
                ts_def=ts, upstream_decorator=_StubUpstream(upstream_df)
            ),
            name="tsm1",
            metric="RSquared",
        )

        gt = deco.build()
        assert isinstance(gt, GT)
        html = gt.as_raw_html()
        # User name + metric name both surfaced in the rendered header.
        assert "tsm1" in html
        assert "RSquared" in html

    def test_unknown_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown time-series metric"):
            TimeSeriesMetricDecorator(
                ts_def=_ts_def(),
                upstream_decorator=_StubUpstream(pl.DataFrame({cols.TICKER: []})),
                name="tsm1",
                metric="NoSuchMetric",
            )


# ── Scalar-reducer registry ──────────────────────────────────────────


class TestRSquaredTrend:
    """Exercise the bundled R² reducer directly so we lock its semantics."""

    def test_perfect_linear_trend_is_one(self) -> None:
        from bestee.stocks.decorators import _rsquared_trend

        # y = 2x + 3
        series = [3.0, 5.0, 7.0, 9.0, 11.0]
        assert _rsquared_trend(series) == pytest.approx(1.0)

    def test_constant_series_is_none(self) -> None:
        from bestee.stocks.decorators import _rsquared_trend

        assert _rsquared_trend([5.0, 5.0, 5.0]) is None

    def test_too_short_is_none(self) -> None:
        from bestee.stocks.decorators import _rsquared_trend

        assert _rsquared_trend([]) is None
        assert _rsquared_trend([1.0]) is None

    def test_noisy_series_under_one(self) -> None:
        from bestee.stocks.decorators import _rsquared_trend

        # A jittered linear series — high but not perfect R².
        series = [1.0, 5.0, 3.0, 8.0, 6.0, 10.0]
        r2 = _rsquared_trend(series)
        assert r2 is not None
        assert 0.0 < r2 < 1.0


class TestRegisterTimeSeriesMetric:
    def test_register_and_use_a_custom_reducer(self) -> None:
        """Registered reducers are immediately callable via the decorator."""
        from bestee.stocks.decorators import (
            _TIME_SERIES_METRICS,
            register_time_series_metric,
        )

        try:
            register_time_series_metric(
                "LastValue", lambda s: float(s[-1]) if s else None
            )

            upstream_df = pl.DataFrame({cols.TICKER: ["AAPL", "MSFT"]})
            ts = _ts_def()
            with patch(_GET_TS_PATCH) as mock_get:
                mock_get.side_effect = lambda ticker, _ts_def: {
                    "AAPL": [10.0, 20.0, 30.0],
                    "MSFT": [100.0, 200.0],
                }[ticker]
                df = TimeSeriesMetricDecorator(
                    ts_def=ts,
                    upstream_decorator=TimeSeriesCacheDecorator(
                        ts_def=ts, upstream_decorator=_StubUpstream(upstream_df)
                    ),
                    name="last",
                    metric="LastValue",
                )._build_df()

            assert df["last"].to_list() == [30.0, 200.0]
        finally:
            _TIME_SERIES_METRICS.pop("LastValue", None)


# ── TIME_SERIES / TIME_SERIES_METRIC DSL wiring ──────────────────────


class TestTimeSeriesDsl:
    def _commands(self, *lines: str) -> list[list[str]]:
        return [line.split() for line in lines]

    def test_builds_cache_then_metric(self) -> None:
        """STOCKS → TIME_SERIES → TIME_SERIES_METRIC produces the right chain."""
        with patch(
            "bestee.stocks.decorators.TickerSummaryDecorator._build_df"
        ) as _mock:
            pipeline = decorator_builder(
                self._commands(
                    "STOCKS",
                    "TIME_SERIES ts1 close_price 2025-01-02 2025-01-08 day 1",
                    "TIME_SERIES_METRIC tsm1 RSquared ts1",
                )
            )

        # Outermost is the metric decorator carrying the requested reducer
        # and the user-supplied column name.
        assert isinstance(pipeline, TimeSeriesMetricDecorator)
        assert pipeline._name == "tsm1"
        assert pipeline._metric_name == "RSquared"
        # Beneath it sits the cache (the producer in the chain).
        assert isinstance(pipeline._upstream, TimeSeriesCacheDecorator)
        assert pipeline._upstream._ts_def == pipeline._ts_def

    def test_metric_without_matching_series_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="undefined name"):
            decorator_builder(
                self._commands(
                    "STOCKS",
                    "TIME_SERIES_METRIC tsm1 RSquared tsMissing",
                )
            )

    def test_metric_wrong_arity_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="requires 3 arguments"):
            decorator_builder(
                self._commands(
                    "STOCKS",
                    "TIME_SERIES ts1 close_price 2025-01-02 2025-01-08 day 1",
                    # Missing the leading column name.
                    "TIME_SERIES_METRIC RSquared ts1",
                )
            )

    def test_unknown_ohlc_field_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="TIME_SERIES field"):
            decorator_builder(
                self._commands(
                    "STOCKS",
                    "TIME_SERIES ts1 NOT_A_FIELD 2025-01-02 2025-01-08 day 1",
                )
            )

    def test_unknown_span_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="TIME_SERIES span"):
            decorator_builder(
                self._commands(
                    "STOCKS",
                    "TIME_SERIES ts1 close_price 2025-01-02 2025-01-08 century 1",
                )
            )

    def test_multiplier_must_be_integer(self) -> None:
        with pytest.raises(ProcessingLevelError, match="multiplier"):
            decorator_builder(
                self._commands(
                    "STOCKS",
                    "TIME_SERIES ts1 close_price 2025-01-02 2025-01-08 day x",
                )
            )

    def test_two_metrics_can_share_one_series(self) -> None:
        """One TIME_SERIES declaration, two metric columns referencing it.

        Registers a temporary ``LastValue`` reducer so we can stack two
        different metrics on the same series; both end up pointing at
        the same :class:`TimeSeriesDef` via the named registry."""
        from bestee.stocks.decorators import (
            _TIME_SERIES_METRICS,
            register_time_series_metric,
        )

        try:
            register_time_series_metric(
                "LastValue", lambda s: float(s[-1]) if s else None
            )
            pipeline = decorator_builder(
                self._commands(
                    "STOCKS",
                    "TIME_SERIES ts1 close_price 2025-01-02 2025-01-08 day 1",
                    "TIME_SERIES_METRIC r2 RSquared ts1",
                    "TIME_SERIES_METRIC last LastValue ts1",
                )
            )
        finally:
            _TIME_SERIES_METRICS.pop("LastValue", None)

        # Outer column = last (added second); one stage down = r2.
        assert isinstance(pipeline, TimeSeriesMetricDecorator)
        assert pipeline._name == "last"
        assert pipeline._metric_name == "LastValue"
        inner = pipeline._upstream
        assert isinstance(inner, TimeSeriesMetricDecorator)
        assert inner._name == "r2"
        assert inner._metric_name == "RSquared"
        # And both point at the same TimeSeriesDef (referenced by name).
        assert pipeline._ts_def == inner._ts_def

    def test_unknown_metric_surfaces_as_processing_error(self) -> None:
        with pytest.raises(ProcessingLevelError, match="Unknown time-series metric"):
            decorator_builder(
                self._commands(
                    "STOCKS",
                    "TIME_SERIES ts1 close_price 2025-01-02 2025-01-08 day 1",
                    "TIME_SERIES_METRIC tsm1 NotAMetric ts1",
                )
            )

    def test_sample_input_lines_build_end_to_end(self) -> None:
        """Lock in support for the exact lines used in
        ``src/bestee/resources/sample_input.txt``: build the chain on
        top of a stub upstream, mock the network, and confirm the
        DataFrame carries a Float64 ``tsm1`` column whose values come
        from the R² reducer."""
        # These two strings should remain in sync with sample_input.txt.
        ts_line = "TIME_SERIES ts1 close_price 2025-01-01 2026-01-01 day 1"
        metric_line = "TIME_SERIES_METRIC tsm1 RSquared ts1"

        # Build chain by walking the level-3 processor directly with a
        # stub upstream — sidesteps the network calls a full STOCKS
        # stage would make.
        from bestee.stocks.decorators import command_processor_level_three

        upstream = _StubUpstream(
            pl.DataFrame({cols.TICKER: ["LINEAR", "CONST", "TOO_SHORT"]})
        )
        decorator: TableDecorator | None = upstream
        ts_defs: dict[str, TimeSeriesDef] = {}
        for raw in (ts_line, metric_line):
            status, decorator, ts_defs = command_processor_level_three(
                raw.split(), decorator, ts_defs
            )
            assert status == 1, f"level-3 processor refused command: {raw!r}"

        # Producer-then-consumer chain shape.
        assert isinstance(decorator, TimeSeriesMetricDecorator)
        assert decorator._name == "tsm1"
        assert decorator._metric_name == "RSquared"
        assert isinstance(decorator._upstream, TimeSeriesCacheDecorator)
        cache_ts = decorator._upstream._ts_def
        assert cache_ts.name == TimeSeriesName.CLOSE_PRICE
        assert cache_ts.span == TimeSeriesSpan.DAY
        assert cache_ts.start == "2025-01-01"
        assert cache_ts.end == "2026-01-01"
        assert cache_ts.multiplier == 1

        fake_series = {
            "LINEAR": [1.0, 2.0, 3.0, 4.0, 5.0],  # perfect trend → R²=1.0
            "CONST": [7.0, 7.0, 7.0],  # zero variance → None
            "TOO_SHORT": [],  # not enough points → None
        }
        with patch(
            _GET_TS_PATCH,
            side_effect=lambda t, _ts: fake_series[t],
        ) as mock_get:
            df = decorator._build_df()

        # Network is hit at most once per ticker (cache works).
        assert mock_get.call_count == 3
        assert df.columns == [cols.TICKER, "tsm1"]
        assert df.schema["tsm1"] == pl.Float64
        values = df["tsm1"].to_list()
        assert values[0] == pytest.approx(1.0)
        assert values[1] is None
        assert values[2] is None


# ── TimeSeriesDerivedDecorator integration ───────────────────────────


def _tagged_ts(
    tag: str,
    *,
    name: TimeSeriesName = TimeSeriesName.CLOSE_PRICE,
    span: TimeSeriesSpan = TimeSeriesSpan.DAY,
    start: str = "2025-01-02",
    end: str = "2025-01-08",
    multiplier: int = 1,
) -> TimeSeriesDef:
    """Build a TimeSeriesDef with the supplied tag (DSL name)."""
    return TimeSeriesDef(
        name=name,
        span=span,
        start=start,
        end=end,
        multiplier=multiplier,
        tag=tag,
    )


class TestTimeSeriesDerivedDecorator:
    def test_subtracts_two_series_elementwise(self) -> None:
        ts1 = _tagged_ts("ts1")
        ts2 = _tagged_ts("ts2")
        ts3 = _tagged_ts("ts3")

        upstream_df = pl.DataFrame({cols.TICKER: ["AAPL"]})
        cache_ts1 = TimeSeriesCacheDecorator(
            ts_def=ts1, upstream_decorator=_StubUpstream(upstream_df)
        )
        cache_ts2 = TimeSeriesCacheDecorator(ts_def=ts2, upstream_decorator=cache_ts1)
        derived = TimeSeriesDerivedDecorator(
            derived_ts_def=ts3,
            expression="ts2-ts1",
            operand_ts_defs={"ts1": ts1, "ts2": ts2},
            upstream_decorator=cache_ts2,
        )

        # Stub the network so each ts has a deterministic series.
        with patch(_GET_TS_PATCH) as mock_get:
            mock_get.side_effect = lambda _ticker, ts_def: {
                ts1: [10.0, 11.0, 12.0],
                ts2: [13.0, 14.0, 15.5],
            }[ts_def]
            result = derived.get_time_series("AAPL", ts3)

        assert result == [pytest.approx(3.0), pytest.approx(3.0), pytest.approx(3.5)]

    @pytest.mark.parametrize(
        "expression,operand_values,expected",
        [
            ("ts1+ts2", {"ts1": [1, 2, 3], "ts2": [10, 20, 30]}, [11, 22, 33]),
            ("ts2/ts1", {"ts1": [2, 4, 5], "ts2": [10, 20, 25]}, [5, 5, 5]),
            ("ts1*ts2", {"ts1": [1, 2, 3], "ts2": [4, 5, 6]}, [4, 10, 18]),
            ("2*ts1 - ts2", {"ts1": [5, 6], "ts2": [3, 4]}, [7, 8]),
            ("-ts1", {"ts1": [1, 2, 3]}, [-1, -2, -3]),
        ],
    )
    def test_supports_full_arithmetic(
        self,
        expression: str,
        operand_values: dict[str, list[float]],
        expected: list[float],
    ) -> None:
        operand_defs = {name: _tagged_ts(name) for name in operand_values}
        ts_out = _tagged_ts("ts_out")
        upstream_df = pl.DataFrame({cols.TICKER: ["AAPL"]})
        decorator: TableDecorator = _StubUpstream(upstream_df)
        for name, td in operand_defs.items():
            decorator = TimeSeriesCacheDecorator(td, upstream_decorator=decorator)
        derived = TimeSeriesDerivedDecorator(
            derived_ts_def=ts_out,
            expression=expression,
            operand_ts_defs=operand_defs,
            upstream_decorator=decorator,
        )
        with patch(_GET_TS_PATCH) as mock_get:
            mock_get.side_effect = lambda _t, td: [
                float(x)
                for x in operand_values[
                    next(n for n, d in operand_defs.items() if d == td)
                ]
            ]
            result = derived.get_time_series("AAPL", ts_out)
        assert result == [pytest.approx(v) for v in expected]

    def test_tag_distinguishes_derived_from_operands(self) -> None:
        """Even when shapes match, ts1 and ts3 don't compare equal."""
        ts1 = _tagged_ts("ts1")
        ts3 = _tagged_ts("ts3")
        # Same OHLC params + range → would be equal without tag.
        assert (ts1.name, ts1.span, ts1.start, ts1.end, ts1.multiplier) == (
            ts3.name,
            ts3.span,
            ts3.start,
            ts3.end,
            ts3.multiplier,
        )
        # But different tags break equality.
        assert ts1 != ts3

    def test_derived_caches_per_ticker(self) -> None:
        ts1 = _tagged_ts("ts1")
        ts3 = _tagged_ts("ts3")
        upstream_df = pl.DataFrame({cols.TICKER: ["AAPL"]})
        cache = TimeSeriesCacheDecorator(
            ts_def=ts1, upstream_decorator=_StubUpstream(upstream_df)
        )
        derived = TimeSeriesDerivedDecorator(
            derived_ts_def=ts3,
            expression="ts1",
            operand_ts_defs={"ts1": ts1},
            upstream_decorator=cache,
        )
        with patch(_GET_TS_PATCH) as mock_get:
            mock_get.return_value = [1.0, 2.0]
            derived.get_time_series("AAPL", ts3)
            derived.get_time_series("AAPL", ts3)
        assert mock_get.call_count == 1

    def test_undefined_operand_raises_at_construction(self) -> None:
        ts1 = _tagged_ts("ts1")
        with pytest.raises(ValueError, match="undefined names"):
            TimeSeriesDerivedDecorator(
                derived_ts_def=_tagged_ts("out"),
                expression="ts2 - ts1",
                operand_ts_defs={"ts1": ts1},
                upstream_decorator=_StubUpstream(pl.DataFrame({cols.TICKER: []})),
            )

    @pytest.mark.parametrize(
        "expression,fragment",
        [
            ("ts1 // ts2", "unsupported construct"),  # floor div not allowed
            ("ts1 ** 2", "unsupported construct"),  # power not allowed
            ("foo(ts1)", "unsupported construct"),  # call not allowed
            ("ts1 = 5", "valid Python syntax"),  # not a valid expression
        ],
    )
    def test_rejects_unsupported_expressions(
        self, expression: str, fragment: str
    ) -> None:
        with pytest.raises(ValueError, match=fragment):
            TimeSeriesDerivedDecorator(
                derived_ts_def=_tagged_ts("out"),
                expression=expression,
                operand_ts_defs={"ts1": _tagged_ts("ts1"), "ts2": _tagged_ts("ts2")},
                upstream_decorator=_StubUpstream(pl.DataFrame({cols.TICKER: []})),
            )


# ── TIME_SERIES_DERIVED DSL wiring ───────────────────────────────────


class TestTimeSeriesDerivedDsl:
    def _commands(self, *lines: str) -> list[list[str]]:
        return [line.split() for line in lines]

    def test_parses_subtraction_into_derived_chain(self) -> None:
        """The exact example from the prompt: TIME_SERIES_DERIVED ts3 ts2-ts1."""
        pipeline = decorator_builder(
            self._commands(
                "STOCKS",
                "TIME_SERIES ts1 close_price 2025-01-01 2026-01-01 day 1",
                "TIME_SERIES ts2 high_price 2025-01-01 2026-01-01 day 1",
                "TIME_SERIES_DERIVED ts3 ts2-ts1",
                "TIME_SERIES_METRIC tsm1 RSquared ts3",
            )
        )
        # Outer = metric for the derived series.
        assert isinstance(pipeline, TimeSeriesMetricDecorator)
        assert pipeline._name == "tsm1"
        assert pipeline._metric_name == "RSquared"
        # One stage down = the derived decorator with the right expression.
        derived = pipeline._upstream
        assert isinstance(derived, TimeSeriesDerivedDecorator)
        assert derived._expression == "ts2-ts1"
        assert sorted(derived._operand_ts_defs) == ["ts1", "ts2"]
        # And the derived ts_def carries the DSL name as its tag.
        assert pipeline._ts_def.tag == "ts3"

    def test_metric_reads_derived_series_end_to_end(self) -> None:
        """ts3 = ts2 - ts1, then RSquared(ts3): chain walks producer → cache."""
        from bestee.stocks.decorators import (
            _TIME_SERIES_METRICS,
            register_time_series_metric,
        )

        try:
            register_time_series_metric(
                "LastValue", lambda s: float(s[-1]) if s else None
            )
            # Build the same chain as the prompt, but on top of a stub.
            from bestee.stocks.decorators import command_processor_level_three

            upstream = _StubUpstream(pl.DataFrame({cols.TICKER: ["AAPL", "MSFT"]}))
            decorator: TableDecorator | None = upstream
            ts_defs: dict[str, TimeSeriesDef] = {}
            for raw in (
                "TIME_SERIES ts1 close_price 2025-01-01 2026-01-01 day 1",
                "TIME_SERIES ts2 high_price 2025-01-01 2026-01-01 day 1",
                "TIME_SERIES_DERIVED ts3 ts2-ts1",
                "TIME_SERIES_METRIC last LastValue ts3",
            ):
                _, decorator, ts_defs = command_processor_level_three(
                    raw.split(), decorator, ts_defs
                )

            ts1, ts2 = ts_defs["ts1"], ts_defs["ts2"]
            per_ticker = {
                "AAPL": {ts1: [10.0, 11.0, 12.0], ts2: [13.0, 15.0, 20.0]},
                "MSFT": {ts1: [100.0, 110.0], ts2: [105.0, 115.0]},
            }
            with patch(_GET_TS_PATCH) as mock_get:
                mock_get.side_effect = lambda ticker, td: per_ticker[ticker][td]
                assert decorator is not None
                df = decorator._build_df()

            # last = (ts2 - ts1)[-1]:
            # AAPL: 20 - 12 = 8;  MSFT: 115 - 110 = 5.
            assert df["last"].to_list() == [
                pytest.approx(8.0),
                pytest.approx(5.0),
            ]
        finally:
            _TIME_SERIES_METRICS.pop("LastValue", None)

    def test_duplicate_name_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="already defined"):
            decorator_builder(
                self._commands(
                    "STOCKS",
                    "TIME_SERIES ts1 close_price 2025-01-01 2026-01-01 day 1",
                    "TIME_SERIES_DERIVED ts1 ts1",
                )
            )

    def test_undefined_operand_raises_as_processing_error(self) -> None:
        with pytest.raises(ProcessingLevelError, match="undefined names"):
            decorator_builder(
                self._commands(
                    "STOCKS",
                    "TIME_SERIES_DERIVED ts3 ts2-ts1",
                )
            )

    def test_no_operands_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="must reference at least one"):
            decorator_builder(
                self._commands(
                    "STOCKS",
                    "TIME_SERIES_DERIVED ts3 42",
                )
            )

    def test_bad_expression_syntax_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="valid Python syntax"):
            decorator_builder(
                self._commands(
                    "STOCKS",
                    "TIME_SERIES ts1 close_price 2025-01-01 2026-01-01 day 1",
                    "TIME_SERIES_DERIVED ts3 ts1+",
                )
            )
