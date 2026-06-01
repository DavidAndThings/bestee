"""Tests for bestee.stocks.decorators — DSL parser, evaluator, and builder."""

import ast
import importlib.resources
from collections.abc import Sequence
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from great_tables import GT

import bestee.resources
from bestee.stocks import columns as cols
from bestee.stocks.builder import (
    ProcessingLevelError,
    _process_time_series,
    _process_time_series_derived,
    _process_time_series_metric,
    decorator_builder,
)
from bestee.stocks.decorators import (
    AppendTablesDecorator,
    AssetScopeDecorator,
    ComputedMetricDecorator,
    FinancialMetricCacheDecorator,
    FinancialsDecorator,
    NoUpstreamError,
    RelativeStrengthIndexDecorator,
    SameSICategoryDecorator,
    SimpleMovingAverageDecorator,
    StochasticOscillatorDecorator,
    TableDecorator,
    TimeSeriesCacheDecorator,
    TimeSeriesDerivedDecorator,
    TimeSeriesMetricDecorator,
    TimeSeriesNotAvailableError,
    relative_strength_index,
    simple_moving_average,
    stochastic_oscillator,
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


def _upstreams_of(decorator: TableDecorator) -> Sequence[TableDecorator]:
    """Assert that *decorator* has upstreams and return them.

    Tests that inspect the constructed chain pass through this so
    ty narrows ``_upstream`` from ``Sequence[…] | None`` to a real
    sequence.
    """
    assert decorator._upstream is not None
    return decorator._upstream


def _apply_ts_commands(
    stub: TableDecorator,
    raw_commands: list[str],
) -> tuple[TableDecorator, dict[str, TimeSeriesDef]]:
    """Walk a sequence of TIME_SERIES* command strings on top of *stub*.

    Returns the final chain head and the populated ``ts_defs`` mapping.
    Sidesteps the full ``decorator_builder`` so tests can start from a
    stub upstream instead of going through the real ``ASSET_SCOPE``
    path.
    """
    current: Sequence[TableDecorator] = [stub]
    ts_defs: dict[str, TimeSeriesDef] = {}
    for raw in raw_commands:
        cmd = raw.split()
        match cmd[0]:
            case "TIME_SERIES":
                current, ts_defs = _process_time_series(cmd, current, ts_defs)
            case "TIME_SERIES_DERIVED":
                current, ts_defs = _process_time_series_derived(cmd, current, ts_defs)
            case "TIME_SERIES_METRIC":
                current = _process_time_series_metric(cmd, current, ts_defs)
            case _:
                msg = f"_apply_ts_commands: unknown command {cmd[0]!r}"
                raise ValueError(msg)
    [head] = current
    return head, ts_defs


# ── ComputedMetricDecorator parsing / validation ─────────────────────


class TestComputedMetricParse:
    @staticmethod
    def _stub() -> _StubUpstream:
        return _StubUpstream(pl.DataFrame({cols.TICKER: []}))

    def test_collects_referenced_names(self) -> None:
        deco = ComputedMetricDecorator("cm1", "(fm2 * 2) / (fm3 + fm4)", self._stub())
        assert deco._referenced_names == ["fm2", "fm3", "fm4"]

    def test_chained_computed_metric_reference(self) -> None:
        """Bare identifiers can reference columns produced upstream."""
        deco = ComputedMetricDecorator("leverage", "roe / roa", self._stub())
        assert deco._referenced_names == ["roa", "roe"]

    def test_numeric_literals_and_unary_minus(self) -> None:
        deco = ComputedMetricDecorator("x", "-(rev * 2.5)", self._stub())
        assert deco._referenced_names == ["rev"]

    def test_syntactically_invalid_arithmetic_raises(self) -> None:
        with pytest.raises(ValueError, match="not valid Python syntax"):
            ComputedMetricDecorator("x", "rev +", self._stub())

    def test_unsupported_construct_raises(self) -> None:
        """Function calls and other non-arithmetic syntax should be rejected."""
        with pytest.raises(ValueError, match="unsupported construct"):
            ComputedMetricDecorator("x", "abs(rev)", self._stub())

    def test_non_numeric_constant_raises(self) -> None:
        with pytest.raises(ValueError, match="must be numeric"):
            ComputedMetricDecorator("x", "rev + 'oops'", self._stub())


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


_BASE = ["ASSET_SCOPE", "STOCKS"]


class TestDecoratorBuilder:
    def test_single_asset_scope_command(self) -> None:
        # A bare ASSET_SCOPE produces an AppendTablesDecorator wrapping
        # a single AssetScopeDecorator (master's "union of sources" shape).
        result = decorator_builder([_BASE])
        assert isinstance(result, AppendTablesDecorator)
        assert isinstance(_upstreams_of(result)[0], AssetScopeDecorator)

    def test_asset_scope_with_same_sic(self) -> None:
        result = decorator_builder([_BASE, ["SAME_SIC_CATEGORY_AS", "AAPL"]])
        # Subsetting wraps its filters in an AppendTablesDecorator.
        assert isinstance(result, AppendTablesDecorator)
        inner = _upstreams_of(result)[0]
        assert isinstance(inner, SameSICategoryDecorator)
        assert inner.ticker == "AAPL"

    def test_financial_metrics_share_one_decorator(self) -> None:
        """Two named FINANCIAL_METRIC lines share one FinancialsDecorator."""
        result = decorator_builder(
            [
                _BASE,
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
                    _BASE,
                    # Missing the leading name.
                    ["FINANCIAL_METRIC", "REVENUE", "2024", "4"],
                ]
            )

    def test_duplicate_financial_metric_name_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="already defined"):
            decorator_builder(
                [
                    _BASE,
                    ["FINANCIAL_METRIC", "rev", "REVENUE", "2024", "4"],
                    ["FINANCIAL_METRIC", "rev", "EBITDA", "2024", "4"],
                ]
            )

    def test_computed_metric_chains_on_top(self) -> None:
        result = decorator_builder(
            [
                _BASE,
                ["FINANCIAL_METRIC", "ni", "NET_INCOME", "2024", "4"],
                ["FINANCIAL_METRIC", "assets", "TOTAL_ASSETS", "2024", "4"],
                ["COMPUTED_METRIC", "roa", "ni", "/", "assets"],
            ]
        )
        assert isinstance(result, ComputedMetricDecorator)
        # Upstream is the FinancialsDecorator from the previous phase.
        [upstream_head] = _upstreams_of(result)
        assert isinstance(upstream_head, FinancialsDecorator)
        assert result._name == "roa"

    def test_two_computed_metrics_chain_in_input_order(self) -> None:
        result = decorator_builder(
            [
                _BASE,
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
        [inner] = _upstreams_of(result)
        assert isinstance(inner, ComputedMetricDecorator)
        assert inner._name == "roe"

    def test_ordering_does_not_matter_within_phases(self) -> None:
        """COMPUTED_METRIC before ASSET_SCOPE still works — phases are reordered."""
        result = decorator_builder(
            [
                ["COMPUTED_METRIC", "doubled", "rev", "*", "2"],
                ["FINANCIAL_METRIC", "rev", "REVENUE", "2024", "4"],
                _BASE,
            ]
        )
        assert isinstance(result, ComputedMetricDecorator)
        [inner] = _upstreams_of(result)
        assert isinstance(inner, FinancialsDecorator)

    def test_unknown_command_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="Unrecognized"):
            decorator_builder([_BASE, ["NOT_A_COMMAND"]])

    def test_same_sic_without_upstream_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="ASSET_SCOPE"):
            decorator_builder([["SAME_SIC_CATEGORY_AS", "AAPL"]])

    def test_financial_metric_without_upstream_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="ASSET_SCOPE"):
            decorator_builder([["FINANCIAL_METRIC", "rev", "REVENUE", "2024", "4"]])

    def test_computed_metric_without_upstream_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="ASSET_SCOPE"):
            decorator_builder([["COMPUTED_METRIC", "x", "rev", "*", "2"]])

    def test_computed_metric_wrong_arity_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="at least 2 arguments"):
            decorator_builder(
                [
                    _BASE,
                    ["COMPUTED_METRIC", "x"],
                ]
            )

    def test_empty_command_list_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="at least one ASSET_SCOPE"):
            decorator_builder([])

    def test_unknown_financial_metric_name_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="Unknown FINANCIAL_METRIC"):
            decorator_builder(
                [
                    _BASE,
                    ["FINANCIAL_METRIC", "x", "NOT_A_METRIC", "2024", "4"],
                ]
            )

    def test_non_integer_fiscal_period_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="must be integers"):
            decorator_builder(
                [
                    _BASE,
                    ["FINANCIAL_METRIC", "x", "REVENUE", "twenty24", "4"],
                ]
            )


# ── TableDecorator base / NoUpstreamError ────────────────────────────


class TestNoUpstreamError:
    def test_build_from_upstream_raises_when_none(self) -> None:
        class _Stage(TableDecorator):
            def _build_df(self) -> pl.DataFrame:
                return self._build_from_upstream_or_error()

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
        result_df = SameSICategoryDecorator("AAPL", upstream)._build_df()
        assert sorted(result_df[cols.TICKER].to_list()) == ["AAPL", "GOOG", "MSFT"]

    def test_missing_ticker_raises(self) -> None:
        df = pl.DataFrame({cols.TICKER: ["AAPL"], cols.SIC_CODE: ["3571"]})
        upstream = _StubUpstream(df)
        with pytest.raises(ValueError, match="not found in upstream"):
            SameSICategoryDecorator("MSFT", upstream).build()

    def test_null_sic_raises(self) -> None:
        df = pl.DataFrame(
            {cols.TICKER: ["AAPL", "MSFT"], cols.SIC_CODE: [None, "3571"]},
            schema={cols.TICKER: pl.Utf8, cols.SIC_CODE: pl.Utf8},
        )
        upstream = _StubUpstream(df)
        with pytest.raises(ValueError, match="no SIC code"):
            SameSICategoryDecorator("AAPL", upstream).build()


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
        deco = ComputedMetricDecorator("roa", "ni / assets", _StubUpstream(upstream_df))
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
            "cm1", "(fm2 * 2) / (fm3 + fm4)", _StubUpstream(upstream_df)
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
        deco = ComputedMetricDecorator("roa", "ni / assets", _StubUpstream(upstream_df))
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
        deco = ComputedMetricDecorator("roa", "ni / assets", _StubUpstream(upstream_df))
        result_df = deco._build_df()
        assert result_df["roa"].to_list() == [None]

    def test_chained_computed_metric_reads_upstream_column(self) -> None:
        """A second ComputedMetricDecorator can reference the first's column."""
        upstream_df = pl.DataFrame({cols.TICKER: ["AAPL", "MSFT"], "roa": [0.25, 0.10]})
        deco = ComputedMetricDecorator(
            "doubled_roa", "roa * 2", _StubUpstream(upstream_df)
        )
        result_df = deco._build_df()
        assert result_df["doubled_roa"].to_list() == [0.5, 0.2]

    def test_missing_upstream_reference_raises(self) -> None:
        upstream_df = pl.DataFrame({cols.TICKER: ["AAPL"]})
        deco = ComputedMetricDecorator("x", "missing * 2", _StubUpstream(upstream_df))
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

        # New construction: pass table upstream + cache markers as siblings.
        stub = _StubUpstream(upstream_df)
        deco = FinancialsDecorator(
            stub,
            FinancialMetricCacheDecorator("fm_ni", ni),
            FinancialMetricCacheDecorator("fm_ta", ta),
        )
        df = deco._build_df()

        # Columns carry the user-supplied names, not the SDK labels.
        assert "fm_ni" in df.columns
        assert "fm_ta" in df.columns
        assert ni.label not in df.columns
        assert df["fm_ni"].to_list() == [100.0, 200.0]
        assert df["fm_ta"].to_list() == [400.0, 1000.0]

    def test_no_caches_passes_upstream_through(self) -> None:
        """With no FinancialMetricCacheDecorator children, no API call should happen."""
        upstream_df = pl.DataFrame({cols.TICKER: ["AAPL"], "Existing": [42.0]})
        deco = FinancialsDecorator(_StubUpstream(upstream_df))
        df = deco._build_df()
        assert df.columns == [cols.TICKER, "Existing"]


# ── Stress-input regression tests ────────────────────────────────────


_STRESS_INPUTS = sorted(
    p.name
    for p in importlib.resources.files(bestee.resources).iterdir()
    if p.name.endswith(".txt")
)


def _read_stress(filename: str) -> list[list[str]]:
    text = (
        importlib.resources.files(bestee.resources)
        .joinpath(filename)
        .read_text(encoding="utf-8")
    )
    return [line.split() for line in text.splitlines() if line.strip()]


def _chain_class_names(decorator: TableDecorator) -> list[str]:
    """Return every *unique* decorator class name reachable from *decorator*.

    The pipeline is a DAG, not a tree — when subsetting builds two
    parallel filters they share the same upstream sub-chain.  We dedupe
    by object identity so the counts reflect distinct instances rather
    than walk visits.
    """
    classes: list[str] = []
    seen_ids: set[int] = set()

    def walk(d: TableDecorator) -> None:
        if id(d) in seen_ids:
            return
        seen_ids.add(id(d))
        classes.append(type(d).__name__)
        for u in d._upstream or []:
            walk(u)

    walk(decorator)
    return classes


def _count_keyword(commands: list[list[str]], keyword: str) -> int:
    return sum(1 for c in commands if c and c[0] == keyword)


@pytest.mark.parametrize("filename", _STRESS_INPUTS)
def test_stress_input_parses_and_builds(filename: str) -> None:
    """Each bundled DSL script parses and assembles into a decorator chain."""
    commands = _read_stress(filename)
    assert commands, f"{filename} has no commands"

    pipeline = decorator_builder(commands)

    # Every stress input ends with at least one transformation, so the
    # output is never just the bare AssetScopeDecorator.
    assert isinstance(pipeline, TableDecorator)


@pytest.mark.parametrize("filename", _STRESS_INPUTS)
def test_stress_input_chain_shape_matches_dsl(filename: str) -> None:
    """The composition of the chain reflects the DSL input exactly.

    For each non-FINANCIAL_METRIC keyword the count of corresponding
    decorators in the chain equals the count of commands.  FINANCIAL_METRIC
    is special: many commands collapse into one :class:`FinancialsDecorator`
    holding the per-metric :class:`FinancialMetricCacheDecorator` markers.
    """
    commands = _read_stress(filename)
    pipeline = decorator_builder(commands)
    chain = _chain_class_names(pipeline)

    expected_one_to_one = {
        "AssetScopeDecorator": "ASSET_SCOPE",
        "SameSICategoryDecorator": "SAME_SIC_CATEGORY_AS",
        "PickTickersDecorator": "PICK_TICKERS",
        "ComputedMetricDecorator": "COMPUTED_METRIC",
        "TimeSeriesCacheDecorator": "TIME_SERIES",
        "TimeSeriesDerivedDecorator": "TIME_SERIES_DERIVED",
        "TimeSeriesMetricDecorator": "TIME_SERIES_METRIC",
    }
    for cls_name, keyword in expected_one_to_one.items():
        expected = _count_keyword(commands, keyword)
        actual = chain.count(cls_name)
        assert actual == expected, (
            f"{filename}: expected {expected} {cls_name} (one per "
            f"{keyword}), got {actual} in chain {chain}"
        )

    # Financials phase batches all metrics into one decorator (if any).
    n_fin = _count_keyword(commands, "FINANCIAL_METRIC")
    assert chain.count("FinancialsDecorator") == (1 if n_fin else 0)
    assert chain.count("FinancialMetricCacheDecorator") == n_fin


# ── End-to-end stress: build a small pipeline and run it with mocks ──


def test_full_pipeline_executes_end_to_end_with_mocks() -> None:
    """Build and execute a pipeline that touches every phase.

    Uses an in-line DSL (not one of the on-disk fixtures) so the test
    stays deterministic — every mocked leaf returns predictable rows
    and the chain's filtering / joining / column-adding semantics can
    be verified row-by-row.  This exercises every decorator class in
    concert, well beyond the smoke-level ``isinstance(TableDecorator)``
    check.
    """
    commands = [
        ["ASSET_SCOPE", "STOCKS"],
        ["SAME_SIC_CATEGORY_AS", "AAPL"],
        ["FINANCIAL_METRIC", "rev", "REVENUE", "2024", "4"],
        ["FINANCIAL_METRIC", "ni", "NET_INCOME", "2024", "4"],
        ["COMPUTED_METRIC", "margin", "ni", "/", "rev"],
        ["TIME_SERIES", "ts1", "close_price", "2024-01-01", "2024-06-01", "day", "1"],
        ["TIME_SERIES_METRIC", "r2", "RSquared", "ts1"],
    ]
    pipeline = decorator_builder(commands)

    # Synthetic base table: SAME_SIC AAPL keeps AAPL+MSFT (SIC 3571),
    # drops F (SIC 3711).
    base_df = pl.DataFrame(
        {
            cols.TICKER: ["AAPL", "MSFT", "F"],
            cols.SIC_CODE: ["3571", "3571", "3711"],
        }
    )

    rev_metric = FinancialMetric(Metric.REVENUE, 2024, 4)
    ni_metric = FinancialMetric(Metric.NET_INCOME, 2024, 4)
    fin_df = pl.DataFrame(
        {
            cols.TICKER: ["AAPL", "MSFT"],
            rev_metric.label: [400.0, 200.0],
            ni_metric.label: [80.0, 40.0],
        }
    )

    # Perfectly linear series → R² = 1.0 for each ticker.
    ts_series = {"AAPL": [1.0, 2.0, 3.0], "MSFT": [10.0, 20.0, 30.0]}

    with (
        patch(
            "bestee.stocks.decorators.AssetScopeDecorator._build_df",
            return_value=base_df,
        ),
        patch(_FINANCIALS_PATCH, return_value=fin_df),
        patch(_GET_TS_PATCH, side_effect=lambda t, _td: ts_series[t]),
    ):
        result = pipeline._build_df()

    # SAME_SIC filters out F; AAPL + MSFT remain.
    assert sorted(result[cols.TICKER].unique().to_list()) == ["AAPL", "MSFT"]

    # Every phase added the column it was supposed to.
    assert {"rev", "ni", "margin", "r2"} <= set(result.columns)

    by_ticker = {row[cols.TICKER]: row for row in result.iter_rows(named=True)}

    # Financial metrics get user-supplied names, not the SDK's labels.
    assert by_ticker["AAPL"]["rev"] == pytest.approx(400.0)
    assert by_ticker["MSFT"]["rev"] == pytest.approx(200.0)

    # Computed metric pulls from the renamed columns.
    assert by_ticker["AAPL"]["margin"] == pytest.approx(80.0 / 400.0)
    assert by_ticker["MSFT"]["margin"] == pytest.approx(40.0 / 200.0)

    # Time-series metric: linear series → perfect trend.
    assert by_ticker["AAPL"]["r2"] == pytest.approx(1.0)
    assert by_ticker["MSFT"]["r2"] == pytest.approx(1.0)


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
            ts,
            "tsm1",
            "RSquared",
            TimeSeriesCacheDecorator(ts, _StubUpstream(upstream_df)),
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
        cache = TimeSeriesCacheDecorator(ts, _StubUpstream(upstream_df))
        TimeSeriesMetricDecorator(ts, "a", "RSquared", cache)._build_df()
        TimeSeriesMetricDecorator(ts, "b", "RSquared", cache)._build_df()

        assert mock_get.call_count == 1

    def test_no_producer_raises_not_available(self) -> None:
        """Without a cache (or other producer) upstream, the chain walk
        runs out of stages and surfaces TimeSeriesNotAvailableError."""
        upstream_df = pl.DataFrame({cols.TICKER: ["AAPL"]})
        deco = TimeSeriesMetricDecorator(
            _ts_def(),
            "tsm1",
            "RSquared",
            _StubUpstream(upstream_df),
        )
        with pytest.raises(TimeSeriesNotAvailableError):
            deco._build_df()

    @patch(_GET_TS_PATCH)
    def test_build_returns_gt_with_metric_column(self, mock_get: MagicMock) -> None:
        mock_get.return_value = [1.0, 2.0, 3.0]
        upstream_df = pl.DataFrame({cols.TICKER: ["AAPL"]})
        ts = _ts_def()
        deco = TimeSeriesMetricDecorator(
            ts,
            "tsm1",
            "RSquared",
            TimeSeriesCacheDecorator(ts, _StubUpstream(upstream_df)),
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
                _ts_def(),
                "tsm1",
                "NoSuchMetric",
                _StubUpstream(pl.DataFrame({cols.TICKER: []})),
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
                    ts,
                    "last",
                    "LastValue",
                    TimeSeriesCacheDecorator(ts, _StubUpstream(upstream_df)),
                )._build_df()

            assert df["last"].to_list() == [30.0, 200.0]
        finally:
            _TIME_SERIES_METRICS.pop("LastValue", None)


# ── TIME_SERIES / TIME_SERIES_METRIC DSL wiring ──────────────────────


class TestTimeSeriesDsl:
    def _commands(self, *lines: str) -> list[list[str]]:
        return [line.split() for line in lines]

    def test_builds_cache_then_metric(self) -> None:
        """ASSET_SCOPE → TIME_SERIES → TIME_SERIES_METRIC produces the right chain."""
        with patch("bestee.stocks.decorators.AssetScopeDecorator._build_df") as _mock:
            pipeline = decorator_builder(
                self._commands(
                    "ASSET_SCOPE STOCKS",
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
        [inner] = _upstreams_of(pipeline)
        assert isinstance(inner, TimeSeriesCacheDecorator)
        assert inner._ts_def == pipeline._ts_def

    def test_metric_without_matching_series_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="undefined name"):
            decorator_builder(
                self._commands(
                    "ASSET_SCOPE STOCKS",
                    "TIME_SERIES_METRIC tsm1 RSquared tsMissing",
                )
            )

    def test_metric_wrong_arity_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="requires 3 arguments"):
            decorator_builder(
                self._commands(
                    "ASSET_SCOPE STOCKS",
                    "TIME_SERIES ts1 close_price 2025-01-02 2025-01-08 day 1",
                    # Missing the leading column name.
                    "TIME_SERIES_METRIC RSquared ts1",
                )
            )

    def test_unknown_ohlc_field_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="TIME_SERIES field"):
            decorator_builder(
                self._commands(
                    "ASSET_SCOPE STOCKS",
                    "TIME_SERIES ts1 NOT_A_FIELD 2025-01-02 2025-01-08 day 1",
                )
            )

    def test_unknown_span_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="TIME_SERIES span"):
            decorator_builder(
                self._commands(
                    "ASSET_SCOPE STOCKS",
                    "TIME_SERIES ts1 close_price 2025-01-02 2025-01-08 century 1",
                )
            )

    def test_multiplier_must_be_integer(self) -> None:
        with pytest.raises(ProcessingLevelError, match="multiplier"):
            decorator_builder(
                self._commands(
                    "ASSET_SCOPE STOCKS",
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
                    "ASSET_SCOPE STOCKS",
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
        [inner] = _upstreams_of(pipeline)
        assert isinstance(inner, TimeSeriesMetricDecorator)
        assert inner._name == "r2"
        assert inner._metric_name == "RSquared"
        # And both point at the same TimeSeriesDef (referenced by name).
        assert pipeline._ts_def == inner._ts_def

    def test_unknown_metric_surfaces_as_processing_error(self) -> None:
        with pytest.raises(ProcessingLevelError, match="Unknown time-series metric"):
            decorator_builder(
                self._commands(
                    "ASSET_SCOPE STOCKS",
                    "TIME_SERIES ts1 close_price 2025-01-02 2025-01-08 day 1",
                    "TIME_SERIES_METRIC tsm1 NotAMetric ts1",
                )
            )

    def test_sample_input_lines_build_end_to_end(self) -> None:
        """Lock in support for the exact lines used in
        ``src/bestee/resources/sample_input.txt``: build the chain on
        top of a stub upstream (sidestepping ``ASSET_SCOPE``), mock
        the network, and confirm the DataFrame carries a Float64
        ``tsm1`` column whose values come from the R² reducer."""
        # These two strings should remain in sync with sample_input.txt.
        upstream = _StubUpstream(
            pl.DataFrame({cols.TICKER: ["LINEAR", "CONST", "TOO_SHORT"]})
        )
        decorator, ts_defs = _apply_ts_commands(
            upstream,
            [
                "TIME_SERIES ts1 close_price 2025-01-01 2026-01-01 day 1",
                "TIME_SERIES_METRIC tsm1 RSquared ts1",
            ],
        )

        # Producer-then-consumer chain shape.
        assert isinstance(decorator, TimeSeriesMetricDecorator)
        assert decorator._name == "tsm1"
        assert decorator._metric_name == "RSquared"
        [cache] = _upstreams_of(decorator)
        assert isinstance(cache, TimeSeriesCacheDecorator)
        cache_ts = cache._ts_def
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
        cache_ts1 = TimeSeriesCacheDecorator(ts1, _StubUpstream(upstream_df))
        cache_ts2 = TimeSeriesCacheDecorator(ts2, cache_ts1)
        derived = TimeSeriesDerivedDecorator(
            ts3, "ts2-ts1", {"ts1": ts1, "ts2": ts2}, cache_ts2
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
        for _, td in operand_defs.items():
            decorator = TimeSeriesCacheDecorator(td, decorator)
        derived = TimeSeriesDerivedDecorator(
            ts_out, expression, operand_defs, decorator
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
        cache = TimeSeriesCacheDecorator(ts1, _StubUpstream(upstream_df))
        derived = TimeSeriesDerivedDecorator(ts3, "ts1", {"ts1": ts1}, cache)
        with patch(_GET_TS_PATCH) as mock_get:
            mock_get.return_value = [1.0, 2.0]
            derived.get_time_series("AAPL", ts3)
            derived.get_time_series("AAPL", ts3)
        assert mock_get.call_count == 1

    def test_undefined_operand_raises_at_construction(self) -> None:
        ts1 = _tagged_ts("ts1")
        with pytest.raises(ValueError, match="undefined names"):
            TimeSeriesDerivedDecorator(
                _tagged_ts("out"),
                "ts2 - ts1",
                {"ts1": ts1},
                _StubUpstream(pl.DataFrame({cols.TICKER: []})),
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
                _tagged_ts("out"),
                expression,
                {"ts1": _tagged_ts("ts1"), "ts2": _tagged_ts("ts2")},
                _StubUpstream(pl.DataFrame({cols.TICKER: []})),
            )


# ── TIME_SERIES_DERIVED DSL wiring ───────────────────────────────────


class TestTimeSeriesDerivedDsl:
    def _commands(self, *lines: str) -> list[list[str]]:
        return [line.split() for line in lines]

    def test_parses_subtraction_into_derived_chain(self) -> None:
        """The exact example from the prompt: TIME_SERIES_DERIVED ts3 ts2-ts1."""
        pipeline = decorator_builder(
            self._commands(
                "ASSET_SCOPE STOCKS",
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
        [derived] = _upstreams_of(pipeline)
        assert isinstance(derived, TimeSeriesDerivedDecorator)
        assert derived._expression == "ts2-ts1"
        assert sorted(derived._operand_ts_defs) == ["ts1", "ts2"]
        # And the derived ts_def carries the DSL name as its tag.
        assert pipeline._ts_def.tag == "ts3"

    def test_metric_reads_derived_series_end_to_end(self) -> None:
        """ts3 = ts2 - ts1, then last(ts3): chain walks producer → cache."""
        from bestee.stocks.decorators import (
            _TIME_SERIES_METRICS,
            register_time_series_metric,
        )

        try:
            register_time_series_metric(
                "LastValue", lambda s: float(s[-1]) if s else None
            )
            upstream = _StubUpstream(pl.DataFrame({cols.TICKER: ["AAPL", "MSFT"]}))
            decorator, ts_defs = _apply_ts_commands(
                upstream,
                [
                    "TIME_SERIES ts1 close_price 2025-01-01 2026-01-01 day 1",
                    "TIME_SERIES ts2 high_price 2025-01-01 2026-01-01 day 1",
                    "TIME_SERIES_DERIVED ts3 ts2-ts1",
                    "TIME_SERIES_METRIC last LastValue ts3",
                ],
            )

            ts1, ts2 = ts_defs["ts1"], ts_defs["ts2"]
            per_ticker = {
                "AAPL": {ts1: [10.0, 11.0, 12.0], ts2: [13.0, 15.0, 20.0]},
                "MSFT": {ts1: [100.0, 110.0], ts2: [105.0, 115.0]},
            }
            with patch(_GET_TS_PATCH) as mock_get:
                mock_get.side_effect = lambda ticker, td: per_ticker[ticker][td]
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
                    "ASSET_SCOPE STOCKS",
                    "TIME_SERIES ts1 close_price 2025-01-01 2026-01-01 day 1",
                    "TIME_SERIES_DERIVED ts1 ts1",
                )
            )

    def test_undefined_operand_raises_as_processing_error(self) -> None:
        with pytest.raises(ProcessingLevelError, match="undefined names"):
            decorator_builder(
                self._commands(
                    "ASSET_SCOPE STOCKS",
                    "TIME_SERIES_DERIVED ts3 ts2-ts1",
                )
            )

    def test_no_operands_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="must reference at least one"):
            decorator_builder(
                self._commands(
                    "ASSET_SCOPE STOCKS",
                    "TIME_SERIES_DERIVED ts3 42",
                )
            )

    def test_bad_expression_syntax_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="valid Python syntax"):
            decorator_builder(
                self._commands(
                    "ASSET_SCOPE STOCKS",
                    "TIME_SERIES ts1 close_price 2025-01-01 2026-01-01 day 1",
                    "TIME_SERIES_DERIVED ts3 ts1+",
                )
            )


# ── stochastic_oscillator math ───────────────────────────────────────


class TestStochasticOscillatorMath:
    def test_steady_uptrend_pegs_k_at_100(self) -> None:
        """A perfectly rising series → latest close == window high → %K = 100."""
        highs = [10.0, 11.0, 12.0, 13.0, 14.0]
        lows = [9.0, 10.0, 11.0, 12.0, 13.0]
        closes = [10.0, 11.0, 12.0, 13.0, 14.0]
        k, d = stochastic_oscillator(highs, lows, closes, k_period=3, d_period=2)
        assert len(k) == 5
        assert k[0] is None and k[1] is None  # warmup
        assert k[2] == pytest.approx(100.0)
        assert k[3] == pytest.approx(100.0)
        assert k[4] == pytest.approx(100.0)
        # %D = 2-period SMA of %K, with extra 1-bar warmup.
        assert d[0] is None and d[1] is None and d[2] is None
        assert d[3] == pytest.approx(100.0)
        assert d[4] == pytest.approx(100.0)

    def test_steady_downtrend_pegs_k_at_zero(self) -> None:
        highs = [14.0, 13.0, 12.0, 11.0, 10.0]
        lows = [13.0, 12.0, 11.0, 10.0, 9.0]
        closes = [13.0, 12.0, 11.0, 10.0, 9.0]
        k, d = stochastic_oscillator(highs, lows, closes, k_period=3, d_period=2)
        assert k[2] == pytest.approx(0.0)
        assert k[4] == pytest.approx(0.0)
        assert d[4] == pytest.approx(0.0)

    def test_mid_range_close_yields_50_percent(self) -> None:
        # Highs/lows define a constant 10-wide window; close sits at mid.
        highs = [110.0, 110.0, 110.0]
        lows = [100.0, 100.0, 100.0]
        closes = [105.0, 105.0, 105.0]
        k, _ = stochastic_oscillator(highs, lows, closes, k_period=3, d_period=1)
        assert k[2] == pytest.approx(50.0)

    def test_zero_range_window_returns_none(self) -> None:
        """All-equal highs and lows → range 0 → %K undefined."""
        highs = [5.0, 5.0, 5.0]
        lows = [5.0, 5.0, 5.0]
        closes = [5.0, 5.0, 5.0]
        k, _ = stochastic_oscillator(highs, lows, closes, k_period=3, d_period=1)
        assert k[2] is None

    def test_d_propagates_none_from_k(self) -> None:
        """If any %K in the d_period window is None, %D is None."""
        highs = [110.0, 110.0, 110.0, 110.0]
        lows = [100.0, 100.0, 100.0, 100.0]
        # Make bar 2 land in a zero-range slice (force None into %K at i=2).
        # Workaround: use a too-short period so warmup overlaps.
        closes = [105.0, 105.0, 105.0, 105.0]
        k, d = stochastic_oscillator(highs, lows, closes, k_period=3, d_period=3)
        # k[0], k[1] are warmup None; k[2..3] are 50.
        # d[0..3] need 3 non-None k values; only k[2] and k[3] non-None
        # so far → d[3] None (window has k[1]=None, k[2]=50, k[3]=50).
        assert d[3] is None
        # Extend by one more bar to give d enough non-None inputs.
        k2, d2 = stochastic_oscillator(
            highs + [110.0], lows + [100.0], closes + [105.0], k_period=3, d_period=3
        )
        assert d2[4] == pytest.approx(50.0)

    def test_empty_input_returns_empty(self) -> None:
        k, d = stochastic_oscillator([], [], [], k_period=14, d_period=3)
        assert k == [] and d == []

    def test_mismatched_lengths_truncate_to_shortest(self) -> None:
        """Three input series of unequal length all get truncated to min()."""
        highs = [10.0, 11.0, 12.0, 13.0]
        lows = [9.0, 10.0, 11.0]
        closes = [10.0, 11.0]
        k, _ = stochastic_oscillator(highs, lows, closes, k_period=1, d_period=1)
        assert len(k) == 2  # = min(4, 3, 2)

    @pytest.mark.parametrize("k,d", [(0, 3), (3, 0), (-1, 3)])
    def test_invalid_periods_raise(self, k: int, d: int) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            stochastic_oscillator([1.0], [1.0], [1.0], k_period=k, d_period=d)


# ── StochasticOscillatorDecorator integration ───────────────────────


def _ts(field: TimeSeriesName, tag: str) -> TimeSeriesDef:
    """Tagged TimeSeriesDef helper for the stochastic tests."""
    return TimeSeriesDef(
        name=field,
        span=TimeSeriesSpan.DAY,
        start="2024-01-01",
        end="2024-04-01",
        multiplier=1,
        tag=tag,
    )


class TestStochasticOscillatorDecorator:
    def test_adds_k_and_d_columns(self) -> None:
        """Each ticker gets a latest %K and %D, suffixed by name."""
        ts_high = _ts(TimeSeriesName.HIGH_PRICE, "ts_h")
        ts_low = _ts(TimeSeriesName.LOW_PRICE, "ts_l")
        ts_close = _ts(TimeSeriesName.CLOSE_PRICE, "ts_c")

        # Stack the three caches the same way the DSL does: each wraps
        # the previous one, so the indicator has a single upstream.
        upstream_df = pl.DataFrame({cols.TICKER: ["AAPL"]})
        cache_h = TimeSeriesCacheDecorator(ts_high, _StubUpstream(upstream_df))
        cache_l = TimeSeriesCacheDecorator(ts_low, cache_h)
        cache_c = TimeSeriesCacheDecorator(ts_close, cache_l)
        chain = StochasticOscillatorDecorator(
            "stoch",
            ts_high,
            ts_low,
            ts_close,
            3,  # k_period
            2,  # d_period
            cache_c,
        )

        # Five-bar rising series → latest %K = 100, latest %D = 100.
        per_def = {
            ts_high: [10.0, 11.0, 12.0, 13.0, 14.0],
            ts_low: [9.0, 10.0, 11.0, 12.0, 13.0],
            ts_close: [10.0, 11.0, 12.0, 13.0, 14.0],
        }
        with patch(_GET_TS_PATCH, side_effect=lambda _t, td: per_def[td]):
            df = chain._build_df()

        assert {"stoch_k", "stoch_d"} <= set(df.columns)
        assert df["stoch_k"][0] == pytest.approx(100.0)
        assert df["stoch_d"][0] == pytest.approx(100.0)

    def test_invalid_periods_raise(self) -> None:
        ts = _ts(TimeSeriesName.CLOSE_PRICE, "ts")
        stub = _StubUpstream(pl.DataFrame({cols.TICKER: []}))
        with pytest.raises(ValueError, match=">= 1"):
            StochasticOscillatorDecorator("stoch", ts, ts, ts, 0, 3, stub)

    def test_missing_series_yields_none(self) -> None:
        """If a chain has no producer for one of the ts_defs, the
        decorator catches TimeSeriesNotAvailableError and emits None."""
        ts_high = _ts(TimeSeriesName.HIGH_PRICE, "ts_h")
        ts_low = _ts(TimeSeriesName.LOW_PRICE, "ts_l")
        ts_close = _ts(TimeSeriesName.CLOSE_PRICE, "ts_c")
        upstream_df = pl.DataFrame({cols.TICKER: ["AAPL"]})
        chain = StochasticOscillatorDecorator(
            "stoch",
            ts_high,
            ts_low,
            ts_close,
            14,
            3,
            _StubUpstream(upstream_df),
        )
        df = chain._build_df()
        assert df["stoch_k"][0] is None
        assert df["stoch_d"][0] is None


# ── STOCHASTIC_OSCILLATOR DSL wiring ────────────────────────────────


class TestStochasticOscillatorDsl:
    def _commands(self, *lines: str) -> list[list[str]]:
        return [line.split() for line in lines]

    def test_builds_chain_with_three_caches_and_oscillator(self) -> None:
        pipeline = decorator_builder(
            self._commands(
                "ASSET_SCOPE STOCKS",
                "TIME_SERIES ts_h high_price  2024-01-01 2024-04-01 day 1",
                "TIME_SERIES ts_l low_price   2024-01-01 2024-04-01 day 1",
                "TIME_SERIES ts_c close_price 2024-01-01 2024-04-01 day 1",
                "STOCHASTIC_OSCILLATOR stoch ts_h ts_l ts_c 14 3",
            )
        )
        assert isinstance(pipeline, StochasticOscillatorDecorator)
        assert pipeline._name == "stoch"
        assert pipeline._k_period == 14
        assert pipeline._d_period == 3

    def test_wrong_arity_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="requires 6 arguments"):
            decorator_builder(
                self._commands(
                    "ASSET_SCOPE STOCKS",
                    "TIME_SERIES ts_h high_price 2024-01-01 2024-04-01 day 1",
                    "STOCHASTIC_OSCILLATOR stoch ts_h",  # too few
                )
            )

    def test_undefined_ts_reference_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="undefined name"):
            decorator_builder(
                self._commands(
                    "ASSET_SCOPE STOCKS",
                    "TIME_SERIES ts_h high_price 2024-01-01 2024-04-01 day 1",
                    "STOCHASTIC_OSCILLATOR stoch ts_h tsMissing ts_h 14 3",
                )
            )

    def test_non_integer_periods_raise(self) -> None:
        with pytest.raises(ProcessingLevelError, match="must be integers"):
            decorator_builder(
                self._commands(
                    "ASSET_SCOPE STOCKS",
                    "TIME_SERIES ts_h high_price  2024-01-01 2024-04-01 day 1",
                    "TIME_SERIES ts_l low_price   2024-01-01 2024-04-01 day 1",
                    "TIME_SERIES ts_c close_price 2024-01-01 2024-04-01 day 1",
                    "STOCHASTIC_OSCILLATOR stoch ts_h ts_l ts_c fourteen 3",
                )
            )

    def test_invalid_periods_surface_as_processing_error(self) -> None:
        with pytest.raises(ProcessingLevelError, match=">= 1"):
            decorator_builder(
                self._commands(
                    "ASSET_SCOPE STOCKS",
                    "TIME_SERIES ts_h high_price  2024-01-01 2024-04-01 day 1",
                    "TIME_SERIES ts_l low_price   2024-01-01 2024-04-01 day 1",
                    "TIME_SERIES ts_c close_price 2024-01-01 2024-04-01 day 1",
                    "STOCHASTIC_OSCILLATOR stoch ts_h ts_l ts_c 0 3",
                )
            )

    def test_oscillator_can_precede_its_ts_declarations(self) -> None:
        """Pass-2 ordering: STOCHASTIC_OSCILLATOR resolves against any
        TIME_SERIES declared anywhere in the input."""
        pipeline = decorator_builder(
            self._commands(
                "ASSET_SCOPE STOCKS",
                # Indicator appears BEFORE its declarations.
                "STOCHASTIC_OSCILLATOR stoch ts_h ts_l ts_c 14 3",
                "TIME_SERIES ts_h high_price  2024-01-01 2024-04-01 day 1",
                "TIME_SERIES ts_l low_price   2024-01-01 2024-04-01 day 1",
                "TIME_SERIES ts_c close_price 2024-01-01 2024-04-01 day 1",
            )
        )
        assert isinstance(pipeline, StochasticOscillatorDecorator)


# ── relative_strength_index math ─────────────────────────────────────


class TestRsiMath:
    def test_steady_uptrend_pegs_rsi_at_100(self) -> None:
        """All-gains series → avg_loss == 0 → RSI = 100 (from period+1 on)."""
        closes = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        rsi = relative_strength_index(closes, period=3)
        # First `period` entries are warmup → None.
        assert rsi[0] is None
        assert rsi[1] is None
        assert rsi[2] is None
        assert rsi[3] == pytest.approx(100.0)
        assert rsi[4] == pytest.approx(100.0)
        assert rsi[5] == pytest.approx(100.0)

    def test_steady_downtrend_pegs_rsi_at_zero(self) -> None:
        """All-losses series → avg_gain == 0, avg_loss > 0 → RSI = 0."""
        closes = [15.0, 14.0, 13.0, 12.0, 11.0, 10.0]
        rsi = relative_strength_index(closes, period=3)
        assert rsi[3] == pytest.approx(0.0)
        assert rsi[5] == pytest.approx(0.0)

    def test_flat_series_returns_none(self) -> None:
        """Constant prices → zero gain *and* zero loss → undefined RSI."""
        closes = [5.0, 5.0, 5.0, 5.0, 5.0]
        rsi = relative_strength_index(closes, period=3)
        assert all(v is None for v in rsi)

    def test_classic_wilder_example(self) -> None:
        """Verify against a hand-computed Wilder smoothing example.

        Closes alternate +1 / -1 four times then +1 again (period=4).

        Initial avg_gain = (1+0+1+0)/4 = 0.5
        Initial avg_loss = (0+1+0+1)/4 = 0.5
        RS = 1.0, RSI[4] = 50.0
        Next delta = +1 (gain=1, loss=0):
          avg_gain = (0.5*3 + 1)/4 = 0.625
          avg_loss = (0.5*3 + 0)/4 = 0.375
          RS = 5/3, RSI[5] = 100 - 100/(1 + 5/3) = 62.5
        """
        closes = [10.0, 11.0, 10.0, 11.0, 10.0, 11.0]
        rsi = relative_strength_index(closes, period=4)
        assert rsi[3] is None  # only `period` entries before [period]
        assert rsi[4] == pytest.approx(50.0)
        assert rsi[5] == pytest.approx(62.5)

    def test_short_series_returns_all_none(self) -> None:
        """Fewer than period+1 closes → no valid RSI anywhere."""
        rsi = relative_strength_index([1.0, 2.0, 3.0], period=14)
        assert rsi == [None, None, None]

    def test_empty_input_returns_empty(self) -> None:
        assert relative_strength_index([], period=14) == []

    def test_rsi_range_is_zero_to_one_hundred(self) -> None:
        """For arbitrary mixed-direction data, RSI stays in [0, 100]."""
        closes = [10.0, 12.0, 11.0, 13.0, 12.5, 14.0, 13.5, 15.0]
        rsi = relative_strength_index(closes, period=3)
        for v in rsi:
            if v is None:
                continue
            assert 0.0 <= v <= 100.0

    @pytest.mark.parametrize("period", [0, -1])
    def test_invalid_period_raises(self, period: int) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            relative_strength_index([1.0, 2.0], period=period)


# ── RelativeStrengthIndexDecorator integration ───────────────────────


class TestRsiDecorator:
    def test_adds_rsi_column(self) -> None:
        """Each ticker gets a latest-RSI value in a single Float64 column."""
        ts_close = _ts(TimeSeriesName.CLOSE_PRICE, "ts_c")

        upstream_df = pl.DataFrame({cols.TICKER: ["AAPL"]})
        cache_c = TimeSeriesCacheDecorator(ts_close, _StubUpstream(upstream_df))
        chain = RelativeStrengthIndexDecorator("rsi", ts_close, 3, cache_c)

        # All-gains series → RSI = 100.
        per_def = {ts_close: [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]}
        with patch(_GET_TS_PATCH, side_effect=lambda _t, td: per_def[td]):
            df = chain._build_df()

        assert "rsi" in df.columns
        assert df["rsi"][0] == pytest.approx(100.0)

    def test_invalid_period_raises(self) -> None:
        ts = _ts(TimeSeriesName.CLOSE_PRICE, "ts")
        stub = _StubUpstream(pl.DataFrame({cols.TICKER: []}))
        with pytest.raises(ValueError, match=">= 1"):
            RelativeStrengthIndexDecorator("rsi", ts, 0, stub)

    def test_missing_series_yields_none(self) -> None:
        """If the chain has no producer for ts_close, RSI is None."""
        ts_close = _ts(TimeSeriesName.CLOSE_PRICE, "ts_c")
        upstream_df = pl.DataFrame({cols.TICKER: ["AAPL"]})
        chain = RelativeStrengthIndexDecorator(
            "rsi", ts_close, 14, _StubUpstream(upstream_df)
        )
        df = chain._build_df()
        assert df["rsi"][0] is None

    def test_flat_series_yields_none(self) -> None:
        """Constant prices through the cache → RSI undefined → None column."""
        ts_close = _ts(TimeSeriesName.CLOSE_PRICE, "ts_c")
        upstream_df = pl.DataFrame({cols.TICKER: ["AAPL"]})
        cache_c = TimeSeriesCacheDecorator(ts_close, _StubUpstream(upstream_df))
        chain = RelativeStrengthIndexDecorator("rsi", ts_close, 3, cache_c)
        per_def = {ts_close: [5.0] * 10}
        with patch(_GET_TS_PATCH, side_effect=lambda _t, td: per_def[td]):
            df = chain._build_df()
        assert df["rsi"][0] is None


# ── RSI DSL wiring ──────────────────────────────────────────────────


class TestRsiDsl:
    def _commands(self, *lines: str) -> list[list[str]]:
        return [line.split() for line in lines]

    def test_builds_chain_with_cache_and_rsi(self) -> None:
        pipeline = decorator_builder(
            self._commands(
                "ASSET_SCOPE STOCKS",
                "TIME_SERIES ts_c close_price 2024-01-01 2024-04-01 day 1",
                "RSI rsi ts_c 14",
            )
        )
        assert isinstance(pipeline, RelativeStrengthIndexDecorator)
        assert pipeline._name == "rsi"
        assert pipeline._period == 14

    def test_wrong_arity_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="requires 3 arguments"):
            decorator_builder(
                self._commands(
                    "ASSET_SCOPE STOCKS",
                    "TIME_SERIES ts_c close_price 2024-01-01 2024-04-01 day 1",
                    "RSI rsi ts_c",  # too few
                )
            )

    def test_undefined_ts_reference_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="undefined name"):
            decorator_builder(
                self._commands(
                    "ASSET_SCOPE STOCKS",
                    "TIME_SERIES ts_c close_price 2024-01-01 2024-04-01 day 1",
                    "RSI rsi tsMissing 14",
                )
            )

    def test_non_integer_period_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="must be an integer"):
            decorator_builder(
                self._commands(
                    "ASSET_SCOPE STOCKS",
                    "TIME_SERIES ts_c close_price 2024-01-01 2024-04-01 day 1",
                    "RSI rsi ts_c fourteen",
                )
            )

    def test_invalid_period_surfaces_as_processing_error(self) -> None:
        with pytest.raises(ProcessingLevelError, match=">= 1"):
            decorator_builder(
                self._commands(
                    "ASSET_SCOPE STOCKS",
                    "TIME_SERIES ts_c close_price 2024-01-01 2024-04-01 day 1",
                    "RSI rsi ts_c 0",
                )
            )

    def test_rsi_can_precede_its_ts_declaration(self) -> None:
        """Pass-2 ordering: RSI resolves against any TIME_SERIES declared
        anywhere in the input."""
        pipeline = decorator_builder(
            self._commands(
                "ASSET_SCOPE STOCKS",
                # Indicator appears BEFORE its declaration.
                "RSI rsi ts_c 14",
                "TIME_SERIES ts_c close_price 2024-01-01 2024-04-01 day 1",
            )
        )
        assert isinstance(pipeline, RelativeStrengthIndexDecorator)


# ── simple_moving_average math ───────────────────────────────────────


class TestSmaMath:
    def test_basic_three_point_window(self) -> None:
        """SMA over period=3 picks up at index 2 and rolls thereafter."""
        sma = simple_moving_average([1.0, 2.0, 3.0, 4.0, 5.0], period=3)
        assert sma[0] is None
        assert sma[1] is None
        assert sma[2] == pytest.approx(2.0)  # (1+2+3)/3
        assert sma[3] == pytest.approx(3.0)  # (2+3+4)/3
        assert sma[4] == pytest.approx(4.0)  # (3+4+5)/3

    def test_period_one_passes_through(self) -> None:
        """period=1 → SMA is the value itself, no warmup."""
        sma = simple_moving_average([2.0, 4.0, 6.0], period=1)
        assert sma == [pytest.approx(2.0), pytest.approx(4.0), pytest.approx(6.0)]

    def test_flat_series(self) -> None:
        sma = simple_moving_average([5.0, 5.0, 5.0, 5.0], period=2)
        assert sma[0] is None
        for v in sma[1:]:
            assert v == pytest.approx(5.0)

    def test_short_series_returns_all_none(self) -> None:
        """Fewer than `period` values → all None."""
        assert simple_moving_average([1.0, 2.0], period=5) == [None, None]

    def test_empty_input_returns_empty(self) -> None:
        assert simple_moving_average([], period=14) == []

    def test_window_equals_length(self) -> None:
        """period == len(values) → exactly one non-None at the end."""
        sma = simple_moving_average([1.0, 2.0, 3.0, 4.0], period=4)
        assert sma[:3] == [None, None, None]
        assert sma[3] == pytest.approx(2.5)

    def test_negative_values_average_correctly(self) -> None:
        """The cumsum trick must work for negative numbers too."""
        sma = simple_moving_average([-1.0, -2.0, -3.0, -4.0], period=2)
        assert sma[0] is None
        assert sma[1] == pytest.approx(-1.5)
        assert sma[2] == pytest.approx(-2.5)
        assert sma[3] == pytest.approx(-3.5)

    @pytest.mark.parametrize("period", [0, -1])
    def test_invalid_period_raises(self, period: int) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            simple_moving_average([1.0, 2.0], period=period)


# ── SimpleMovingAverageDecorator integration ─────────────────────────


class TestSmaDecorator:
    def test_adds_sma_column(self) -> None:
        """Each ticker gets a latest-SMA value in a single Float64 column."""
        ts_close = _ts(TimeSeriesName.CLOSE_PRICE, "ts_c")

        upstream_df = pl.DataFrame({cols.TICKER: ["AAPL"]})
        cache_c = TimeSeriesCacheDecorator(ts_close, _StubUpstream(upstream_df))
        chain = SimpleMovingAverageDecorator("sma3", ts_close, 3, cache_c)

        # Latest 3-bar SMA over [1,2,3,4,5] is (3+4+5)/3 = 4.0.
        per_def = {ts_close: [1.0, 2.0, 3.0, 4.0, 5.0]}
        with patch(_GET_TS_PATCH, side_effect=lambda _t, td: per_def[td]):
            df = chain._build_df()

        assert "sma3" in df.columns
        assert df["sma3"][0] == pytest.approx(4.0)

    def test_invalid_period_raises(self) -> None:
        ts = _ts(TimeSeriesName.CLOSE_PRICE, "ts")
        stub = _StubUpstream(pl.DataFrame({cols.TICKER: []}))
        with pytest.raises(ValueError, match=">= 1"):
            SimpleMovingAverageDecorator("sma", ts, 0, stub)

    def test_missing_series_yields_none(self) -> None:
        """If the chain has no producer for the ts, SMA is None."""
        ts_close = _ts(TimeSeriesName.CLOSE_PRICE, "ts_c")
        upstream_df = pl.DataFrame({cols.TICKER: ["AAPL"]})
        chain = SimpleMovingAverageDecorator(
            "sma", ts_close, 20, _StubUpstream(upstream_df)
        )
        df = chain._build_df()
        assert df["sma"][0] is None

    def test_short_series_yields_none(self) -> None:
        """Series shorter than period → last SMA value is None."""
        ts_close = _ts(TimeSeriesName.CLOSE_PRICE, "ts_c")
        upstream_df = pl.DataFrame({cols.TICKER: ["AAPL"]})
        cache_c = TimeSeriesCacheDecorator(ts_close, _StubUpstream(upstream_df))
        chain = SimpleMovingAverageDecorator("sma", ts_close, 10, cache_c)
        per_def = {ts_close: [1.0, 2.0, 3.0]}
        with patch(_GET_TS_PATCH, side_effect=lambda _t, td: per_def[td]):
            df = chain._build_df()
        assert df["sma"][0] is None


# ── SMA DSL wiring ───────────────────────────────────────────────────


class TestSmaDsl:
    def _commands(self, *lines: str) -> list[list[str]]:
        return [line.split() for line in lines]

    def test_builds_chain_with_cache_and_sma(self) -> None:
        pipeline = decorator_builder(
            self._commands(
                "ASSET_SCOPE STOCKS",
                "TIME_SERIES ts_c close_price 2024-01-01 2024-04-01 day 1",
                "SMA sma20 ts_c 20",
            )
        )
        assert isinstance(pipeline, SimpleMovingAverageDecorator)
        assert pipeline._name == "sma20"
        assert pipeline._period == 20

    def test_wrong_arity_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="requires 3 arguments"):
            decorator_builder(
                self._commands(
                    "ASSET_SCOPE STOCKS",
                    "TIME_SERIES ts_c close_price 2024-01-01 2024-04-01 day 1",
                    "SMA sma ts_c",  # too few
                )
            )

    def test_undefined_ts_reference_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="undefined name"):
            decorator_builder(
                self._commands(
                    "ASSET_SCOPE STOCKS",
                    "TIME_SERIES ts_c close_price 2024-01-01 2024-04-01 day 1",
                    "SMA sma tsMissing 20",
                )
            )

    def test_non_integer_period_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="must be an integer"):
            decorator_builder(
                self._commands(
                    "ASSET_SCOPE STOCKS",
                    "TIME_SERIES ts_c close_price 2024-01-01 2024-04-01 day 1",
                    "SMA sma ts_c twenty",
                )
            )

    def test_invalid_period_surfaces_as_processing_error(self) -> None:
        with pytest.raises(ProcessingLevelError, match=">= 1"):
            decorator_builder(
                self._commands(
                    "ASSET_SCOPE STOCKS",
                    "TIME_SERIES ts_c close_price 2024-01-01 2024-04-01 day 1",
                    "SMA sma ts_c 0",
                )
            )

    def test_sma_can_precede_its_ts_declaration(self) -> None:
        """Pass-2 ordering: SMA resolves against any TIME_SERIES declared
        anywhere in the input."""
        pipeline = decorator_builder(
            self._commands(
                "ASSET_SCOPE STOCKS",
                # Indicator appears BEFORE its declaration.
                "SMA sma ts_c 20",
                "TIME_SERIES ts_c close_price 2024-01-01 2024-04-01 day 1",
            )
        )
        assert isinstance(pipeline, SimpleMovingAverageDecorator)

    def test_sma_works_on_derived_series(self) -> None:
        """SMA can target a derived series (any TimeSeriesDef in the registry)."""
        pipeline = decorator_builder(
            self._commands(
                "ASSET_SCOPE STOCKS",
                "TIME_SERIES ts_o open_price  2024-01-01 2024-04-01 day 1",
                "TIME_SERIES ts_c close_price 2024-01-01 2024-04-01 day 1",
                "TIME_SERIES_DERIVED spread ts_c-ts_o",
                "SMA sma_spread spread 10",
            )
        )
        assert isinstance(pipeline, SimpleMovingAverageDecorator)
        assert pipeline._name == "sma_spread"
