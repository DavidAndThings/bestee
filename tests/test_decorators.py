"""Tests for bestee.stocks.decorators — KnowledgeBase pipeline."""

import ast
import importlib.resources
from collections.abc import Sequence
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

import bestee.resources
from bestee.stocks.builder import (
    ProcessingLevelError,
    _process_time_series,
    _process_time_series_derived,
    _process_time_series_metric,
    decorator_builder,
)
from bestee.stocks.decorators import (
    AssetScopeDecorator,
    ComputedMetricDecorator,
    FinancialMetricCacheDecorator,
    FinancialsDecorator,
    KnowledgeBaseDecorator,
    MergeDecorator,
    NoUpstreamError,
    RelativeStrengthIndexDecorator,
    SameSICategoryDecorator,
    SimpleMovingAverageDecorator,
    StochasticOscillatorDecorator,
    TimeSeriesCacheDecorator,
    TimeSeriesDerivedDecorator,
    TimeSeriesMetricDecorator,
    relative_strength_index,
    simple_moving_average,
    stochastic_oscillator,
)
from bestee.stocks.models import (
    FinancialMetric,
    KnowledgeBase,
    Metric,
    Stock,
    TimeSeriesDef,
    TimeSeriesName,
    TimeSeriesSpan,
)

# ── Test helpers ─────────────────────────────────────────────────────


class _StubUpstream(KnowledgeBaseDecorator):
    """Source stub that hands a pre-built :class:`KnowledgeBase` downstream."""

    def __init__(self, kb: KnowledgeBase):
        super().__init__(None)
        self._kb = kb

    def build_kb(self) -> KnowledgeBase:
        return self._kb


def _stub_with_tickers(
    tickers: Sequence[str], *, sic_code: str | None = None
) -> _StubUpstream:
    """Build a stub whose KB has one :class:`Stock` per ticker."""
    kb = KnowledgeBase()
    for t in tickers:
        kb.stock_data[t] = Stock(ticker=t, sic_code=sic_code)
    return _StubUpstream(kb)


def _upstreams_of(
    decorator: KnowledgeBaseDecorator,
) -> Sequence[KnowledgeBaseDecorator]:
    assert decorator._upstream is not None
    return decorator._upstream


def _apply_ts_commands(
    stub: KnowledgeBaseDecorator,
    raw_commands: list[str],
) -> tuple[KnowledgeBaseDecorator, dict[str, TimeSeriesDef]]:
    """Walk a sequence of TIME_SERIES* command strings on top of *stub*."""
    current: Sequence[KnowledgeBaseDecorator] = [stub]
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


_GET_TS_PATCH = "bestee.stocks.decorators.get_time_series"
_FINANCIALS_PATCH = "bestee.stocks.decorators.build_financials_df"


def _ts_def(
    name: TimeSeriesName = TimeSeriesName.CLOSE_PRICE,
    span: TimeSeriesSpan = TimeSeriesSpan.DAY,
    start: str = "2025-01-02",
    end: str = "2025-01-08",
    multiplier: int = 1,
    tag: str | None = None,
) -> TimeSeriesDef:
    return TimeSeriesDef(
        name=name, span=span, start=start, end=end, multiplier=multiplier, tag=tag
    )


# ── ComputedMetricDecorator parsing / validation ─────────────────────


class TestComputedMetricParse:
    @staticmethod
    def _stub() -> _StubUpstream:
        return _stub_with_tickers([])

    def test_collects_referenced_names(self) -> None:
        deco = ComputedMetricDecorator("cm1", "(fm2 * 2) / (fm3 + fm4)", self._stub())
        assert deco._referenced_names == ["fm2", "fm3", "fm4"]

    def test_chained_computed_metric_reference(self) -> None:
        deco = ComputedMetricDecorator("leverage", "roe / roa", self._stub())
        assert deco._referenced_names == ["roa", "roe"]

    def test_numeric_literals_and_unary_minus(self) -> None:
        deco = ComputedMetricDecorator("x", "-(rev * 2.5)", self._stub())
        assert deco._referenced_names == ["rev"]

    def test_syntactically_invalid_arithmetic_raises(self) -> None:
        with pytest.raises(ValueError, match="not valid Python syntax"):
            ComputedMetricDecorator("x", "rev +", self._stub())

    def test_unsupported_construct_raises(self) -> None:
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
        env: dict[str, float | None] = {"_v1": 25.0}
        assert ComputedMetricDecorator._evaluate(_expr("_v0 / _v1"), env) is None

    def test_explicit_none_propagates(self) -> None:
        env: dict[str, float | None] = {"_v0": None, "_v1": 25.0}
        assert ComputedMetricDecorator._evaluate(_expr("_v0 * _v1"), env) is None

    def test_unsupported_operator_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported binary operator"):
            ComputedMetricDecorator._evaluate(_expr("5 & 3"), {})

    def test_unsupported_literal_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported literal"):
            ComputedMetricDecorator._evaluate(_expr("'a string'"), {})


# ── decorator_builder dispatch ───────────────────────────────────────


_BASE = ["ASSET_SCOPE", "STOCKS"]


class TestDecoratorBuilder:
    def test_single_asset_scope_command(self) -> None:
        result = decorator_builder([_BASE])
        assert isinstance(result, MergeDecorator)
        assert isinstance(_upstreams_of(result)[0], AssetScopeDecorator)

    def test_asset_scope_with_same_sic(self) -> None:
        result = decorator_builder([_BASE, ["SAME_SIC_CATEGORY_AS", "AAPL"]])
        assert isinstance(result, MergeDecorator)
        inner = _upstreams_of(result)[0]
        assert isinstance(inner, SameSICategoryDecorator)
        assert inner.ticker == "AAPL"

    def test_financial_metrics_share_one_decorator(self) -> None:
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
        assert isinstance(result, ComputedMetricDecorator)
        assert result._name == "roa"
        [inner] = _upstreams_of(result)
        assert isinstance(inner, ComputedMetricDecorator)
        assert inner._name == "roe"

    def test_ordering_does_not_matter_within_phases(self) -> None:
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


# ── KnowledgeBaseDecorator base / NoUpstreamError ────────────────────


class TestNoUpstreamError:
    def test_build_from_upstream_raises_when_none(self) -> None:
        class _Stage(KnowledgeBaseDecorator):
            def build_kb(self) -> KnowledgeBase:
                return self._build_from_upstream_or_error()

        with pytest.raises(NoUpstreamError):
            _Stage().build_kb()


# ── SameSICategoryDecorator integration ──────────────────────────────


class TestSameSICategoryDecorator:
    def test_filters_by_target_sic(self) -> None:
        kb = KnowledgeBase()
        kb.stock_data["AAPL"] = Stock(ticker="AAPL", sic_code="3571")
        kb.stock_data["MSFT"] = Stock(ticker="MSFT", sic_code="3571")
        kb.stock_data["GOOG"] = Stock(ticker="GOOG", sic_code="3571")
        kb.stock_data["F"] = Stock(ticker="F", sic_code="3711")
        result_kb = SameSICategoryDecorator("AAPL", _StubUpstream(kb)).build_kb()
        assert sorted(result_kb.stock_data) == ["AAPL", "GOOG", "MSFT"]

    def test_missing_ticker_raises(self) -> None:
        stub = _stub_with_tickers(["AAPL"], sic_code="3571")
        with pytest.raises(ValueError, match="not found in upstream"):
            SameSICategoryDecorator("MSFT", stub).build_kb()

    def test_null_sic_raises(self) -> None:
        kb = KnowledgeBase()
        kb.stock_data["AAPL"] = Stock(ticker="AAPL", sic_code=None)
        with pytest.raises(ValueError, match="no SIC code"):
            SameSICategoryDecorator("AAPL", _StubUpstream(kb)).build_kb()


# ── ComputedMetricDecorator integration ──────────────────────────────


class TestComputedMetricBuild:
    def test_computes_per_ticker_from_named_metrics(self) -> None:
        kb = KnowledgeBase()
        kb.stock_data["AAPL"] = Stock(
            ticker="AAPL", numeric_metrics={"ni": 100.0, "assets": 400.0}
        )
        kb.stock_data["MSFT"] = Stock(
            ticker="MSFT", numeric_metrics={"ni": 200.0, "assets": 1000.0}
        )
        deco = ComputedMetricDecorator("roa", "ni / assets", _StubUpstream(kb))
        result_kb = deco.build_kb()
        assert result_kb.stock_data["AAPL"].numeric_metrics["roa"] == pytest.approx(
            0.25
        )
        assert result_kb.stock_data["MSFT"].numeric_metrics["roa"] == pytest.approx(0.2)

    def test_complex_expression(self) -> None:
        kb = KnowledgeBase()
        kb.stock_data["AAPL"] = Stock(
            ticker="AAPL", numeric_metrics={"fm2": 10.0, "fm3": 5.0, "fm4": 5.0}
        )
        kb.stock_data["MSFT"] = Stock(
            ticker="MSFT", numeric_metrics={"fm2": 20.0, "fm3": 30.0, "fm4": 10.0}
        )
        deco = ComputedMetricDecorator(
            "cm1", "(fm2 * 2) / (fm3 + fm4)", _StubUpstream(kb)
        )
        result_kb = deco.build_kb()
        assert result_kb.stock_data["AAPL"].numeric_metrics["cm1"] == pytest.approx(2.0)
        assert result_kb.stock_data["MSFT"].numeric_metrics["cm1"] == pytest.approx(1.0)

    def test_missing_data_yields_none(self) -> None:
        kb = KnowledgeBase()
        kb.stock_data["AAPL"] = Stock(
            ticker="AAPL", numeric_metrics={"ni": 100.0, "assets": 400.0}
        )
        kb.stock_data["MSFT"] = Stock(
            ticker="MSFT", numeric_metrics={"ni": 200.0, "assets": None}
        )
        deco = ComputedMetricDecorator("roa", "ni / assets", _StubUpstream(kb))
        result_kb = deco.build_kb()
        assert result_kb.stock_data["AAPL"].numeric_metrics["roa"] == pytest.approx(
            0.25
        )
        assert result_kb.stock_data["MSFT"].numeric_metrics["roa"] is None

    def test_division_by_zero_yields_none(self) -> None:
        kb = KnowledgeBase()
        kb.stock_data["AAPL"] = Stock(
            ticker="AAPL", numeric_metrics={"ni": 100.0, "assets": 0.0}
        )
        deco = ComputedMetricDecorator("roa", "ni / assets", _StubUpstream(kb))
        result_kb = deco.build_kb()
        assert result_kb.stock_data["AAPL"].numeric_metrics["roa"] is None

    def test_chained_computed_metric_reads_upstream_value(self) -> None:
        kb = KnowledgeBase()
        kb.stock_data["AAPL"] = Stock(ticker="AAPL", numeric_metrics={"roa": 0.25})
        kb.stock_data["MSFT"] = Stock(ticker="MSFT", numeric_metrics={"roa": 0.10})
        deco = ComputedMetricDecorator("doubled_roa", "roa * 2", _StubUpstream(kb))
        result_kb = deco.build_kb()
        assert result_kb.stock_data["AAPL"].numeric_metrics[
            "doubled_roa"
        ] == pytest.approx(0.5)
        assert result_kb.stock_data["MSFT"].numeric_metrics[
            "doubled_roa"
        ] == pytest.approx(0.2)

    def test_missing_reference_yields_none(self) -> None:
        """An expression referencing an unknown name propagates None per stock."""
        stub = _stub_with_tickers(["AAPL"])
        deco = ComputedMetricDecorator("x", "missing * 2", stub)
        result_kb = deco.build_kb()
        assert result_kb.stock_data["AAPL"].numeric_metrics["x"] is None


# ── FinancialsDecorator name handling ────────────────────────────────


class TestFinancialsDecoratorNames:
    @patch(_FINANCIALS_PATCH)
    def test_renames_columns_from_label_to_name(self, mock_build: MagicMock) -> None:
        ni = FinancialMetric(Metric.NET_INCOME, 2024, 4)
        ta = FinancialMetric(Metric.TOTAL_ASSETS, 2024, 4)
        mock_build.return_value = pl.DataFrame(
            {
                "Ticker": ["AAPL", "MSFT"],
                ni.label: [100.0, 200.0],
                ta.label: [400.0, 1000.0],
            }
        )
        stub = _stub_with_tickers(["AAPL", "MSFT"])
        deco = FinancialsDecorator(
            stub,
            FinancialMetricCacheDecorator("fm_ni", ni),
            FinancialMetricCacheDecorator("fm_ta", ta),
        )
        kb = deco.build_kb()
        assert kb.stock_data["AAPL"].numeric_metrics["fm_ni"] == pytest.approx(100.0)
        assert kb.stock_data["AAPL"].numeric_metrics["fm_ta"] == pytest.approx(400.0)
        assert kb.stock_data["MSFT"].numeric_metrics["fm_ni"] == pytest.approx(200.0)
        # Each Stock's alias_map records the binding.
        assert kb.stock_data["AAPL"].alias_map["fm_ni"] == ni
        assert kb.stock_data["AAPL"].alias_map["fm_ta"] == ta

    def test_no_caches_passes_upstream_through(self) -> None:
        stub = _stub_with_tickers(["AAPL"])
        deco = FinancialsDecorator(stub)
        kb = deco.build_kb()
        # No FinancialMetric requests → no API call, no new metrics.
        assert kb.stock_data["AAPL"].numeric_metrics == {}


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


def _chain_class_names(decorator: KnowledgeBaseDecorator) -> list[str]:
    classes: list[str] = []
    seen_ids: set[int] = set()

    def walk(d: KnowledgeBaseDecorator) -> None:
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
    commands = _read_stress(filename)
    assert commands, f"{filename} has no commands"
    pipeline = decorator_builder(commands)
    assert isinstance(pipeline, KnowledgeBaseDecorator)


@pytest.mark.parametrize("filename", _STRESS_INPUTS)
def test_stress_input_chain_shape_matches_dsl(filename: str) -> None:
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
    n_fin = _count_keyword(commands, "FINANCIAL_METRIC")
    assert chain.count("FinancialsDecorator") == (1 if n_fin else 0)
    assert chain.count("FinancialMetricCacheDecorator") == n_fin


# ── End-to-end stress: build a small pipeline and run it with mocks ──


def test_full_pipeline_executes_end_to_end_with_mocks() -> None:
    """Build and execute a pipeline that touches every phase."""
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

    # Mock the source: pretend the ASSET_SCOPE returned three tickers (one
    # with a different SIC so SAME_SIC will drop it).
    def fake_asset_scope_kb() -> KnowledgeBase:
        kb = KnowledgeBase()
        kb.stock_data["AAPL"] = Stock(ticker="AAPL", sic_code="3571")
        kb.stock_data["MSFT"] = Stock(ticker="MSFT", sic_code="3571")
        kb.stock_data["F"] = Stock(ticker="F", sic_code="3711")
        return kb

    rev_metric = FinancialMetric(Metric.REVENUE, 2024, 4)
    ni_metric = FinancialMetric(Metric.NET_INCOME, 2024, 4)
    fin_df = pl.DataFrame(
        {
            "Ticker": ["AAPL", "MSFT"],
            rev_metric.label: [400.0, 200.0],
            ni_metric.label: [80.0, 40.0],
        }
    )

    ts_series = {"AAPL": [1.0, 2.0, 3.0], "MSFT": [10.0, 20.0, 30.0]}

    with (
        patch(
            "bestee.stocks.decorators.AssetScopeDecorator.build_kb",
            side_effect=fake_asset_scope_kb,
        ),
        patch(_FINANCIALS_PATCH, return_value=fin_df),
        patch(_GET_TS_PATCH, side_effect=lambda t, _td: ts_series[t]),
    ):
        kb = pipeline.build_kb()

    assert sorted(kb.stock_data) == ["AAPL", "MSFT"]
    aapl = kb.stock_data["AAPL"]
    msft = kb.stock_data["MSFT"]
    assert aapl.numeric_metrics["rev"] == pytest.approx(400.0)
    assert msft.numeric_metrics["rev"] == pytest.approx(200.0)
    assert aapl.numeric_metrics["margin"] == pytest.approx(80.0 / 400.0)
    assert msft.numeric_metrics["margin"] == pytest.approx(40.0 / 200.0)
    assert aapl.numeric_metrics["r2"] == pytest.approx(1.0)
    assert msft.numeric_metrics["r2"] == pytest.approx(1.0)
    assert aapl.numeric_time_series["ts1"] == [1.0, 2.0, 3.0]


# ── TimeSeriesMetricDecorator integration ────────────────────────────


class TestTimeSeriesMetricDecorator:
    @patch(_GET_TS_PATCH)
    def test_adds_scalar_metric_per_ticker(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = lambda ticker, _td: {
            "AAPL": [100.0, 101.0, 102.0],
            "MSFT": [200.0, 201.0],
        }[ticker]
        stub = _stub_with_tickers(["AAPL", "MSFT"])
        ts = _ts_def(tag="ts1")
        chain = TimeSeriesMetricDecorator(
            "tsm1",
            "ts1",
            "RSquared",
            TimeSeriesCacheDecorator("ts1", ts, stub),
        )
        kb = chain.build_kb()
        assert kb.stock_data["AAPL"].numeric_metrics["tsm1"] == pytest.approx(1.0)
        assert kb.stock_data["MSFT"].numeric_metrics["tsm1"] == pytest.approx(1.0)

    @patch(_GET_TS_PATCH)
    def test_cache_fetches_each_ticker_once(self, mock_get: MagicMock) -> None:
        mock_get.return_value = [1.0, 2.0]
        stub = _stub_with_tickers(["AAPL"])
        ts = _ts_def(tag="ts1")
        cache = TimeSeriesCacheDecorator("ts1", ts, stub)
        # Build the cache once — fetch happens here.
        cache.build_kb()
        TimeSeriesMetricDecorator("a", "ts1", "RSquared", cache).build_kb()
        TimeSeriesMetricDecorator("b", "ts1", "RSquared", cache).build_kb()
        # Cache decorator fetches once per ticker per _build_kb call; the
        # metric decorators read pre-stored series off the stock, so no
        # extra fetches happen.  Total: 3 (one per _build_kb above).
        assert mock_get.call_count == 3

    def test_missing_series_yields_none(self) -> None:
        """Without an upstream cache, the source alias isn't in the stock — None."""
        stub = _stub_with_tickers(["AAPL"])
        deco = TimeSeriesMetricDecorator("tsm1", "ts1", "RSquared", stub)
        kb = deco.build_kb()
        assert kb.stock_data["AAPL"].numeric_metrics["tsm1"] is None

    def test_unknown_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown time-series metric"):
            TimeSeriesMetricDecorator(
                "tsm1",
                "ts1",
                "NoSuchMetric",
                _stub_with_tickers([]),
            )


# ── Scalar-reducer registry ──────────────────────────────────────────


class TestRSquaredTrend:
    def test_perfect_linear_trend_is_one(self) -> None:
        from bestee.stocks.decorators import _rsquared_trend

        assert _rsquared_trend([3.0, 5.0, 7.0, 9.0, 11.0]) == pytest.approx(1.0)

    def test_constant_series_is_none(self) -> None:
        from bestee.stocks.decorators import _rsquared_trend

        assert _rsquared_trend([5.0, 5.0, 5.0]) is None

    def test_too_short_is_none(self) -> None:
        from bestee.stocks.decorators import _rsquared_trend

        assert _rsquared_trend([]) is None
        assert _rsquared_trend([1.0]) is None

    def test_noisy_series_under_one(self) -> None:
        from bestee.stocks.decorators import _rsquared_trend

        r2 = _rsquared_trend([1.0, 5.0, 3.0, 8.0, 6.0, 10.0])
        assert r2 is not None
        assert 0.0 < r2 < 1.0


class TestRegisterTimeSeriesMetric:
    def test_register_and_use_a_custom_reducer(self) -> None:
        from bestee.stocks.decorators import (
            _TIME_SERIES_METRICS,
            register_time_series_metric,
        )

        try:
            register_time_series_metric(
                "LastValue", lambda s: float(s[-1]) if s else None
            )
            stub = _stub_with_tickers(["AAPL", "MSFT"])
            ts = _ts_def(tag="ts1")
            with patch(_GET_TS_PATCH) as mock_get:
                mock_get.side_effect = lambda ticker, _td: {
                    "AAPL": [10.0, 20.0, 30.0],
                    "MSFT": [100.0, 200.0],
                }[ticker]
                kb = TimeSeriesMetricDecorator(
                    "last",
                    "ts1",
                    "LastValue",
                    TimeSeriesCacheDecorator("ts1", ts, stub),
                ).build_kb()
            assert kb.stock_data["AAPL"].numeric_metrics["last"] == 30.0
            assert kb.stock_data["MSFT"].numeric_metrics["last"] == 200.0
        finally:
            _TIME_SERIES_METRICS.pop("LastValue", None)


# ── TIME_SERIES / TIME_SERIES_METRIC DSL wiring ──────────────────────


class TestTimeSeriesDsl:
    def _commands(self, *lines: str) -> list[list[str]]:
        return [line.split() for line in lines]

    def test_builds_cache_then_metric(self) -> None:
        pipeline = decorator_builder(
            self._commands(
                "ASSET_SCOPE STOCKS",
                "TIME_SERIES ts1 close_price 2025-01-02 2025-01-08 day 1",
                "TIME_SERIES_METRIC tsm1 RSquared ts1",
            )
        )
        assert isinstance(pipeline, TimeSeriesMetricDecorator)
        assert pipeline._name == "tsm1"
        assert pipeline._metric_name == "RSquared"
        assert pipeline._source_alias == "ts1"
        [inner] = _upstreams_of(pipeline)
        assert isinstance(inner, TimeSeriesCacheDecorator)
        assert inner._name == "ts1"

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

    def test_unknown_metric_surfaces_as_processing_error(self) -> None:
        with pytest.raises(ProcessingLevelError, match="Unknown time-series metric"):
            decorator_builder(
                self._commands(
                    "ASSET_SCOPE STOCKS",
                    "TIME_SERIES ts1 close_price 2025-01-02 2025-01-08 day 1",
                    "TIME_SERIES_METRIC tsm1 NotAMetric ts1",
                )
            )


# ── TimeSeriesDerivedDecorator integration ───────────────────────────


class TestTimeSeriesDerivedDecorator:
    @patch(_GET_TS_PATCH)
    def test_subtracts_two_series_elementwise(self, mock_get: MagicMock) -> None:
        ts1 = _ts_def(tag="ts1")
        ts2 = _ts_def(name=TimeSeriesName.HIGH_PRICE, tag="ts2")
        ts3 = _ts_def(tag="ts3")
        mock_get.side_effect = lambda _t, td: (
            [10.0, 11.0, 12.0] if td == ts1 else [13.0, 14.0, 15.5]
        )
        stub = _stub_with_tickers(["AAPL"])
        cache1 = TimeSeriesCacheDecorator("ts1", ts1, stub)
        cache2 = TimeSeriesCacheDecorator("ts2", ts2, cache1)
        derived = TimeSeriesDerivedDecorator(
            "ts3", "ts2-ts1", ["ts1", "ts2"], ts3, cache2
        )
        kb = derived.build_kb()
        assert kb.stock_data["AAPL"].numeric_time_series["ts3"] == [
            pytest.approx(3.0),
            pytest.approx(3.0),
            pytest.approx(3.5),
        ]

    def test_tag_distinguishes_derived_from_operands(self) -> None:
        ts1 = _ts_def(tag="ts1")
        ts3 = _ts_def(tag="ts3")
        assert (ts1.name, ts1.span, ts1.start, ts1.end, ts1.multiplier) == (
            ts3.name,
            ts3.span,
            ts3.start,
            ts3.end,
            ts3.multiplier,
        )
        assert ts1 != ts3

    def test_undefined_operand_raises_at_construction(self) -> None:
        with pytest.raises(ValueError, match="undefined names"):
            TimeSeriesDerivedDecorator(
                "out",
                "ts2 - ts1",
                ["ts1"],
                _ts_def(tag="out"),
                _stub_with_tickers([]),
            )

    @pytest.mark.parametrize(
        "expression,fragment",
        [
            ("ts1 // ts2", "unsupported construct"),
            ("ts1 ** 2", "unsupported construct"),
            ("foo(ts1)", "unsupported construct"),
            ("ts1 = 5", "valid Python syntax"),
        ],
    )
    def test_rejects_unsupported_expressions(
        self, expression: str, fragment: str
    ) -> None:
        with pytest.raises(ValueError, match=fragment):
            TimeSeriesDerivedDecorator(
                "out",
                expression,
                ["ts1", "ts2"],
                _ts_def(tag="out"),
                _stub_with_tickers([]),
            )


# ── TIME_SERIES_DERIVED DSL wiring ───────────────────────────────────


class TestTimeSeriesDerivedDsl:
    def _commands(self, *lines: str) -> list[list[str]]:
        return [line.split() for line in lines]

    def test_parses_subtraction_into_derived_chain(self) -> None:
        pipeline = decorator_builder(
            self._commands(
                "ASSET_SCOPE STOCKS",
                "TIME_SERIES ts1 close_price 2025-01-01 2026-01-01 day 1",
                "TIME_SERIES ts2 high_price 2025-01-01 2026-01-01 day 1",
                "TIME_SERIES_DERIVED ts3 ts2-ts1",
                "TIME_SERIES_METRIC tsm1 RSquared ts3",
            )
        )
        assert isinstance(pipeline, TimeSeriesMetricDecorator)
        assert pipeline._name == "tsm1"
        assert pipeline._metric_name == "RSquared"
        [derived] = _upstreams_of(pipeline)
        assert isinstance(derived, TimeSeriesDerivedDecorator)
        assert derived._expression == "ts2-ts1"
        assert derived._derived_ts_def.tag == "ts3"

    @patch(_GET_TS_PATCH)
    def test_metric_reads_derived_series_end_to_end(self, mock_get: MagicMock) -> None:
        from bestee.stocks.decorators import (
            _TIME_SERIES_METRICS,
            register_time_series_metric,
        )

        try:
            register_time_series_metric(
                "LastValue", lambda s: float(s[-1]) if s else None
            )
            stub = _stub_with_tickers(["AAPL", "MSFT"])
            decorator, ts_defs = _apply_ts_commands(
                stub,
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
            mock_get.side_effect = lambda ticker, td: per_ticker[ticker][td]
            kb = decorator.build_kb()
            # last = (ts2 - ts1)[-1]: AAPL → 20-12=8;  MSFT → 115-110=5.
            assert kb.stock_data["AAPL"].numeric_metrics["last"] == pytest.approx(8.0)
            assert kb.stock_data["MSFT"].numeric_metrics["last"] == pytest.approx(5.0)
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
        highs = [10.0, 11.0, 12.0, 13.0, 14.0]
        lows = [9.0, 10.0, 11.0, 12.0, 13.0]
        closes = [10.0, 11.0, 12.0, 13.0, 14.0]
        k, d = stochastic_oscillator(highs, lows, closes, k_period=3, d_period=2)
        assert k[0] is None and k[1] is None
        assert k[2] == pytest.approx(100.0)
        assert d[3] == pytest.approx(100.0)

    def test_steady_downtrend_pegs_k_at_zero(self) -> None:
        highs = [14.0, 13.0, 12.0, 11.0, 10.0]
        lows = [13.0, 12.0, 11.0, 10.0, 9.0]
        closes = [13.0, 12.0, 11.0, 10.0, 9.0]
        k, d = stochastic_oscillator(highs, lows, closes, k_period=3, d_period=2)
        assert k[2] == pytest.approx(0.0)
        assert d[4] == pytest.approx(0.0)

    def test_mid_range_close_yields_50_percent(self) -> None:
        highs = [110.0, 110.0, 110.0]
        lows = [100.0, 100.0, 100.0]
        closes = [105.0, 105.0, 105.0]
        k, _ = stochastic_oscillator(highs, lows, closes, k_period=3, d_period=1)
        assert k[2] == pytest.approx(50.0)

    def test_zero_range_window_returns_none(self) -> None:
        highs = [5.0, 5.0, 5.0]
        lows = [5.0, 5.0, 5.0]
        closes = [5.0, 5.0, 5.0]
        k, _ = stochastic_oscillator(highs, lows, closes, k_period=3, d_period=1)
        assert k[2] is None

    def test_empty_input_returns_empty(self) -> None:
        k, d = stochastic_oscillator([], [], [], k_period=14, d_period=3)
        assert k == [] and d == []

    @pytest.mark.parametrize("k,d", [(0, 3), (3, 0), (-1, 3)])
    def test_invalid_periods_raise(self, k: int, d: int) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            stochastic_oscillator([1.0], [1.0], [1.0], k_period=k, d_period=d)


# ── StochasticOscillatorDecorator integration ───────────────────────


class TestStochasticOscillatorDecorator:
    def test_adds_full_series_and_latest_per_ticker(self) -> None:
        """The full %K / %D series land in numeric_time_series; the
        latest value mirrors into numeric_metrics under the same key."""
        stub = _stub_with_tickers(["AAPL"])
        ts_high = _ts_def(name=TimeSeriesName.HIGH_PRICE, tag="ts_h")
        ts_low = _ts_def(name=TimeSeriesName.LOW_PRICE, tag="ts_l")
        ts_close = _ts_def(name=TimeSeriesName.CLOSE_PRICE, tag="ts_c")
        cache_h = TimeSeriesCacheDecorator("ts_h", ts_high, stub)
        cache_l = TimeSeriesCacheDecorator("ts_l", ts_low, cache_h)
        cache_c = TimeSeriesCacheDecorator("ts_c", ts_close, cache_l)
        chain = StochasticOscillatorDecorator(
            "stoch", "ts_h", "ts_l", "ts_c", 3, 2, ts_close, cache_c
        )

        def fake(_t: str, td: TimeSeriesDef) -> list[float]:
            match td:
                case _ if td == ts_high:
                    return [10.0, 11.0, 12.0, 13.0, 14.0]
                case _ if td == ts_low:
                    return [9.0, 10.0, 11.0, 12.0, 13.0]
                case _:
                    return [10.0, 11.0, 12.0, 13.0, 14.0]

        with patch(_GET_TS_PATCH, side_effect=fake):
            kb = chain.build_kb()

        stock = kb.stock_data["AAPL"]
        assert stock.numeric_metrics["stoch_k"] == pytest.approx(100.0)
        assert stock.numeric_metrics["stoch_d"] == pytest.approx(100.0)
        # Full series stored too (length 5).
        assert len(stock.numeric_time_series["stoch_k"]) == 5
        assert len(stock.numeric_time_series["stoch_d"]) == 5

    def test_invalid_periods_raise(self) -> None:
        ts = _ts_def(tag="ts")
        with pytest.raises(ValueError, match=">= 1"):
            StochasticOscillatorDecorator(
                "stoch", "ts", "ts", "ts", 0, 3, ts, _stub_with_tickers([])
            )

    def test_missing_source_series_yields_none(self) -> None:
        """No upstream cache populates the source aliases → None."""
        stub = _stub_with_tickers(["AAPL"])
        ts_close = _ts_def(tag="ts_c")
        chain = StochasticOscillatorDecorator(
            "stoch", "ts_h", "ts_l", "ts_c", 14, 3, ts_close, stub
        )
        kb = chain.build_kb()
        assert kb.stock_data["AAPL"].numeric_metrics["stoch_k"] is None
        assert kb.stock_data["AAPL"].numeric_metrics["stoch_d"] is None


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
                    "STOCHASTIC_OSCILLATOR stoch ts_h",
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
        pipeline = decorator_builder(
            self._commands(
                "ASSET_SCOPE STOCKS",
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
        closes = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        rsi = relative_strength_index(closes, period=3)
        assert rsi[0] is None and rsi[1] is None and rsi[2] is None
        assert rsi[3] == pytest.approx(100.0)
        assert rsi[5] == pytest.approx(100.0)

    def test_steady_downtrend_pegs_rsi_at_zero(self) -> None:
        closes = [15.0, 14.0, 13.0, 12.0, 11.0, 10.0]
        rsi = relative_strength_index(closes, period=3)
        assert rsi[3] == pytest.approx(0.0)
        assert rsi[5] == pytest.approx(0.0)

    def test_flat_series_returns_none(self) -> None:
        closes = [5.0, 5.0, 5.0, 5.0, 5.0]
        rsi = relative_strength_index(closes, period=3)
        assert all(v is None for v in rsi)

    def test_classic_wilder_example(self) -> None:
        closes = [10.0, 11.0, 10.0, 11.0, 10.0, 11.0]
        rsi = relative_strength_index(closes, period=4)
        assert rsi[3] is None
        assert rsi[4] == pytest.approx(50.0)
        assert rsi[5] == pytest.approx(62.5)

    def test_short_series_returns_all_none(self) -> None:
        assert relative_strength_index([1.0, 2.0, 3.0], period=14) == [None, None, None]

    def test_empty_input_returns_empty(self) -> None:
        assert relative_strength_index([], period=14) == []

    def test_rsi_range_is_zero_to_one_hundred(self) -> None:
        closes = [10.0, 12.0, 11.0, 13.0, 12.5, 14.0, 13.5, 15.0]
        for v in relative_strength_index(closes, period=3):
            if v is None:
                continue
            assert 0.0 <= v <= 100.0

    @pytest.mark.parametrize("period", [0, -1])
    def test_invalid_period_raises(self, period: int) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            relative_strength_index([1.0, 2.0], period=period)


# ── RelativeStrengthIndexDecorator integration ───────────────────────


class TestRsiDecorator:
    def test_adds_rsi_series_and_latest(self) -> None:
        stub = _stub_with_tickers(["AAPL"])
        ts_close = _ts_def(tag="ts_c")
        cache_c = TimeSeriesCacheDecorator("ts_c", ts_close, stub)
        chain = RelativeStrengthIndexDecorator("rsi", "ts_c", 3, ts_close, cache_c)
        with patch(_GET_TS_PATCH, return_value=[10.0, 11.0, 12.0, 13.0, 14.0, 15.0]):
            kb = chain.build_kb()
        stock = kb.stock_data["AAPL"]
        assert stock.numeric_metrics["rsi"] == pytest.approx(100.0)
        assert len(stock.numeric_time_series["rsi"]) == 6

    def test_invalid_period_raises(self) -> None:
        ts = _ts_def(tag="ts")
        with pytest.raises(ValueError, match=">= 1"):
            RelativeStrengthIndexDecorator("rsi", "ts", 0, ts, _stub_with_tickers([]))

    def test_missing_series_yields_none(self) -> None:
        stub = _stub_with_tickers(["AAPL"])
        ts_close = _ts_def(tag="ts_c")
        chain = RelativeStrengthIndexDecorator("rsi", "ts_c", 14, ts_close, stub)
        kb = chain.build_kb()
        assert kb.stock_data["AAPL"].numeric_metrics["rsi"] is None

    def test_flat_series_yields_none(self) -> None:
        stub = _stub_with_tickers(["AAPL"])
        ts_close = _ts_def(tag="ts_c")
        cache_c = TimeSeriesCacheDecorator("ts_c", ts_close, stub)
        chain = RelativeStrengthIndexDecorator("rsi", "ts_c", 3, ts_close, cache_c)
        with patch(_GET_TS_PATCH, return_value=[5.0] * 10):
            kb = chain.build_kb()
        assert kb.stock_data["AAPL"].numeric_metrics["rsi"] is None


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
                    "RSI rsi ts_c",
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
        pipeline = decorator_builder(
            self._commands(
                "ASSET_SCOPE STOCKS",
                "RSI rsi ts_c 14",
                "TIME_SERIES ts_c close_price 2024-01-01 2024-04-01 day 1",
            )
        )
        assert isinstance(pipeline, RelativeStrengthIndexDecorator)


# ── simple_moving_average math ───────────────────────────────────────


class TestSmaMath:
    def test_basic_three_point_window(self) -> None:
        sma = simple_moving_average([1.0, 2.0, 3.0, 4.0, 5.0], period=3)
        assert sma[0] is None and sma[1] is None
        assert sma[2] == pytest.approx(2.0)
        assert sma[3] == pytest.approx(3.0)
        assert sma[4] == pytest.approx(4.0)

    def test_period_one_passes_through(self) -> None:
        sma = simple_moving_average([2.0, 4.0, 6.0], period=1)
        assert sma == [pytest.approx(2.0), pytest.approx(4.0), pytest.approx(6.0)]

    def test_flat_series(self) -> None:
        sma = simple_moving_average([5.0, 5.0, 5.0, 5.0], period=2)
        assert sma[0] is None
        for v in sma[1:]:
            assert v == pytest.approx(5.0)

    def test_short_series_returns_all_none(self) -> None:
        assert simple_moving_average([1.0, 2.0], period=5) == [None, None]

    def test_empty_input_returns_empty(self) -> None:
        assert simple_moving_average([], period=14) == []

    def test_window_equals_length(self) -> None:
        sma = simple_moving_average([1.0, 2.0, 3.0, 4.0], period=4)
        assert sma[:3] == [None, None, None]
        assert sma[3] == pytest.approx(2.5)

    def test_negative_values_average_correctly(self) -> None:
        sma = simple_moving_average([-1.0, -2.0, -3.0, -4.0], period=2)
        assert sma[0] is None
        assert sma[3] == pytest.approx(-3.5)

    @pytest.mark.parametrize("period", [0, -1])
    def test_invalid_period_raises(self, period: int) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            simple_moving_average([1.0, 2.0], period=period)


# ── SimpleMovingAverageDecorator integration ─────────────────────────


class TestSmaDecorator:
    def test_adds_sma_series_and_latest(self) -> None:
        stub = _stub_with_tickers(["AAPL"])
        ts_close = _ts_def(tag="ts_c")
        cache_c = TimeSeriesCacheDecorator("ts_c", ts_close, stub)
        chain = SimpleMovingAverageDecorator("sma3", "ts_c", 3, ts_close, cache_c)
        with patch(_GET_TS_PATCH, return_value=[1.0, 2.0, 3.0, 4.0, 5.0]):
            kb = chain.build_kb()
        stock = kb.stock_data["AAPL"]
        # Latest 3-bar SMA over [1,2,3,4,5] = (3+4+5)/3 = 4.0.
        assert stock.numeric_metrics["sma3"] == pytest.approx(4.0)
        assert len(stock.numeric_time_series["sma3"]) == 5

    def test_invalid_period_raises(self) -> None:
        ts = _ts_def(tag="ts")
        with pytest.raises(ValueError, match=">= 1"):
            SimpleMovingAverageDecorator("sma", "ts", 0, ts, _stub_with_tickers([]))

    def test_missing_series_yields_none(self) -> None:
        stub = _stub_with_tickers(["AAPL"])
        ts_close = _ts_def(tag="ts_c")
        chain = SimpleMovingAverageDecorator("sma", "ts_c", 20, ts_close, stub)
        kb = chain.build_kb()
        assert kb.stock_data["AAPL"].numeric_metrics["sma"] is None

    def test_short_series_yields_none(self) -> None:
        stub = _stub_with_tickers(["AAPL"])
        ts_close = _ts_def(tag="ts_c")
        cache_c = TimeSeriesCacheDecorator("ts_c", ts_close, stub)
        chain = SimpleMovingAverageDecorator("sma", "ts_c", 10, ts_close, cache_c)
        with patch(_GET_TS_PATCH, return_value=[1.0, 2.0, 3.0]):
            kb = chain.build_kb()
        assert kb.stock_data["AAPL"].numeric_metrics["sma"] is None


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
                    "SMA sma ts_c",
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
        pipeline = decorator_builder(
            self._commands(
                "ASSET_SCOPE STOCKS",
                "SMA sma ts_c 20",
                "TIME_SERIES ts_c close_price 2024-01-01 2024-04-01 day 1",
            )
        )
        assert isinstance(pipeline, SimpleMovingAverageDecorator)

    def test_sma_works_on_derived_series(self) -> None:
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


# ── Cross-indicator chaining via the alias registry ──────────────────


class TestIndicatorAliasChaining:
    """Indicator output aliases (rsi*, sma*, *_k, *_d) feed back into
    the ``ts_defs`` registry, so later DERIVED / METRIC / indicator
    lines can reference them just like raw TIME_SERIES."""

    def _commands(self, *lines: str) -> list[list[str]]:
        return [line.split() for line in lines]

    def test_derived_can_reference_rsi_and_sma(self) -> None:
        pipeline = decorator_builder(
            self._commands(
                "ASSET_SCOPE STOCKS",
                "TIME_SERIES ts_c close_price 2024-01-01 2024-04-01 day 1",
                "RSI rsi14 ts_c 14",
                "SMA sma20 ts_c 20",
                # Chains off two indicator-produced aliases.
                "TIME_SERIES_DERIVED gap sma20 - rsi14",
            )
        )
        assert isinstance(pipeline, TimeSeriesDerivedDecorator)
        assert pipeline._name == "gap"
        assert pipeline._expression == "sma20 - rsi14"

    def test_sma_of_rsi_chains(self) -> None:
        """SMA can smooth an RSI series — classic technical-analysis pattern."""
        pipeline = decorator_builder(
            self._commands(
                "ASSET_SCOPE STOCKS",
                "TIME_SERIES ts_c close_price 2024-01-01 2024-04-01 day 1",
                "RSI rsi14 ts_c 14",
                "SMA smoothed_rsi rsi14 5",
            )
        )
        assert isinstance(pipeline, SimpleMovingAverageDecorator)
        assert pipeline._name == "smoothed_rsi"
        assert pipeline._ts_alias == "rsi14"

    def test_ts_metric_can_reduce_an_indicator(self) -> None:
        """TIME_SERIES_METRIC on RSI: e.g. R² of the RSI series."""
        pipeline = decorator_builder(
            self._commands(
                "ASSET_SCOPE STOCKS",
                "TIME_SERIES ts_c close_price 2024-01-01 2024-04-01 day 1",
                "RSI rsi14 ts_c 14",
                "TIME_SERIES_METRIC rsi_trend RSquared rsi14",
            )
        )
        assert isinstance(pipeline, TimeSeriesMetricDecorator)
        assert pipeline._source_alias == "rsi14"

    def test_stochastic_k_and_d_aliases_are_referenceable(self) -> None:
        """%K and %D both register and can be referenced by name."""
        pipeline = decorator_builder(
            self._commands(
                "ASSET_SCOPE STOCKS",
                "TIME_SERIES ts_h high_price  2024-01-01 2024-04-01 day 1",
                "TIME_SERIES ts_l low_price   2024-01-01 2024-04-01 day 1",
                "TIME_SERIES ts_c close_price 2024-01-01 2024-04-01 day 1",
                "STOCHASTIC_OSCILLATOR stoch ts_h ts_l ts_c 14 3",
                "TIME_SERIES_DERIVED kd_spread stoch_k - stoch_d",
            )
        )
        assert isinstance(pipeline, TimeSeriesDerivedDecorator)
        assert pipeline._expression == "stoch_k - stoch_d"

    def test_out_of_order_reference_raises(self) -> None:
        """An indicator-produced alias referenced BEFORE the indicator
        appears in input order is rejected — pass-2 is strict left-to-right."""
        with pytest.raises(ProcessingLevelError, match="undefined names"):
            decorator_builder(
                self._commands(
                    "ASSET_SCOPE STOCKS",
                    "TIME_SERIES ts_c close_price 2024-01-01 2024-04-01 day 1",
                    # Derived uses rsi14 BEFORE RSI declares it.
                    "TIME_SERIES_DERIVED bad 2 * rsi14",
                    "RSI rsi14 ts_c 14",
                )
            )

    def test_indicator_alias_collision_raises(self) -> None:
        with pytest.raises(ProcessingLevelError, match="already defined"):
            decorator_builder(
                self._commands(
                    "ASSET_SCOPE STOCKS",
                    "TIME_SERIES ts_c close_price 2024-01-01 2024-04-01 day 1",
                    "RSI dup ts_c 14",
                    "SMA dup ts_c 20",  # Reuses the alias `dup`.
                )
            )

    @patch(_GET_TS_PATCH)
    def test_chained_indicators_compute_end_to_end(self, mock_get: MagicMock) -> None:
        """SMA-of-RSI actually evaluates correctly through the pipeline.

        Steadily rising closes → RSI saturates at 100 → SMA(100) = 100.
        Bypasses ASSET_SCOPE by stitching the chain onto a stub upstream.
        """
        from bestee.stocks.builder import _process_rsi, _process_sma

        mock_get.return_value = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]
        stub = _stub_with_tickers(["AAPL"])
        decorator, ts_defs = _apply_ts_commands(
            stub,
            ["TIME_SERIES ts_c close_price 2025-01-01 2025-01-10 day 1"],
        )
        current: Sequence[KnowledgeBaseDecorator] = [decorator]
        current, ts_defs = _process_rsi(["RSI", "rsi", "ts_c", "3"], current, ts_defs)
        current, ts_defs = _process_sma(
            ["SMA", "smoothed_rsi", "rsi", "2"], current, ts_defs
        )
        [head] = current
        kb = head.build_kb()
        stock = kb.stock_data["AAPL"]
        assert stock.numeric_metrics["rsi"] == pytest.approx(100.0)
        assert stock.numeric_metrics["smoothed_rsi"] == pytest.approx(100.0)
        # Full SMA series is stored, not just the scalar.
        assert "smoothed_rsi" in stock.numeric_time_series
