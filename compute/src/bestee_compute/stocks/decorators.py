"""KnowledgeBase-building decorators for the DSL pipeline.

Each stage in the pipeline implements :meth:`KnowledgeBaseDecorator.build_kb`,
which receives the upstream :class:`KnowledgeBase` (or builds an initial
one in the case of source stages) and **mutates it in place** before
returning it.  Downstream stages then layer their own additions on top.

The shared payload is per-ticker.  :class:`Stock` carries categorical
identity fields (name, SIC code, exchange, …), a ``numeric_metrics``
dict for scalar values (revenue, latest RSI, …), a ``numeric_time_series``
dict for full OHLC / indicator series, and an ``alias_map`` recording
the origin of each populated alias (a :class:`FinancialMetric` or
:class:`TimeSeriesDef`).  Every key in those dicts is the DSL alias the
user wrote — e.g. an ``RSI rsi14 ts_c 14`` line populates
``stock.numeric_time_series["rsi14"]`` with the full RSI series and
``stock.alias_map["rsi14"]`` with the synthesised :class:`TimeSeriesDef`.

Stages are stitched into a chain by :mod:`bestee_compute.stocks.builder`; the
caller invokes :meth:`KnowledgeBaseDecorator.build_kb` to materialise
the :class:`KnowledgeBase` for inspection or rendering.
"""

from __future__ import annotations

import ast
import logging
import operator
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from bestee_compute.stocks import columns as cols
from bestee_compute.stocks.financials import build_financials_df
from bestee_compute.stocks.market import get_time_series
from bestee_compute.stocks.models import (
    FinancialMetric,
    KnowledgeBase,
    Stock,
    TimeSeriesDef,
)
from bestee_compute.stocks.tickers import get_all_tickers_df, get_ticker_details_df

logger = logging.getLogger(__name__)

type Command = Sequence[str]


class NoUpstreamError(Exception):
    pass


# ── Base class ───────────────────────────────────────────────────────


class KnowledgeBaseDecorator(ABC):
    """Pipeline stage that mutates and forwards a :class:`KnowledgeBase`."""

    def __init__(
        self,
        upstream_decorators: Sequence[KnowledgeBaseDecorator] | None = None,
    ):
        self._upstream = upstream_decorators

    def _build_from_upstream(self) -> KnowledgeBase:
        """Return the combined upstream :class:`KnowledgeBase`.

        Single-upstream case delegates to that upstream's :meth:`build_kb`.
        Subclasses with fan-in (multiple upstreams) override this.
        """
        assert self._upstream is not None
        if len(self._upstream) == 1:
            return self._upstream[0].build_kb()
        msg = (
            f"{type(self).__name__} has {len(self._upstream)} upstreams; "
            "override _build_from_upstream to combine them."
        )
        raise NotImplementedError(msg)

    def _build_from_upstream_or_error(self) -> KnowledgeBase:
        if self._upstream is None:
            raise NoUpstreamError("No upstream decorator")
        return self._build_from_upstream()

    @abstractmethod
    def build_kb(self) -> KnowledgeBase:
        """Return this stage's output KnowledgeBase."""


# ── Source stage ─────────────────────────────────────────────────────


# DataFrame column → Stock attribute name.  Mirrors tickers._DETAIL_FIELDS
# but goes the other direction (the details DataFrame uses cols.* labels;
# we turn each row into a Stock).
_STOCK_CATEGORICAL_FIELDS: dict[str, str] = {
    cols.NAME: "name",
    cols.DESCRIPTION: "description",
    cols.TYPE: "type",
    cols.MARKET: "market",
    cols.LOCALE: "locale",
    cols.PRIMARY_EXCHANGE: "primary_exchange",
    cols.CURRENCY: "currency_name",
    cols.CIK: "cik",
    cols.COMPOSITE_FIGI: "composite_figi",
    cols.SHARE_CLASS_FIGI: "share_class_figi",
    cols.SIC_CODE: "sic_code",
    cols.SIC_DESCRIPTION: "sic_description",
    cols.MARKET_CAP: "market_cap",
    cols.SHARES_OUTSTANDING: "share_class_shares_outstanding",
    cols.WEIGHTED_SHARES_OUTSTANDING: "weighted_shares_outstanding",
    cols.TOTAL_EMPLOYEES: "total_employees",
    cols.LIST_DATE: "list_date",
    cols.HOMEPAGE_URL: "homepage_url",
    cols.PHONE_NUMBER: "phone_number",
    cols.TICKER_ROOT: "ticker_root",
}


def _stock_from_row(row: Mapping[str, Any]) -> Stock | None:
    """Build a :class:`Stock` from one ticker-details DataFrame row."""
    ticker = row.get(cols.TICKER)
    if ticker is None:
        return None
    kwargs: dict[str, Any] = {"ticker": ticker}
    for col_name, attr in _STOCK_CATEGORICAL_FIELDS.items():
        kwargs[attr] = row.get(col_name)
    return Stock(**kwargs)


class AssetScopeDecorator(KnowledgeBaseDecorator):
    """Seed the KnowledgeBase with one :class:`Stock` per ticker in scope.

    Fetches the active ticker list for the given ``ticker_type`` plus the
    full ticker-details payload, then materialises a :class:`Stock` for
    every symbol with its categorical fields populated.  Numeric payloads
    start empty — later stages fill them.
    """

    def __init__(self, ticker_type: str):
        super().__init__(None)
        self.ticker_type = ticker_type

    def build_kb(self) -> KnowledgeBase:
        symbols_df = get_all_tickers_df(ticker_type=self.ticker_type, active=True)
        symbols: list[str] = symbols_df[cols.TICKER].to_list()
        details_df = get_ticker_details_df(symbols)
        kb = KnowledgeBase()
        for row in details_df.iter_rows(named=True):
            stock = _stock_from_row(row)
            if stock is None:
                continue
            kb.stock_data[stock.ticker] = stock
        logger.info(
            "AssetScopeDecorator(%r): seeded KB with %d stock(s)",
            self.ticker_type,
            len(kb.stock_data),
        )
        return kb


# ── Subsetting ───────────────────────────────────────────────────────


class SameSICategoryDecorator(KnowledgeBaseDecorator):
    """Prune ``stock_data`` to tickers sharing the target's SIC code."""

    def __init__(self, ticker: str, *upstream: KnowledgeBaseDecorator):
        super().__init__(upstream)
        self.ticker = ticker

    def build_kb(self) -> KnowledgeBase:
        kb = self._build_from_upstream_or_error()
        target = kb.stock_data.get(self.ticker)
        if target is None:
            msg = f"Ticker {self.ticker!r} not found in upstream KnowledgeBase"
            raise ValueError(msg)
        if target.sic_code is None:
            msg = f"Ticker {self.ticker!r} has no SIC code"
            raise ValueError(msg)
        target_sic = target.sic_code
        kb.stock_data = {
            t: s for t, s in kb.stock_data.items() if s.sic_code == target_sic
        }
        logger.info(
            "SameSICategoryDecorator(%r): %d stock(s) share SIC %s",
            self.ticker,
            len(kb.stock_data),
            target_sic,
        )
        return kb


class PickTickersDecorator(KnowledgeBaseDecorator):
    """Prune ``stock_data`` to a fixed list of tickers."""

    def __init__(self, tickers: Sequence[str], *upstream: KnowledgeBaseDecorator):
        super().__init__(upstream)
        self.tickers = list(tickers)

    def build_kb(self) -> KnowledgeBase:
        kb = self._build_from_upstream_or_error()
        wanted = set(self.tickers)
        kb.stock_data = {t: s for t, s in kb.stock_data.items() if t in wanted}
        logger.info(
            "PickTickersDecorator(%d wanted): %d kept",
            len(wanted),
            len(kb.stock_data),
        )
        return kb


class MergeDecorator(KnowledgeBaseDecorator):
    """Merge multiple upstream :class:`KnowledgeBase`s into one."""

    def __init__(self, *upstream: KnowledgeBaseDecorator):
        super().__init__(upstream)

    def build_kb(self) -> KnowledgeBase:
        kb = KnowledgeBase()
        assert self._upstream is not None, "MergeDecorator requires upstream decorators"

        for upstream in self._upstream:
            kb.merge_with(upstream.build_kb())

        return kb


# ── Financial metrics ────────────────────────────────────────────────


class FinancialMetricCacheDecorator(KnowledgeBaseDecorator):
    """Config-only marker registering one ``(alias, FinancialMetric)`` pair.

    Passed alongside the table-providing upstream of a
    :class:`FinancialsDecorator`, which collects them into a single
    batched API call.
    """

    def __init__(
        self,
        metric_name: str,
        metric: FinancialMetric,
        *upstream: KnowledgeBaseDecorator,
    ):
        super().__init__(upstream)
        self.metric_name = metric_name
        self.metric = metric

    def build_kb(self) -> KnowledgeBase:
        msg = (
            "FinancialMetricCacheDecorator is a config-only stage; pass it "
            "as an upstream to a FinancialsDecorator instead of building it "
            "directly."
        )
        raise NoUpstreamError(msg)


class FinancialsDecorator(KnowledgeBaseDecorator):
    """Batch-fetch every registered FinancialMetric and write into stocks.

    One upstream is the table-providing chain head (any KB-producing
    decorator); the rest are :class:`FinancialMetricCacheDecorator`
    markers that declare ``(alias, FinancialMetric)`` pairs.  All metrics
    are fetched in one ``build_financials_df`` round-trip and stamped
    into ``stock.numeric_metrics`` per ticker.
    """

    def __init__(self, *upstream: KnowledgeBaseDecorator):
        super().__init__(upstream)

    @property
    def _named_metrics(self) -> dict[str, FinancialMetric]:
        assert self._upstream is not None
        named: dict[str, FinancialMetric] = {}
        for d in self._upstream:
            if isinstance(d, FinancialMetricCacheDecorator):
                named[d.metric_name] = d.metric
        return named

    @property
    def metrics(self) -> list[FinancialMetric]:
        """The bare metric requests (no alias) — handy in tests."""
        return list(self._named_metrics.values())

    def _build_from_upstream(self) -> KnowledgeBase:
        assert self._upstream is not None
        table_upstreams = [
            d
            for d in self._upstream
            if not isinstance(d, FinancialMetricCacheDecorator)
        ]
        if len(table_upstreams) != 1:
            msg = (
                "FinancialsDecorator expects exactly one non-cache upstream "
                f"to provide the KnowledgeBase; got {len(table_upstreams)}."
            )
            raise ValueError(msg)
        return table_upstreams[0].build_kb()

    def build_kb(self) -> KnowledgeBase:
        kb = self._build_from_upstream_or_error()
        named = self._named_metrics
        if not named:
            return kb
        tickers = list(kb.stock_data)
        ordered_aliases = list(named)
        ordered_metrics = [named[a] for a in ordered_aliases]
        metrics_df = build_financials_df(tickers=tickers, metrics=ordered_metrics)
        label_to_alias = {named[a].label: a for a in ordered_aliases}
        for row in metrics_df.iter_rows(named=True):
            ticker = row.get(cols.TICKER)
            if ticker is None:
                continue
            stock = kb.stock_data.get(ticker)
            if stock is None:
                continue
            for label, alias in label_to_alias.items():
                value = row.get(label)
                stock.set_numeric_metric(
                    alias,
                    named[alias],
                    float(value) if value is not None else None,
                )
        logger.info(
            "FinancialsDecorator: %d metric(s) × %d ticker(s)",
            len(named),
            len(tickers),
        )
        return kb


# ── Computed metric (scalar arithmetic) ──────────────────────────────


class ComputedMetricDecorator(KnowledgeBaseDecorator):
    """Add a derived scalar metric defined by a Python arithmetic expression.

    Identifiers in the expression resolve against names already present
    in ``stock.numeric_metrics`` (from earlier ``FINANCIAL_METRIC`` /
    ``COMPUTED_METRIC`` / ``TIME_SERIES_METRIC`` / ``RSI`` / ``SMA`` /
    ``STOCHASTIC_OSCILLATOR`` lines).  Allowed: ``+ - * / ** %``,
    unary ``±``, parens, and numeric literals.

    Evaluation parses with :mod:`ast` (not ``eval``); the AST is
    validated against an allow-list at construction time.  ``None``
    propagates for any missing operand, divide-by-zero, or mod-by-zero
    so a single bad data point doesn't sink the whole metric.
    """

    _ALLOWED_BINOPS: tuple[type[ast.operator], ...] = (
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
    )
    _ALLOWED_UNARYOPS: tuple[type[ast.unaryop], ...] = (ast.UAdd, ast.USub)

    def __init__(self, name: str, expression: str, *upstream: KnowledgeBaseDecorator):
        super().__init__(upstream)
        self._name = name
        self._expression = expression
        try:
            self._ast_tree = ast.parse(expression, mode="eval").body
        except SyntaxError as err:
            msg = (
                f"COMPUTED_METRIC expression {expression!r} is not "
                f"valid Python syntax: {err.msg}"
            )
            raise ValueError(msg) from err
        self._validate(self._ast_tree)
        self._referenced_names: list[str] = sorted(self._collect_names(self._ast_tree))
        logger.info(
            "ComputedMetricDecorator %r references %d name(s): %s",
            self._name,
            len(self._referenced_names),
            self._referenced_names,
        )

    @classmethod
    def _collect_names(cls, node: ast.AST) -> set[str]:
        return {sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name)}

    @classmethod
    def _validate(cls, node: ast.AST) -> None:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name):
                continue
            if isinstance(sub, ast.Constant):
                if not isinstance(sub.value, int | float):
                    msg = (
                        f"COMPUTED_METRIC constants must be numeric, got {sub.value!r}"
                    )
                    raise ValueError(msg)
                continue
            if isinstance(sub, ast.UnaryOp) and isinstance(
                sub.op, cls._ALLOWED_UNARYOPS
            ):
                continue
            if isinstance(sub, ast.BinOp) and isinstance(sub.op, cls._ALLOWED_BINOPS):
                continue
            if isinstance(sub, ast.Expression):
                continue
            if isinstance(sub, ast.operator | ast.unaryop | ast.expr_context):
                continue
            msg = (
                "COMPUTED_METRIC expression contains an unsupported "
                f"construct: {type(sub).__name__}"
            )
            raise ValueError(msg)

    @classmethod
    def _evaluate(
        cls,
        node: ast.AST,
        env: Mapping[str, float | None],
    ) -> float | None:
        if isinstance(node, ast.Expression):
            return cls._evaluate(node.body, env)
        if isinstance(node, ast.BinOp):
            left = cls._evaluate(node.left, env)
            right = cls._evaluate(node.right, env)
            if left is None or right is None:
                return None
            op = node.op
            if isinstance(op, ast.Add):
                return left + right
            if isinstance(op, ast.Sub):
                return left - right
            if isinstance(op, ast.Mult):
                return left * right
            if isinstance(op, ast.Div):
                if right == 0:
                    return None
                return left / right
            if isinstance(op, ast.Pow):
                return float(left**right)
            if isinstance(op, ast.Mod):
                if right == 0:
                    return None
                return left % right
            msg = f"Unsupported binary operator: {type(op).__name__}"
            raise ValueError(msg)
        if isinstance(node, ast.UnaryOp):
            operand_value = cls._evaluate(node.operand, env)
            if operand_value is None:
                return None
            if isinstance(node.op, ast.USub):
                return -operand_value
            if isinstance(node.op, ast.UAdd):
                return operand_value
            msg = f"Unsupported unary operator: {type(node.op).__name__}"
            raise ValueError(msg)
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, int | float):
                return float(value)
            msg = f"Unsupported literal: {value!r}"
            raise ValueError(msg)
        if isinstance(node, ast.Name):
            return env.get(node.id)
        msg = f"Unsupported expression node: {type(node).__name__}"
        raise ValueError(msg)

    def build_kb(self) -> KnowledgeBase:
        kb = self._build_from_upstream_or_error()
        non_null = 0
        for stock in kb.stock_data.values():
            env: dict[str, float | None] = {}
            for n in self._referenced_names:
                env[n] = stock.numeric_metrics.get(n)
            try:
                value = self._evaluate(self._ast_tree, env)
            except TypeError, ZeroDivisionError, OverflowError:
                value = None
            stock.numeric_metrics[self._name] = value
            if value is not None:
                non_null += 1
        logger.info(
            "ComputedMetricDecorator %r: %d ticker(s) (%d non-null)",
            self._name,
            len(kb.stock_data),
            non_null,
        )
        return kb


# ── Time-series cache (raw OHLC fetch) ───────────────────────────────


class TimeSeriesCacheDecorator(KnowledgeBaseDecorator):
    """Fetch and store one OHLC series per ticker under a DSL alias.

    Registers ``alias_map[alias] = ts_def`` and populates
    ``stock.numeric_time_series[alias]`` for every stock in scope.  The
    fetch is eager — once stock_data has been subset (by SAME_SIC /
    PICK_TICKERS), this stage hits the API once per surviving ticker.
    """

    def __init__(
        self,
        alias: str,
        ts_def: TimeSeriesDef,
        *upstream: KnowledgeBaseDecorator,
    ):
        super().__init__(upstream)
        self._name = alias
        self._ts_def = ts_def

    def build_kb(self) -> KnowledgeBase:
        kb = self._build_from_upstream_or_error()
        for ticker, stock in kb.stock_data.items():
            values = get_time_series(ticker, self._ts_def)
            stock.set_numeric_time_series(self._name, self._ts_def, list(values))
        logger.info(
            "TimeSeriesCacheDecorator %r: fetched %d ticker(s)",
            self._name,
            len(kb.stock_data),
        )
        return kb


# ── Scalar reducers over a time series ───────────────────────────────


type TimeSeriesScalarMetric = Callable[[Sequence[float]], float | None]


def _rsquared_trend(series: Sequence[float]) -> float | None:
    """R² of a least-squares linear fit of *series* against its index.

    Equivalent to the squared Pearson correlation between the bar index
    (0, 1, 2, …) and the value at that bar.  A linear trend scores 1.0;
    a flat or noisy series scores near 0.  Returns ``None`` for series
    too short (< 2 valid points after dropping NaNs) or with zero
    variance.  NaN-padded warmup from upstream indicators is dropped so
    R² can be applied to an RSI or SMA series cleanly.
    """
    arr = np.asarray(series, dtype=np.float64)
    valid = ~np.isnan(arr)
    if int(valid.sum()) < 2:
        return None
    xs = np.arange(arr.size, dtype=np.float64)[valid]
    ys = arr[valid]
    if float(ys.var()) == 0.0:
        return None
    r = np.corrcoef(xs, ys)[0, 1]
    if not np.isfinite(r):
        return None
    return float(r * r)


_TIME_SERIES_METRICS: dict[str, TimeSeriesScalarMetric] = {
    "RSquared": _rsquared_trend,
}


def register_time_series_metric(name: str, fn: TimeSeriesScalarMetric) -> None:
    _TIME_SERIES_METRICS[name] = fn


def available_time_series_metrics() -> list[str]:
    return sorted(_TIME_SERIES_METRICS)


# ── Time-series derivations and reductions ───────────────────────────


class TimeSeriesDerivedDecorator(KnowledgeBaseDecorator):
    """Define a new series as arithmetic over previously-stored ones.

    Reads operand series from each :class:`Stock`'s ``numeric_time_series``
    by alias, evaluates the expression element-wise via numpy (broadcasting
    works), and writes the result under the new alias.  Allowed operators:
    ``+ - * /``, unary minus, numeric literals.
    """

    _ALLOWED_BINOPS: tuple[type[ast.operator], ...] = (
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
    )
    _BINOP_FUNCS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    def __init__(
        self,
        alias: str,
        expression: str,
        known_aliases: Sequence[str],
        derived_ts_def: TimeSeriesDef,
        *upstream: KnowledgeBaseDecorator,
    ):
        super().__init__(upstream)
        self._name = alias
        self._expression = expression
        self._derived_ts_def = derived_ts_def
        try:
            self._ast_tree = ast.parse(expression, mode="eval").body
        except SyntaxError as err:
            msg = (
                f"TIME_SERIES_DERIVED expression {expression!r} is not "
                f"valid Python syntax: {err.msg}"
            )
            raise ValueError(msg) from err
        self._validate(self._ast_tree)
        self._operand_names = sorted(self._collect_names(self._ast_tree))
        missing = [n for n in self._operand_names if n not in set(known_aliases)]
        if missing:
            msg = (
                f"TIME_SERIES_DERIVED expression {expression!r} refers to "
                f"undefined names {missing!r}. Defined: {sorted(known_aliases)}."
            )
            raise ValueError(msg)

    @classmethod
    def _collect_names(cls, node: ast.AST) -> set[str]:
        return {sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name)}

    @classmethod
    def _validate(cls, node: ast.AST) -> None:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name):
                continue
            if isinstance(sub, ast.Constant):
                if not isinstance(sub.value, int | float):
                    msg = (
                        "TIME_SERIES_DERIVED constants must be numeric, "
                        f"got {sub.value!r}"
                    )
                    raise ValueError(msg)
                continue
            if isinstance(sub, ast.UnaryOp) and isinstance(sub.op, ast.USub):
                continue
            if isinstance(sub, ast.BinOp) and isinstance(sub.op, cls._ALLOWED_BINOPS):
                continue
            if isinstance(sub, ast.Expression):
                continue
            if isinstance(sub, ast.operator | ast.unaryop | ast.expr_context):
                continue
            msg = (
                "TIME_SERIES_DERIVED expression contains an unsupported "
                f"construct: {type(sub).__name__}"
            )
            raise ValueError(msg)

    @classmethod
    def _evaluate(
        cls,
        node: ast.AST,
        operands: Mapping[str, np.ndarray],
    ) -> Any:
        if isinstance(node, ast.Name):
            return operands[node.id]
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -cls._evaluate(node.operand, operands)
        if isinstance(node, ast.BinOp) and isinstance(node.op, cls._ALLOWED_BINOPS):
            left = cls._evaluate(node.left, operands)
            right = cls._evaluate(node.right, operands)
            return cls._BINOP_FUNCS[type(node.op)](left, right)
        msg = (
            "TIME_SERIES_DERIVED expression contains an unsupported "
            f"construct at runtime: {type(node).__name__}"
        )
        raise ValueError(msg)

    def build_kb(self) -> KnowledgeBase:
        kb = self._build_from_upstream_or_error()
        for stock in kb.stock_data.values():
            operands: dict[str, np.ndarray] = {}
            missing = False
            for n in self._operand_names:
                series = stock.numeric_time_series.get(n)
                if series is None:
                    missing = True
                    break
                operands[n] = np.asarray(series, dtype=np.float64)
            if missing:
                stock.set_numeric_time_series(self._name, self._derived_ts_def, [])
                continue
            result = self._evaluate(self._ast_tree, operands)
            stock.set_numeric_time_series(
                self._name,
                self._derived_ts_def,
                np.asarray(result, dtype=np.float64).tolist(),
            )
        logger.info(
            "TimeSeriesDerivedDecorator %r: applied to %d ticker(s)",
            self._name,
            len(kb.stock_data),
        )
        return kb


class TimeSeriesMetricDecorator(KnowledgeBaseDecorator):
    """Reduce a per-ticker series to one scalar via a registered metric.

    Looks up the series by its alias on each :class:`Stock` and writes
    the reducer's output into ``stock.numeric_metrics[alias]``.  Use
    :func:`register_time_series_metric` to add reducers; built-ins
    include ``"RSquared"``.
    """

    def __init__(
        self,
        alias: str,
        source_alias: str,
        metric: str,
        *upstream: KnowledgeBaseDecorator,
    ):
        super().__init__(upstream)
        if metric not in _TIME_SERIES_METRICS:
            msg = (
                f"Unknown time-series metric {metric!r}. "
                f"Available: {available_time_series_metrics()}."
            )
            raise ValueError(msg)
        self._name = alias
        self._source_alias = source_alias
        self._metric_name = metric
        self._metric_fn = _TIME_SERIES_METRICS[metric]

    def build_kb(self) -> KnowledgeBase:
        kb = self._build_from_upstream_or_error()
        non_null = 0
        for stock in kb.stock_data.values():
            series = stock.numeric_time_series.get(self._source_alias)
            if series is None:
                stock.numeric_metrics[self._name] = None
                continue
            value = self._metric_fn(series)
            stock.numeric_metrics[self._name] = value
            if value is not None:
                non_null += 1
        logger.info(
            "TimeSeriesMetricDecorator %r (metric=%r): %d ticker(s) (%d non-null)",
            self._name,
            self._metric_name,
            len(kb.stock_data),
            non_null,
        )
        return kb


# ── Stochastic oscillator ────────────────────────────────────────────


def stochastic_oscillator(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    k_period: int,
    d_period: int,
) -> tuple[list[float | None], list[float | None]]:
    """Compute the classic stochastic-oscillator %K and %D series."""
    if k_period < 1 or d_period < 1:
        msg = (
            f"stochastic_oscillator periods must be >= 1; "
            f"got k_period={k_period}, d_period={d_period}"
        )
        raise ValueError(msg)
    n = min(len(highs), len(lows), len(closes))
    if n == 0:
        return [], []
    highs_arr = np.asarray(highs[:n], dtype=np.float64)
    lows_arr = np.asarray(lows[:n], dtype=np.float64)
    closes_arr = np.asarray(closes[:n], dtype=np.float64)
    k_values: list[float | None] = []
    for i in range(n):
        if i < k_period - 1:
            k_values.append(None)
            continue
        window_low = float(lows_arr[i - k_period + 1 : i + 1].min())
        window_high = float(highs_arr[i - k_period + 1 : i + 1].max())
        denom = window_high - window_low
        if denom == 0.0:
            k_values.append(None)
            continue
        k_values.append(100.0 * (float(closes_arr[i]) - window_low) / denom)
    d_values: list[float | None] = []
    warmup = k_period - 1 + d_period - 1
    for i in range(n):
        if i < warmup:
            d_values.append(None)
            continue
        window = k_values[i - d_period + 1 : i + 1]
        if any(v is None for v in window):
            d_values.append(None)
            continue
        d_values.append(sum(v for v in window if v is not None) / d_period)
    return k_values, d_values


class StochasticOscillatorDecorator(KnowledgeBaseDecorator):
    """Compute %K and %D and store them as time series + latest scalars.

    Writes the **full** %K and %D series under
    ``stock.numeric_time_series["<name>_k"]`` and ``"<name>_d"`` (so
    downstream indicators can chain off them), and the latest values
    into ``stock.numeric_metrics`` under the same keys.  Both indicator
    aliases are registered in ``alias_map`` against a synthesised
    :class:`TimeSeriesDef` cloned from the close-series shape.
    """

    def __init__(
        self,
        name: str,
        ts_high_alias: str,
        ts_low_alias: str,
        ts_close_alias: str,
        k_period: int,
        d_period: int,
        close_ts_def: TimeSeriesDef,
        *upstream: KnowledgeBaseDecorator,
    ):
        super().__init__(upstream)
        if k_period < 1 or d_period < 1:
            msg = (
                f"STOCHASTIC_OSCILLATOR periods must be >= 1; "
                f"got k_period={k_period}, d_period={d_period}"
            )
            raise ValueError(msg)
        self._name = name
        self._ts_high_alias = ts_high_alias
        self._ts_low_alias = ts_low_alias
        self._ts_close_alias = ts_close_alias
        self._k_period = k_period
        self._d_period = d_period
        self._k_alias = f"{name}_k"
        self._d_alias = f"{name}_d"
        # Indicator series inherit the close series' time shape so a
        # downstream SMA/RSI on them gets a coherent TimeSeriesDef.
        self._k_ts_def = TimeSeriesDef(
            name=close_ts_def.name,
            span=close_ts_def.span,
            start=close_ts_def.start,
            end=close_ts_def.end,
            multiplier=close_ts_def.multiplier,
            tag=self._k_alias,
        )
        self._d_ts_def = TimeSeriesDef(
            name=close_ts_def.name,
            span=close_ts_def.span,
            start=close_ts_def.start,
            end=close_ts_def.end,
            multiplier=close_ts_def.multiplier,
            tag=self._d_alias,
        )

    def build_kb(self) -> KnowledgeBase:
        kb = self._build_from_upstream_or_error()
        for stock in kb.stock_data.values():
            highs = stock.numeric_time_series.get(self._ts_high_alias)
            lows = stock.numeric_time_series.get(self._ts_low_alias)
            closes = stock.numeric_time_series.get(self._ts_close_alias)
            if highs is None or lows is None or closes is None:
                stock.set_numeric_time_series(self._k_alias, self._k_ts_def, [])
                stock.set_numeric_time_series(self._d_alias, self._d_ts_def, [])
                stock.set_numeric_metric(self._k_alias, self._k_ts_def, None)
                stock.set_numeric_metric(self._d_alias, self._d_ts_def, None)
                continue
            k_series, d_series = stochastic_oscillator(
                highs, lows, closes, self._k_period, self._d_period
            )
            # None entries can't sit in a list[float]; pad with NaN so the
            # full bar grid is preserved.  Latest scalar mirrors into
            # numeric_metrics under the same alias.
            stock.set_numeric_time_series(
                self._k_alias,
                self._k_ts_def,
                [v if v is not None else float("nan") for v in k_series],
            )
            stock.set_numeric_time_series(
                self._d_alias,
                self._d_ts_def,
                [v if v is not None else float("nan") for v in d_series],
            )
            stock.set_numeric_metric(
                self._k_alias,
                self._k_ts_def,
                k_series[-1] if k_series else None,
            )
            stock.set_numeric_metric(
                self._d_alias,
                self._d_ts_def,
                d_series[-1] if d_series else None,
            )
        logger.info(
            "StochasticOscillatorDecorator %r (k=%d, d=%d): %d ticker(s)",
            self._name,
            self._k_period,
            self._d_period,
            len(kb.stock_data),
        )
        return kb


# ── Relative Strength Index (Wilder's smoothing) ─────────────────────


def relative_strength_index(
    closes: Sequence[float],
    period: int,
) -> list[float | None]:
    """Compute Wilder's Relative Strength Index series."""
    if period < 1:
        msg = f"RSI period must be >= 1; got {period}"
        raise ValueError(msg)
    n = len(closes)
    if n < period + 1:
        return [None] * n
    arr = np.asarray(closes, dtype=np.float64)
    deltas = np.diff(arr)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    def _rsi_from(g: float, ll: float) -> float | None:
        if g == 0.0 and ll == 0.0:
            return None
        if ll == 0.0:
            return 100.0
        rs = g / ll
        return 100.0 - 100.0 / (1.0 + rs)

    avg_gain = float(gains[:period].mean())
    avg_loss = float(losses[:period].mean())
    rsi: list[float | None] = [None] * n
    rsi[period] = _rsi_from(avg_gain, avg_loss)
    for i in range(period + 1, n):
        gain = float(gains[i - 1])
        loss = float(losses[i - 1])
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        rsi[i] = _rsi_from(avg_gain, avg_loss)
    return rsi


class RelativeStrengthIndexDecorator(KnowledgeBaseDecorator):
    """Compute Wilder's RSI and store as a time series + latest scalar.

    Writes the **full** RSI series under ``stock.numeric_time_series[alias]``
    and the latest value under ``stock.numeric_metrics[alias]``.
    Registers ``alias_map[alias]`` against a derived TimeSeriesDef
    cloned from the source close series.
    """

    def __init__(
        self,
        name: str,
        ts_close_alias: str,
        period: int,
        close_ts_def: TimeSeriesDef,
        *upstream: KnowledgeBaseDecorator,
    ):
        super().__init__(upstream)
        if period < 1:
            msg = f"RSI period must be >= 1; got {period}"
            raise ValueError(msg)
        self._name = name
        self._ts_close_alias = ts_close_alias
        self._period = period
        self._derived_ts_def = TimeSeriesDef(
            name=close_ts_def.name,
            span=close_ts_def.span,
            start=close_ts_def.start,
            end=close_ts_def.end,
            multiplier=close_ts_def.multiplier,
            tag=name,
        )

    def build_kb(self) -> KnowledgeBase:
        kb = self._build_from_upstream_or_error()
        for stock in kb.stock_data.values():
            closes = stock.numeric_time_series.get(self._ts_close_alias)
            if closes is None:
                stock.set_numeric_time_series(self._name, self._derived_ts_def, [])
                stock.set_numeric_metric(self._name, self._derived_ts_def, None)
                continue
            rsi_series = relative_strength_index(closes, self._period)
            stock.set_numeric_time_series(
                self._name,
                self._derived_ts_def,
                [v if v is not None else float("nan") for v in rsi_series],
            )
            stock.set_numeric_metric(
                self._name,
                self._derived_ts_def,
                rsi_series[-1] if rsi_series else None,
            )
        logger.info(
            "RelativeStrengthIndexDecorator %r (period=%d): %d ticker(s)",
            self._name,
            self._period,
            len(kb.stock_data),
        )
        return kb


# ── Simple Moving Average ────────────────────────────────────────────


def simple_moving_average(
    values: Sequence[float],
    period: int,
) -> list[float | None]:
    """Compute the simple moving average series.

    Bars whose window has fewer than ``period`` observations, or whose
    window contains a ``NaN`` (which is how the pipeline encodes the
    warmup of upstream indicators), produce ``None``.  This keeps SMA
    composable with other indicators: stacking ``SMA smoothed rsi14 5``
    on top of an RSI series still yields a usable scalar in the tail.
    """
    if period < 1:
        msg = f"SMA period must be >= 1; got {period}"
        raise ValueError(msg)
    n = len(values)
    if n == 0:
        return []
    arr = np.asarray(values, dtype=np.float64)
    sma: list[float | None] = [None] * n
    for i in range(period - 1, n):
        window = arr[i - period + 1 : i + 1]
        if bool(np.isnan(window).any()):
            sma[i] = None
            continue
        sma[i] = float(window.mean())
    return sma


class SimpleMovingAverageDecorator(KnowledgeBaseDecorator):
    """Compute SMA and store as a time series + latest scalar.

    Works on any registered alias — raw closes, derived spreads, or
    other indicator outputs.
    """

    def __init__(
        self,
        name: str,
        ts_alias: str,
        period: int,
        source_ts_def: TimeSeriesDef,
        *upstream: KnowledgeBaseDecorator,
    ):
        super().__init__(upstream)
        if period < 1:
            msg = f"SMA period must be >= 1; got {period}"
            raise ValueError(msg)
        self._name = name
        self._ts_alias = ts_alias
        self._period = period
        self._derived_ts_def = TimeSeriesDef(
            name=source_ts_def.name,
            span=source_ts_def.span,
            start=source_ts_def.start,
            end=source_ts_def.end,
            multiplier=source_ts_def.multiplier,
            tag=name,
        )

    def build_kb(self) -> KnowledgeBase:
        kb = self._build_from_upstream_or_error()
        for stock in kb.stock_data.values():
            series = stock.numeric_time_series.get(self._ts_alias)
            if series is None:
                stock.set_numeric_time_series(self._name, self._derived_ts_def, [])
                stock.set_numeric_metric(self._name, self._derived_ts_def, None)
                continue
            sma_series = simple_moving_average(series, self._period)
            stock.set_numeric_time_series(
                self._name,
                self._derived_ts_def,
                [v if v is not None else float("nan") for v in sma_series],
            )
            stock.set_numeric_metric(
                self._name,
                self._derived_ts_def,
                sma_series[-1] if sma_series else None,
            )
        logger.info(
            "SimpleMovingAverageDecorator %r (period=%d): %d ticker(s)",
            self._name,
            self._period,
            len(kb.stock_data),
        )
        return kb
