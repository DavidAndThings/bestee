from __future__ import annotations

import ast
import logging
import operator
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import polars as pl
from great_tables import GT

from bestee.stocks import columns as cols
from bestee.stocks.financials import build_financials_df
from bestee.stocks.market import get_time_series
from bestee.stocks.models import FinancialMetric, TimeSeriesDef
from bestee.stocks.tickers import get_all_tickers_df, get_ticker_details_df

logger = logging.getLogger(__name__)

type Command = Sequence[str]


class NoUpstreamError(Exception):
    pass


class TimeSeriesNotAvailableError(Exception):
    pass


class TableDecorator(ABC):
    """Base class for pipeline stages that produce a tabular result.

    Stages chain internally as DataFrame → DataFrame via :meth:`_build_df`
    so the pipeline avoids round-tripping through Great Tables between
    every step.  Each stage's :meth:`build` wraps the final DataFrame in
    a styled :class:`GT` for end-user display.
    """

    def __init__(self, upstream_decorators: Sequence[TableDecorator] | None = None):
        self._upstream = upstream_decorators

    def _build_from_upstream(self) -> pl.DataFrame:
        """Return the combined upstream DataFrame.

        Default supports the common single-upstream case by delegating
        to that upstream's :meth:`_build_df`.  Subclasses with multiple
        upstreams (fan-in) override this to combine them.

        Callers should prefer :meth:`_build_from_upstream_or_error`,
        which checks for an empty / missing upstream chain first; this
        method assumes ``self._upstream`` is a non-empty sequence.
        """
        assert self._upstream is not None
        if len(self._upstream) == 1:
            return self._upstream[0]._build_df()
        msg = (
            f"{type(self).__name__} has {len(self._upstream)} upstreams; "
            "override _build_from_upstream to combine them."
        )
        raise NotImplementedError(msg)

    def _build_from_upstream_or_error(self) -> pl.DataFrame:
        """Return the combined upstream DataFrame, or raise if none."""
        if self._upstream is None:
            raise NoUpstreamError("No upstream decorator")
        return self._build_from_upstream()

    @abstractmethod
    def _build_df(self) -> pl.DataFrame:
        """Return this stage's output as a Polars DataFrame (no styling)."""

    @abstractmethod
    def build(self) -> GT:
        """Return this stage's output as a styled Great Tables object."""

    def get_time_series_from_upstream(
        self, ticker: str, ts_def: TimeSeriesDef
    ) -> Sequence[float]:
        """Walk the upstream(s) looking for one that can serve *ts_def*.

        Each upstream gets a try; ones that raise
        :class:`TimeSeriesNotAvailableError` are skipped.  If none
        serve it, propagates the same exception.
        """
        for decorator in self._upstream or []:
            try:
                return decorator.get_time_series(ticker, ts_def)
            except TimeSeriesNotAvailableError:
                continue
        raise TimeSeriesNotAvailableError

    def get_time_series(self, ticker: str, ts_def: TimeSeriesDef) -> Sequence[float]:
        """Return the time series for *ticker*.

        Default delegates to upstream via
        :meth:`get_time_series_from_upstream`.  Stages that actually
        produce series (e.g. :class:`TimeSeriesCacheDecorator`,
        :class:`TimeSeriesDerivedDecorator`) override this.
        """
        return self.get_time_series_from_upstream(ticker, ts_def)


class AssetScopeDecorator(TableDecorator):
    def __init__(
        self,
        ticker_type: str,
    ):
        super().__init__(None)
        self.ticker_type = ticker_type

    def _build_df(self) -> pl.DataFrame:
        symbols_df = get_all_tickers_df(ticker_type=self.ticker_type, active=True)
        symbols: list[str] = symbols_df[cols.TICKER].to_list()
        return get_ticker_details_df(symbols)

    def build(self) -> GT:
        df = self._build_df()
        return (
            GT(df)
            .tab_header(
                title="Ticker Details",
                subtitle=f"{df.height} tickers from the Massive API",
            )
            .sub_missing(missing_text="—")
        )


class SameSICategoryDecorator(TableDecorator):
    def __init__(self, ticker: str, *upstream: TableDecorator):
        super().__init__(upstream)
        self.ticker = ticker

    def _build_df(self) -> pl.DataFrame:
        df = self._build_from_upstream_or_error()

        # Locate the target ticker's SIC code.
        target_rows = df.filter(pl.col(cols.TICKER) == self.ticker)
        if target_rows.height == 0:
            msg = f"Ticker {self.ticker!r} not found in upstream table"
            raise ValueError(msg)
        target_sic = target_rows[cols.SIC_CODE].item(0)
        if target_sic is None:
            msg = f"Ticker {self.ticker!r} has no SIC code in upstream table"
            raise ValueError(msg)

        # Filter the upstream to rows sharing the same SIC code.
        return df.filter(pl.col(cols.SIC_CODE) == target_sic)

    def build(self) -> GT:
        filtered = self._build_df()
        target_sic = filtered.filter(pl.col(cols.TICKER) == self.ticker)[
            cols.SIC_CODE
        ].item(0)
        return (
            GT(filtered)
            .tab_header(
                title="Tickers in Same SIC Category",
                subtitle=(
                    f"SIC {target_sic} · same category as {self.ticker} · "
                    f"{filtered.height} tickers"
                ),
            )
            .sub_missing(missing_text="—")
        )


class PickTickersDecorator(TableDecorator):
    """Config-only marker that registers a ticker list for a downstream
    :class:`FinancialsDecorator` to fetch metrics for."""

    def __init__(self, tickers: Sequence[str], *upstream: TableDecorator):
        super().__init__(upstream)
        self.tickers = tickers

    def _build_df(self) -> pl.DataFrame:
        df = self._build_from_upstream_or_error()
        return df.filter(pl.col(cols.TICKER).is_in(self.tickers))

    def build(self) -> GT:
        return GT(self._build_df())


class AppendTablesDecorator(TableDecorator):
    """Append multiple upstream tables vertically into one DataFrame."""

    def __init__(self, *upstream: TableDecorator):
        super().__init__(upstream)

    def _build_df(self) -> pl.DataFrame:
        assert self._upstream is not None, (
            "AppendTablesDecorator requires upstream decorators"
        )
        dfs = [upstream._build_df() for upstream in self._upstream]
        return pl.concat(dfs)

    def build(self) -> GT:
        return GT(self._build_df())


class FinancialMetricCacheDecorator(TableDecorator):
    """Config-only marker that registers one ``(name, FinancialMetric)``
    pair for a downstream :class:`FinancialsDecorator` to batch-fetch.

    It's a :class:`TableDecorator` subclass purely so it can be passed
    alongside the table-providing upstream in
    ``FinancialsDecorator(*upstream)``; it doesn't itself build a
    DataFrame.  Calling :meth:`_build_df` or :meth:`build` on it
    directly is a programmer error.
    """

    def __init__(
        self, metric_name: str, metric: FinancialMetric, *upstream: TableDecorator
    ):
        super().__init__(upstream)
        self.metric_name = metric_name
        self.metric = metric

    def _build_df(self) -> pl.DataFrame:
        msg = (
            "FinancialMetricCacheDecorator is a config-only stage; pass it "
            "as an upstream to a FinancialsDecorator instead of building it "
            "directly."
        )
        raise NoUpstreamError(msg)

    def build(self) -> GT:
        return GT(self._build_df())


class FinancialsDecorator(TableDecorator):
    """Add named financial-metric columns from a single batched API call.

    Constructed with a mix of upstreams: exactly one
    table-providing decorator (the chain head) plus zero or more
    :class:`FinancialMetricCacheDecorator` instances that declare which
    ``(name, FinancialMetric)`` pairs to fetch.  All metrics are
    fetched in one ``build_financials_df`` round-trip and joined to
    the upstream table, with each column renamed from the SDK's
    default ``"<base_label> (FY… Q…)"`` form to the user's name.
    """

    def __init__(self, *upstream: TableDecorator):
        super().__init__(upstream)

    @property
    def _named_metrics(self) -> dict[str, FinancialMetric]:
        assert self._upstream is not None, (
            "upstream_decorators must be provided for the FinancialsDecorator"
        )
        named_metrics: dict[str, FinancialMetric] = {}
        for decorator in self._upstream:
            if isinstance(decorator, FinancialMetricCacheDecorator):
                named_metrics[decorator.metric_name] = decorator.metric
        return named_metrics

    @property
    def metrics(self) -> list[FinancialMetric]:
        """The bare metric requests (without names) — useful for tests
        that just want to see what got registered."""
        return list(self._named_metrics.values())

    def _build_from_upstream(self) -> pl.DataFrame:
        """Pick the lone table upstream out of the mix and build it.

        The cache upstreams are config-only and don't contribute to
        the table — they're consumed via :attr:`_named_metrics`.
        """
        assert self._upstream is not None
        table_upstreams = [
            d
            for d in self._upstream
            if not isinstance(d, FinancialMetricCacheDecorator)
        ]
        if len(table_upstreams) != 1:
            msg = (
                "FinancialsDecorator expects exactly one non-cache upstream "
                f"to provide the table; got {len(table_upstreams)}."
            )
            raise ValueError(msg)
        return table_upstreams[0]._build_df()

    def _build_df(self) -> pl.DataFrame:
        upstream_df = self._build_from_upstream_or_error()

        # No metrics requested — pass the upstream through untouched
        # so we don't make a wasted API call.
        if not self._named_metrics:
            return upstream_df

        tickers: list[str] = upstream_df[cols.TICKER].to_list()
        ordered_names = list(self._named_metrics)
        ordered_metrics = [self._named_metrics[n] for n in ordered_names]
        metrics_df = build_financials_df(tickers=tickers, metrics=ordered_metrics)
        # Re-label each column from the SDK's auto-generated metric.label
        # to the user-supplied name.
        rename_map = {self._named_metrics[name].label: name for name in ordered_names}
        metrics_df = metrics_df.rename(rename_map)
        # Polars DataFrames support .join(); GT objects do not.
        return upstream_df.join(metrics_df, on=cols.TICKER, how="inner")

    def build(self) -> GT:
        df = self._build_df()
        return (
            GT(df)
            .tab_header(
                title="Tickers with Financial Metrics",
                subtitle=(
                    f"{df.height} tickers · {len(self._named_metrics)} "
                    "financial metric(s)"
                ),
            )
            .sub_missing(missing_text="—")
        )


class ComputedMetricDecorator(TableDecorator):
    """Add a computed-metric column derived from a Python-arithmetic expression.

    DSL: ``COMPUTED_METRIC <name> <expression>``.  *expression* is a
    plain Python arithmetic expression whose identifiers reference
    upstream columns — typically those added by ``FINANCIAL_METRIC``
    lines (each labelled with its user-supplied name) or earlier
    ``COMPUTED_METRIC`` lines.  Allowed operators: ``+ - * / ** %``,
    unary ``±`` and parens; numeric literals are fine.

    Example::

        # After FINANCIAL_METRIC fm2 NET_INCOME 2025 4
        #       FINANCIAL_METRIC fm3 TOTAL_ASSETS 2025 4
        #       FINANCIAL_METRIC fm4 TOTAL_ASSETS 2025 3
        deco = ComputedMetricDecorator(
            name="cm1",
            expression="(fm2 * 2) / (fm3 + fm4)",
            upstream=fin_decorator,
        )

    Evaluation parses with :mod:`ast` rather than ``eval``, so no
    arbitrary code can run; the AST is validated against an allow-list
    of node types at construction time.  Per-ticker, ``None``
    propagates cleanly for any missing operand, divide-by-zero, or
    modulo-by-zero so a single bad data point doesn't sink the column.
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

    def __init__(self, name: str, expression: str, *upstream: TableDecorator):
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

    # ── Parsing helpers ──────────────────────────────────────────────

    @classmethod
    def _collect_names(cls, node: ast.AST) -> set[str]:
        return {sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name)}

    @classmethod
    def _validate(cls, node: ast.AST) -> None:
        """Walk the AST and raise on anything outside the allowed subset."""
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

    # ── Safe AST evaluation ──────────────────────────────────────────

    @classmethod
    def _evaluate(
        cls,
        node: ast.AST,
        env: dict[str, float | None],
    ) -> float | None:
        """Recursively evaluate the AST using *env* for placeholders.

        Returns *None* if any referenced value is missing or if a
        division-by-zero is encountered, so a single missing data point
        propagates as a missing result rather than an exception.
        """
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
            operand = cls._evaluate(node.operand, env)
            if operand is None:
                return None
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return operand
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

    # ── Build ────────────────────────────────────────────────────────

    def _build_df(self) -> pl.DataFrame:
        upstream_df = self._build_from_upstream_or_error()

        missing = [n for n in self._referenced_names if n not in upstream_df.columns]
        if missing:
            msg = (
                f"COMPUTED_METRIC {self._name!r} expression references "
                f"unknown name(s) {missing!r}. "
                f"Available columns: {upstream_df.columns}."
            )
            raise ValueError(msg)

        computed: list[float | None] = []
        for row in upstream_df.iter_rows(named=True):
            env: dict[str, float | None] = {}
            for n in self._referenced_names:
                raw = row.get(n)
                env[n] = float(raw) if raw is not None else None
            try:
                value = self._evaluate(self._ast_tree, env)
            except TypeError, ZeroDivisionError, OverflowError:
                value = None
            computed.append(value)

        logger.info(
            "Computed %r for %d tickers (%d non-null)",
            self._name,
            len(computed),
            sum(1 for v in computed if v is not None),
        )

        return upstream_df.with_columns(
            pl.Series(name=self._name, values=computed, dtype=pl.Float64)
        )

    def build(self) -> GT:
        df = self._build_df()
        return (
            GT(df)
            .tab_header(
                title=f"With Computed Metric: {self._name}",
                subtitle=f"{df.height} tickers",
            )
            .sub_missing(missing_text="—")
        )


class TimeSeriesCacheDecorator(TableDecorator):
    """Side-channel stage that memoises one ``TimeSeriesDef``'s fetches.

    Inserted between two stages, it leaves the tabular pipeline
    unchanged (``_build_df`` and ``build`` pass straight through to
    the upstream), but intercepts the chain-walking
    :meth:`TableDecorator.get_time_series` call so downstream
    consumers share a single network round-trip per ticker.

    A cache only serves the ``TimeSeriesDef`` it was created with —
    requests for other definitions fall through to upstream stages.
    """

    def __init__(self, ts_def: TimeSeriesDef, *upstream: TableDecorator):
        super().__init__(upstream)
        self._ts_def = ts_def
        self._cache: dict[str, Sequence[float]] = {}

    def _build_df(self) -> pl.DataFrame:
        return self._build_from_upstream_or_error()

    def build(self) -> GT:
        return GT(self._build_df())

    def get_time_series(self, ticker: str, ts_def: TimeSeriesDef) -> Sequence[float]:
        if ts_def != self._ts_def:
            return self.get_time_series_from_upstream(ticker, ts_def)
        if ticker in self._cache:
            return self._cache[ticker]
        values = get_time_series(ticker, self._ts_def)
        self._cache[ticker] = values
        return values


# ── Scalar reducers over a time series ───────────────────────────────


type TimeSeriesScalarMetric = Callable[[Sequence[float]], float | None]
"""A function that reduces a time series to one scalar (or ``None``)."""


def _rsquared_trend(series: Sequence[float]) -> float | None:
    """R² of a least-squares linear fit of *series* against its index.

    Equivalent to the squared Pearson correlation between the bar
    index (0, 1, 2, …) and the value at that bar.  A series that
    trends perfectly linearly scores 1.0; a flat or noisy series
    scores near 0.

    Returns ``None`` for series too short to fit (< 2 points) or with
    zero variance (constant series — R² is undefined).
    """
    arr = np.asarray(series, dtype=np.float64)
    if arr.size < 2:
        return None
    # Short-circuit on a constant series — np.corrcoef would emit a
    # divide-by-zero RuntimeWarning and return NaN here.
    if arr.var() == 0.0:
        return None
    xs = np.arange(arr.size, dtype=np.float64)
    r = np.corrcoef(xs, arr)[0, 1]
    if not np.isfinite(r):
        return None
    return float(r * r)


_TIME_SERIES_METRICS: dict[str, TimeSeriesScalarMetric] = {
    "RSquared": _rsquared_trend,
}


def register_time_series_metric(name: str, fn: TimeSeriesScalarMetric) -> None:
    """Register a scalar reducer so it's callable from the DSL or directly.

    Once registered, ``TimeSeriesMetricDecorator(metric=name, ...)``
    and ``TIME_SERIES_METRIC <name> <ts>`` in the builder DSL both
    pick *fn* up by name.  Names are matched case-sensitively to
    match the DSL surface.

    Args:
        name: Public name (and resulting column header).
        fn: ``(Sequence[float]) -> float | None`` reducer.  Return
            ``None`` to signal "no value" — the column dtype is
            ``Float64`` so nulls flow through naturally.
    """
    _TIME_SERIES_METRICS[name] = fn


def available_time_series_metrics() -> list[str]:
    """Return the names of all registered scalar reducers, sorted."""
    return sorted(_TIME_SERIES_METRICS)


class TimeSeriesDerivedDecorator(TableDecorator):
    """Define a new time series as arithmetic over previously-declared ones.

    Drop-in pass-through for the tabular pipeline (``_build_df`` and
    ``build`` delegate to the upstream), but exposes a fresh series
    via the chain-walking :meth:`TableDecorator.get_time_series` hook.
    Downstream stages (typically a :class:`TimeSeriesMetricDecorator`)
    read the derived series by its own :class:`TimeSeriesDef` exactly
    like any other producer.

    Expressions are restricted to ``+``, ``-``, ``*``, ``/``, unary
    minus, and numeric constants over the named operand series.
    Element-wise evaluation goes through numpy so scalar broadcasting
    works (``2*ts1 - ts2``).  Operand series must align bar-by-bar —
    when they don't, numpy raises during the first ticker's fetch and
    the error surfaces with the offending pair.
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
        derived_ts_def: TimeSeriesDef,
        expression: str,
        operand_ts_defs: Mapping[str, TimeSeriesDef],
        *upstream_decorators: TableDecorator,
    ):
        super().__init__(upstream_decorators)
        self._ts_def = derived_ts_def
        self._expression = expression
        try:
            self._ast_tree = ast.parse(expression, mode="eval").body
        except SyntaxError as err:
            msg = (
                f"TIME_SERIES_DERIVED expression {expression!r} is not "
                f"valid Python syntax: {err.msg}"
            )
            raise ValueError(msg) from err
        # Validate AST early so a typo doesn't only surface mid-build.
        self._validate(self._ast_tree)
        self._operand_names = sorted(self._collect_names(self._ast_tree))
        missing = [n for n in self._operand_names if n not in operand_ts_defs]
        if missing:
            msg = (
                f"TIME_SERIES_DERIVED expression {expression!r} refers to "
                f"undefined names {missing!r}. "
                f"Defined: {sorted(operand_ts_defs)}."
            )
            raise ValueError(msg)
        self._operand_ts_defs: dict[str, TimeSeriesDef] = {
            n: operand_ts_defs[n] for n in self._operand_names
        }
        self._cache: dict[str, list[float]] = {}

    @classmethod
    def _collect_names(cls, node: ast.AST) -> set[str]:
        return {sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name)}

    @classmethod
    def _validate(cls, node: ast.AST) -> None:
        """Walk the AST and raise on anything outside the allowed subset."""
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
                # AST visits operators and load/store contexts on their
                # own; the BinOp/UnaryOp/Name checks above cover their
                # enclosing nodes, so a bare visit here is fine.
                continue
            msg = (
                "TIME_SERIES_DERIVED expression contains an unsupported "
                f"construct: {type(sub).__name__}"
            )
            raise ValueError(msg)

    def _build_df(self) -> pl.DataFrame:
        return self._build_from_upstream_or_error()

    def build(self) -> GT:
        return GT(self._build_df())

    def get_time_series(self, ticker: str, ts_def: TimeSeriesDef) -> Sequence[float]:
        if ts_def != self._ts_def:
            return self.get_time_series_from_upstream(ticker, ts_def)
        if ticker in self._cache:
            return self._cache[ticker]
        if self._upstream is None:
            raise NoUpstreamError(
                "TimeSeriesDerivedDecorator needs an upstream to fetch operands"
            )
        operands: dict[str, np.ndarray] = {
            name: np.asarray(
                self.get_time_series_from_upstream(ticker, self._operand_ts_defs[name]),
                dtype=np.float64,
            )
            for name in self._operand_names
        }
        result = self._evaluate(self._ast_tree, operands)
        values: list[float] = np.asarray(result, dtype=np.float64).tolist()
        self._cache[ticker] = values
        return values

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
        # Should never hit this branch — _validate ran at construction
        # time — but keep the message friendly if it ever does.
        msg = (
            "TIME_SERIES_DERIVED expression contains an unsupported "
            f"construct at runtime: {type(node).__name__}"
        )
        raise ValueError(msg)


class TimeSeriesMetricDecorator(TableDecorator):
    """Add a named scalar column derived from a per-ticker time series.

    For every ticker in the upstream table, pulls the series via
    :meth:`TableDecorator.get_time_series` (which walks the chain to a
    producer — typically :class:`TimeSeriesCacheDecorator`) and
    applies a named scalar reducer from
    :data:`_TIME_SERIES_METRICS` (e.g. ``"RSquared"``).  The reducer's
    return value lands in a new ``Float64`` column carrying the
    user-supplied *name*, mirroring how ``FINANCIAL_METRIC`` and
    ``COMPUTED_METRIC`` lines label their output.

    Register additional reducers via
    :func:`register_time_series_metric`.  Wrap with a
    :class:`TimeSeriesCacheDecorator` upstream so the network is hit
    once per ticker even when several metric columns share a series.
    """

    def __init__(
        self,
        ts_def: TimeSeriesDef,
        name: str,
        metric: str,
        *upstream_decorators: TableDecorator,
    ):
        super().__init__(upstream_decorators)
        if metric not in _TIME_SERIES_METRICS:
            msg = (
                f"Unknown time-series metric {metric!r}. "
                f"Available: {available_time_series_metrics()}."
            )
            raise ValueError(msg)
        self._ts_def = ts_def
        self._name = name
        self._metric_name = metric
        self._metric_fn = _TIME_SERIES_METRICS[metric]

    def _build_df(self) -> pl.DataFrame:
        upstream_df = self._build_from_upstream_or_error()
        tickers: list[str] = upstream_df[cols.TICKER].to_list()
        values: list[float | None] = []
        for ticker in tickers:
            series = self.get_time_series(ticker, self._ts_def)
            values.append(self._metric_fn(series))
        logger.info(
            "Computed %r (metric=%r) for %d tickers (%d non-null)",
            self._name,
            self._metric_name,
            len(values),
            sum(1 for v in values if v is not None),
        )
        return upstream_df.with_columns(
            pl.Series(name=self._name, values=values, dtype=pl.Float64)
        )

    def build(self) -> GT:
        df = self._build_df()
        return (
            GT(df)
            .tab_header(
                title=(f"With Time-Series Metric: {self._name} ({self._metric_name})"),
                subtitle=f"{df.height} tickers",
            )
            .sub_missing(missing_text="—")
        )
