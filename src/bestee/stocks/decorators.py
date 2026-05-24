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
from bestee.stocks.models import (
    FinancialMetric,
    Metric,
    TimeSeriesDef,
    TimeSeriesName,
    TimeSeriesSpan,
)
from bestee.stocks.tickers import get_all_tickers_df, get_ticker_details_df

logger = logging.getLogger(__name__)

type Command = Sequence[str]


class NoUpstreamError(Exception):
    pass


class ProcessingLevelError(Exception):
    pass


class TableDecorator(ABC):
    """Base class for pipeline stages that produce a tabular result.

    Stages chain internally as DataFrame → DataFrame via :meth:`_build_df`
    so the pipeline avoids round-tripping through Great Tables between
    every step.  Each stage's :meth:`build` wraps the final DataFrame in
    a styled :class:`GT` for end-user display.
    """

    def __init__(self, upstream_decorator: TableDecorator | None = None):
        self._upstream = upstream_decorator

    def _build_upstream_df(self) -> pl.DataFrame:
        """Return the upstream stage's DataFrame, or raise if none."""
        if self._upstream is None:
            raise NoUpstreamError("No upstream decorator")
        return self._upstream._build_df()

    @abstractmethod
    def _build_df(self) -> pl.DataFrame:
        """Return the stage's output as a Polars DataFrame (no styling)."""

    @abstractmethod
    def build(self) -> GT:
        """Return the stage's output as a styled Great Tables object."""

    def _get_time_series(self, ticker: str, ts_def: TimeSeriesDef) -> Sequence[float]:
        """Return the time series for *ticker* if this stage knows how.

        Stages that don't supply time-series data should leave the
        default — it raises :class:`NotImplementedError`, which lets
        :meth:`get_time_series` fall back to the upstream stage.
        """
        raise NotImplementedError

    def get_time_series(self, ticker: str, ts_def: TimeSeriesDef) -> Sequence[float]:
        """Walk the decorator chain until a stage can serve the series.

        Each stage gets a chance to provide the data via
        :meth:`_get_time_series`; on :class:`NotImplementedError` the
        call delegates upstream.  If no stage in the chain implements
        it, the original :class:`NotImplementedError` propagates.
        """
        try:
            return self._get_time_series(ticker, ts_def)
        except NotImplementedError:
            if self._upstream is not None:
                return self._upstream.get_time_series(ticker, ts_def)
            raise


class TickerSummaryDecorator(TableDecorator):
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
    def __init__(self, ticker: str, upstream: TableDecorator):
        super().__init__(upstream)
        self.ticker = ticker

    def _build_df(self) -> pl.DataFrame:
        df = self._build_upstream_df()

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


class FinancialsDecorator(TableDecorator):
    """Add named financial-metric columns from a single batched API call.

    The DSL form is ``FINANCIAL_METRIC <name> <metric> <year> <quarter>``
    where ``<name>`` is the user-supplied column header that downstream
    ``COMPUTED_METRIC`` lines reference.  All metrics added before
    :meth:`_build_df` are fetched in one round-trip and then renamed
    from the SDK's default ``"<base_label> (FY… Q…)"`` form to the
    user's name.
    """

    def __init__(self, upstream: TableDecorator):
        super().__init__(upstream)
        # Order matters for stable column ordering in the joined frame.
        self._named_metrics: dict[str, FinancialMetric] = {}

    @property
    def metrics(self) -> list[FinancialMetric]:
        """The bare metric requests (without names) — useful for tests
        that just want to see what got registered."""
        return list(self._named_metrics.values())

    def add_metric(self, name: str, metric: FinancialMetric) -> None:
        """Register *metric* under *name*.

        Raises:
            ValueError: If *name* is already defined on this decorator
                (the DSL forbids duplicate ``FINANCIAL_METRIC`` names so
                downstream references resolve unambiguously).
        """
        if name in self._named_metrics:
            msg = f"FINANCIAL_METRIC name {name!r} is already defined"
            raise ValueError(msg)
        self._named_metrics[name] = metric

    def _build_df(self) -> pl.DataFrame:
        upstream_df = self._build_upstream_df()

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

    def __init__(self, name: str, expression: str, upstream: TableDecorator):
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
        upstream_df = self._build_upstream_df()

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
            except (TypeError, ZeroDivisionError, OverflowError):
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

    def __init__(
        self, ts_def: TimeSeriesDef, upstream_decorator: TableDecorator | None = None
    ):
        super().__init__(upstream_decorator)
        self._ts_def = ts_def
        self._cache: dict[str, Sequence[float]] = {}

    def _build_df(self) -> pl.DataFrame:
        return self._build_upstream_df()

    def build(self) -> GT:
        if self._upstream is None:
            raise NoUpstreamError("TimeSeriesCacheDecorator has no upstream to render")
        return self._upstream.build()

    def _get_time_series(self, ticker: str, ts_def: TimeSeriesDef) -> Sequence[float]:
        if ts_def != self._ts_def:
            raise NotImplementedError
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
        upstream_decorator: TableDecorator,
    ):
        super().__init__(upstream_decorator)
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
        return self._build_upstream_df()

    def build(self) -> GT:
        if self._upstream is None:
            raise NoUpstreamError(
                "TimeSeriesDerivedDecorator has no upstream to render"
            )
        return self._upstream.build()

    def _get_time_series(self, ticker: str, ts_def: TimeSeriesDef) -> Sequence[float]:
        if ts_def != self._ts_def:
            raise NotImplementedError
        if ticker in self._cache:
            return self._cache[ticker]
        if self._upstream is None:
            raise NoUpstreamError(
                "TimeSeriesDerivedDecorator needs an upstream to fetch operands"
            )
        operands: dict[str, np.ndarray] = {
            name: np.asarray(
                self._upstream.get_time_series(ticker, self._operand_ts_defs[name]),
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
        upstream_decorator: TableDecorator,
        *,
        name: str,
        metric: str,
    ):
        super().__init__(upstream_decorator)
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
        upstream_df = self._build_upstream_df()
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


_NUM_PROCESSING_LEVELS = 4


def decorator_builder(commands: Sequence[Command]) -> TableDecorator:
    """Build a chained :class:`TableDecorator` from a sequence of commands.

    Commands are processed in four logical phases so that ordering
    across phases in the input is irrelevant (ordering *within* the
    TIME_SERIES phase still matters, since references resolve by name):

    * **Level 0** — ``STOCKS``, ``SAME_SIC_CATEGORY_AS``
      (set up the base table and any filtering)
    * **Level 1** — ``FINANCIAL_METRIC``
      (accumulated into a single :class:`FinancialsDecorator` so the
      financial-statement endpoints are called once per period)
    * **Level 2** — ``COMPUTED_METRIC``
      (each command becomes its own :class:`ComputedMetricDecorator`,
      chained on top of the previous one)
    * **Level 3** — ``TIME_SERIES`` / ``TIME_SERIES_DERIVED`` /
      ``TIME_SERIES_METRIC``
      (``TIME_SERIES`` declarations create a
      :class:`TimeSeriesCacheDecorator` and register the series under
      a name; ``TIME_SERIES_DERIVED`` builds a new named series as
      arithmetic over existing ones; ``TIME_SERIES_METRIC`` references
      any of those names to attach a
      :class:`TimeSeriesMetricDecorator` column)

    Args:
        commands: Each command is a sequence of whitespace-separated
            tokens, where ``command[0]`` is the keyword.

    Returns:
        The outermost (final) decorator in the chain.

    Raises:
        ProcessingLevelError: If a command is unrecognized, or if a
            dependent command appears without its required upstream
            (e.g. ``COMPUTED_METRIC`` before any ``STOCKS``).
    """
    cmd_queue: list[Command] = [*commands]
    decorator: TableDecorator | None = None

    for level in range(_NUM_PROCESSING_LEVELS):
        decorator, cmd_queue = decorator_builder_one_pass(cmd_queue, decorator, level)

    if cmd_queue:
        unknown = [cmd[0] for cmd in cmd_queue]
        msg = f"Unrecognized command keyword(s): {unknown}"
        raise ProcessingLevelError(msg)

    if decorator is None:
        msg = "No decorator was built — commands list produced nothing"
        raise ProcessingLevelError(msg)

    return decorator


def decorator_builder_one_pass(
    commands: Sequence[Command],
    decorator: TableDecorator | None,
    processing_level: int,
) -> tuple[TableDecorator | None, list[Command]]:
    """Run a single processing pass over *commands*.

    Returns:
        The (possibly updated) ``decorator`` plus the list of commands
        that this level did not consume.
    """
    new_commands: list[Command] = []

    # Level 1 accumulates FINANCIAL_METRIC commands into a single
    # FinancialsDecorator so that downstream API calls are batched.
    fin_decorator: FinancialsDecorator | None = None
    # Level 3 needs to remember TimeSeriesDefs by name so a later
    # TIME_SERIES_METRIC in the same pass can reference them.
    ts_defs: dict[str, TimeSeriesDef] = {}

    for command in commands:
        match processing_level:
            case 0:
                status, decorator = command_processor_level_zero(command, decorator)
            case 1:
                status, decorator, fin_decorator = command_processor_level_one(
                    command, decorator, fin_decorator
                )
            case 2:
                status, decorator = command_processor_level_two(command, decorator)
            case 3:
                status, decorator, ts_defs = command_processor_level_three(
                    command, decorator, ts_defs
                )
            case _:
                msg = f"Invalid processing level: {processing_level}"
                raise ProcessingLevelError(msg)

        if status == 0:
            new_commands.append(command)

    return decorator, new_commands


def command_processor_level_zero(
    command: Command,
    decorator: TableDecorator | None,
) -> tuple[int, TableDecorator | None]:
    """Handle base-table commands (STOCKS, SAME_SIC_CATEGORY_AS)."""
    match command[0]:
        case "STOCKS":
            return 1, TickerSummaryDecorator(ticker_type="CS")
        case "SAME_SIC_CATEGORY_AS":
            if decorator is None:
                msg = "SAME_SIC_CATEGORY_AS requires an upstream (use STOCKS first)"
                raise ProcessingLevelError(msg)
            return 1, SameSICategoryDecorator(ticker=command[1], upstream=decorator)
        case _:
            return 0, decorator


def command_processor_level_one(
    command: Command,
    decorator: TableDecorator | None,
    fin_decorator: FinancialsDecorator | None,
) -> tuple[int, TableDecorator | None, FinancialsDecorator | None]:
    """Handle named ``FINANCIAL_METRIC`` commands.

    Form: ``FINANCIAL_METRIC <name> <metric> <fiscal_year> <fiscal_quarter>``.
    The first ``FINANCIAL_METRIC`` instantiates a shared
    :class:`FinancialsDecorator`; subsequent ones register more named
    metrics on the same decorator so a single API call covers all of
    them.  Downstream ``COMPUTED_METRIC`` lines reference the metric by
    *name* (e.g. ``fm1``).
    """
    match command[0]:
        case "FINANCIAL_METRIC":
            if len(command) != 5:
                msg = (
                    "FINANCIAL_METRIC requires 4 arguments: "
                    "<name> <metric> <fiscal_year> <fiscal_quarter>"
                )
                raise ProcessingLevelError(msg)
            if fin_decorator is None:
                if decorator is None:
                    msg = "FINANCIAL_METRIC requires an upstream (use STOCKS first)"
                    raise ProcessingLevelError(msg)
                fin_decorator = FinancialsDecorator(upstream=decorator)
                decorator = fin_decorator
            name, metric_token, year_token, quarter_token = command[1:5]
            try:
                metric_enum = Metric[metric_token]
            except KeyError as err:
                msg = (
                    f"Unknown FINANCIAL_METRIC metric: {metric_token!r}. "
                    f"Available: {sorted(m.name for m in Metric)}"
                )
                raise ProcessingLevelError(msg) from err
            try:
                fiscal_year = int(year_token)
                fiscal_quarter = int(quarter_token)
            except ValueError as err:
                msg = (
                    "FINANCIAL_METRIC fiscal_year and fiscal_quarter must be "
                    f"integers, got year={year_token!r} quarter={quarter_token!r}"
                )
                raise ProcessingLevelError(msg) from err
            try:
                fin_decorator.add_metric(
                    name=name,
                    metric=FinancialMetric(
                        metric=metric_enum,
                        fiscal_year=fiscal_year,
                        fiscal_quarter=fiscal_quarter,
                    ),
                )
            except ValueError as err:
                raise ProcessingLevelError(str(err)) from err
            return 1, decorator, fin_decorator
        case _:
            return 0, decorator, fin_decorator


def command_processor_level_two(
    command: Command,
    decorator: TableDecorator | None,
) -> tuple[int, TableDecorator | None]:
    """Handle named ``COMPUTED_METRIC`` commands.

    Form: ``COMPUTED_METRIC <name> <expression>``.  The expression is
    plain Python arithmetic over upstream column names (e.g. the
    ``fm…`` names declared by earlier ``FINANCIAL_METRIC`` lines, or
    other ``COMPUTED_METRIC`` names defined upstream in the same
    pipeline).
    """
    match command[0]:
        case "COMPUTED_METRIC":
            if decorator is None:
                msg = "COMPUTED_METRIC requires an upstream (use STOCKS first)"
                raise ProcessingLevelError(msg)
            if len(command) < 3:
                msg = (
                    "COMPUTED_METRIC requires at least 2 arguments: <name> <expression>"
                )
                raise ProcessingLevelError(msg)
            name = command[1]
            # Tokens 2.. are the whitespace-tokenised expression; rejoin
            # with single spaces — ast.parse is tolerant of whitespace.
            expression = " ".join(command[2:])
            try:
                metric_decorator = ComputedMetricDecorator(
                    name=name, expression=expression, upstream=decorator
                )
            except ValueError as err:
                raise ProcessingLevelError(str(err)) from err
            return 1, metric_decorator
        case _:
            return 0, decorator


def command_processor_level_three(
    command: Command,
    decorator: TableDecorator | None,
    ts_defs: dict[str, TimeSeriesDef],
) -> tuple[int, TableDecorator | None, dict[str, TimeSeriesDef]]:
    """Handle TIME_SERIES (cache), TIME_SERIES_DERIVED (synthesised),
    and TIME_SERIES_METRIC (column) commands.

    Within a single ``decorator_builder`` pass, ``TIME_SERIES`` lines
    register a named :class:`TimeSeriesDef` (and chain a
    :class:`TimeSeriesCacheDecorator` so the data is fetched once),
    ``TIME_SERIES_DERIVED`` lines define a new named series as
    arithmetic over previously-declared ones (chaining a
    :class:`TimeSeriesDerivedDecorator`), and ``TIME_SERIES_METRIC``
    lines reference any of those names to attach a
    :class:`TimeSeriesMetricDecorator` column.  Names disambiguate
    chain-walking lookups via the ``TimeSeriesDef.tag`` field.
    """
    match command[0]:
        case "TIME_SERIES":
            if decorator is None:
                msg = "TIME_SERIES requires an upstream (use STOCKS first)"
                raise ProcessingLevelError(msg)
            # name field start end span multiplier
            if len(command) != 7:
                msg = (
                    "TIME_SERIES requires 6 arguments: "
                    "<name> <ohlc_field> <start> <end> <span> <multiplier>"
                )
                raise ProcessingLevelError(msg)
            name, field, start, end, span_token, multiplier_token = command[1:7]
            if name in ts_defs:
                msg = f"TIME_SERIES name {name!r} is already defined"
                raise ProcessingLevelError(msg)
            try:
                ts_name = TimeSeriesName(field)
            except ValueError as err:
                msg = (
                    f"Unknown TIME_SERIES field {field!r}. "
                    f"Available: {sorted(n.value for n in TimeSeriesName)}"
                )
                raise ProcessingLevelError(msg) from err
            try:
                ts_span = TimeSeriesSpan(span_token)
            except ValueError as err:
                msg = (
                    f"Unknown TIME_SERIES span {span_token!r}. "
                    f"Available: {sorted(s.value for s in TimeSeriesSpan)}"
                )
                raise ProcessingLevelError(msg) from err
            try:
                multiplier = int(multiplier_token)
            except ValueError as err:
                msg = (
                    "TIME_SERIES multiplier must be an integer, "
                    f"got {multiplier_token!r}"
                )
                raise ProcessingLevelError(msg) from err
            ts_def = TimeSeriesDef(
                name=ts_name,
                span=ts_span,
                start=start,
                end=end,
                multiplier=multiplier,
                tag=name,
            )
            ts_defs = {**ts_defs, name: ts_def}
            return (
                1,
                TimeSeriesCacheDecorator(ts_def, upstream_decorator=decorator),
                ts_defs,
            )
        case "TIME_SERIES_DERIVED":
            if decorator is None:
                msg = "TIME_SERIES_DERIVED requires an upstream (use STOCKS first)"
                raise ProcessingLevelError(msg)
            if len(command) != 3:
                msg = "TIME_SERIES_DERIVED requires 2 arguments: <name> <expression>"
                raise ProcessingLevelError(msg)
            derived_name, expression = command[1], command[2]
            if derived_name in ts_defs:
                msg = f"TIME_SERIES_DERIVED name {derived_name!r} is already defined"
                raise ProcessingLevelError(msg)
            try:
                parsed = ast.parse(expression, mode="eval").body
            except SyntaxError as err:
                msg = (
                    f"TIME_SERIES_DERIVED expression {expression!r} is not "
                    f"valid Python syntax: {err.msg}"
                )
                raise ProcessingLevelError(msg) from err
            ref_names = sorted(TimeSeriesDerivedDecorator._collect_names(parsed))
            if not ref_names:
                msg = (
                    f"TIME_SERIES_DERIVED expression {expression!r} "
                    "must reference at least one TIME_SERIES name"
                )
                raise ProcessingLevelError(msg)
            unknown = [n for n in ref_names if n not in ts_defs]
            if unknown:
                msg = (
                    f"TIME_SERIES_DERIVED refers to undefined names {unknown!r}. "
                    f"Defined: {sorted(ts_defs)}."
                )
                raise ProcessingLevelError(msg)
            # Synthesize a TimeSeriesDef for the derived series.  Shape
            # is copied from the first operand (operands must align for
            # arithmetic to make sense — the derived decorator's
            # element-wise eval will fail with a clear numpy error if
            # operands don't broadcast).
            first_operand = ts_defs[ref_names[0]]
            derived_ts = TimeSeriesDef(
                name=first_operand.name,
                span=first_operand.span,
                start=first_operand.start,
                end=first_operand.end,
                multiplier=first_operand.multiplier,
                tag=derived_name,
            )
            ts_defs = {**ts_defs, derived_name: derived_ts}
            try:
                derived_decorator = TimeSeriesDerivedDecorator(
                    derived_ts_def=derived_ts,
                    expression=expression,
                    operand_ts_defs=ts_defs,
                    upstream_decorator=decorator,
                )
            except ValueError as err:
                raise ProcessingLevelError(str(err)) from err
            return 1, derived_decorator, ts_defs
        case "TIME_SERIES_METRIC":
            if decorator is None:
                msg = "TIME_SERIES_METRIC requires an upstream (use STOCKS first)"
                raise ProcessingLevelError(msg)
            if len(command) != 4:
                msg = (
                    "TIME_SERIES_METRIC requires 3 arguments: <name> <metric> <ts_name>"
                )
                raise ProcessingLevelError(msg)
            column_name, metric_name, ts_name_ref = command[1], command[2], command[3]
            ts_def = ts_defs.get(ts_name_ref)
            if ts_def is None:
                msg = (
                    f"TIME_SERIES_METRIC refers to undefined name {ts_name_ref!r}. "
                    f"Defined: {sorted(ts_defs)}"
                )
                raise ProcessingLevelError(msg)
            try:
                metric_decorator = TimeSeriesMetricDecorator(
                    ts_def,
                    upstream_decorator=decorator,
                    name=column_name,
                    metric=metric_name,
                )
            except ValueError as err:
                # Surface unknown-metric errors at pipeline-build time
                # with the same exception type as other DSL failures.
                raise ProcessingLevelError(str(err)) from err
            return 1, metric_decorator, ts_defs
        case _:
            return 0, decorator, ts_defs
