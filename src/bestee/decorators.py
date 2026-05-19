from __future__ import annotations

import ast
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import polars as pl
from great_tables import GT

from bestee import columns as cols
from bestee.financials import build_financials_table
from bestee.models import FinancialMetric, Metric
from bestee.tickers import get_all_tickers, get_ticker_details

logger = logging.getLogger(__name__)

type Command = Sequence[str]


class NoUpstreamError(Exception):
    pass


class ProcessingLevelError(Exception):
    pass


class TableDecorator(ABC):
    def __init__(self, upstream_decorator: TableDecorator | None = None):
        self._upstream = upstream_decorator

    def _build_upstream(self) -> GT:
        if self._upstream is None:
            raise NoUpstreamError("No upstream decorator")

        return self._upstream.build()

    @abstractmethod
    def build(
        self,
    ) -> GT:
        pass


class TickerSummaryDecorator(TableDecorator):
    def __init__(
        self,
        ticker_type: str,
    ):
        super().__init__(None)
        self.ticker_type = ticker_type

    def build(
        self,
    ) -> GT:
        all_tickers_table = get_all_tickers(ticker_type=self.ticker_type, active=True)
        symbols: list[str] = all_tickers_table._tbl_data[cols.TICKER].to_list()
        return get_ticker_details(symbols)


class SameSICategoryDecorator(TableDecorator):
    def __init__(self, ticker: str, upstream: TableDecorator):
        super().__init__(upstream)
        self.ticker = ticker

    def build(
        self,
    ) -> GT:
        upstream_table = self._build_upstream()
        df: pl.DataFrame = upstream_table._tbl_data

        # Locate the target ticker's SIC code.
        target_rows = df.filter(pl.col(cols.TICKER) == self.ticker)
        if target_rows.height == 0:
            msg = f"Ticker {self.ticker!r} not found in upstream table"
            raise ValueError(msg)
        target_sic = target_rows[cols.SIC_CODE].item(0)

        # Filter the upstream to rows sharing the same SIC code.
        filtered = df.filter(pl.col(cols.SIC_CODE) == target_sic)

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
    def __init__(self, upstream: TableDecorator):
        super().__init__(upstream)
        self.metrics: list[FinancialMetric] = []

    def add_metric(self, metric: FinancialMetric):
        self.metrics.append(metric)

    def build(
        self,
    ) -> GT:
        upstream_table = self._build_upstream()
        upstream_df: pl.DataFrame = upstream_table._tbl_data

        # No metrics requested — pass the upstream through untouched
        # so we don't make a wasted API call.
        if not self.metrics:
            return upstream_table

        tickers: list[str] = upstream_df[cols.TICKER].to_list()
        metrics_table = build_financials_table(
            tickers=tickers,
            metrics=self.metrics,
        )
        metrics_df: pl.DataFrame = metrics_table._tbl_data

        # Polars DataFrames support .join(); GT objects do not.
        joined = upstream_df.join(metrics_df, on=cols.TICKER, how="inner")

        return (
            GT(joined)
            .tab_header(
                title="Tickers with Financial Metrics",
                subtitle=(
                    f"{joined.height} tickers · {len(self.metrics)} financial metric(s)"
                ),
            )
            .sub_missing(missing_text="—")
        )


class ComputedMetricDecorator(TableDecorator):
    """Add a computed-metric column derived from an arithmetic expression.

    Supports a small DSL of the form::

        COMPUTED_METRIC <NAME> <expression>

    where ``<expression>`` may reference any number of financial metrics
    via::

        FINANCIAL_METRIC <metric_name> <fiscal_year> <fiscal_quarter>

    and may also reference **previously-computed columns** in the
    upstream table via::

        COMPUTED_METRIC <NAME>

    (no body — just the two tokens).  This lets you chain decorators::

        deco = ComputedMetricDecorator(
            "COMPUTED_METRIC ROA "
            "(FINANCIAL_METRIC NET_INCOME 2025 4) / "
            "(FINANCIAL_METRIC TOTAL_ASSETS 2025 4)",
            upstream=base,
        )
        deco = ComputedMetricDecorator(
            "COMPUTED_METRIC DOUBLED_ROA (COMPUTED_METRIC ROA) * 2",
            upstream=deco,
        )

    Expressions combine these references with the standard arithmetic
    operators (``+ - * / ** %``), unary minus, parentheses, and numeric
    literals.

    Evaluation is done by parsing the expression with :mod:`ast` rather
    than ``eval``, so no arbitrary code can run.  Tickers for which any
    referenced metric is missing receive ``None`` in the new column.
    """

    # Top-level expression: COMPUTED_METRIC <name> <body>
    _COMPUTED_METRIC_RE = re.compile(
        r"^\s*COMPUTED_METRIC\s+(\w+)\s+(.+)$",
        re.DOTALL,
    )
    # Inline financial-metric reference.
    _FM_RE = re.compile(r"FINANCIAL_METRIC\s+(\w+)\s+(\d+)\s+(\d+)")
    # Inline computed-metric reference — just the name, no body.
    _CM_REF_RE = re.compile(r"COMPUTED_METRIC\s+(\w+)")

    def __init__(self, expression: str, upstream: TableDecorator):
        super().__init__(upstream)
        self.expression = expression
        (
            self._metric_label,
            self._fm_references,
            self._cm_references,
            self._ast_tree,
        ) = self._parse(expression)
        logger.info(
            "ComputedMetricDecorator parsed %r → "
            "%d FINANCIAL_METRIC ref(s), %d COMPUTED_METRIC ref(s)",
            self._metric_label,
            len(self._fm_references),
            len(self._cm_references),
        )

    # ── Parsing ──────────────────────────────────────────────────────

    @classmethod
    def _parse(
        cls,
        expression: str,
    ) -> tuple[str, list[FinancialMetric], list[str], ast.Expression]:
        """Parse the DSL expression.

        Returns:
            A 4-tuple of ``(column_label, fm_refs, cm_refs, ast_tree)``
            where *cm_refs* is the list of upstream column names (already
            capitalized) referenced via ``COMPUTED_METRIC <NAME>``.
        """
        match = cls._COMPUTED_METRIC_RE.match(expression)
        if match is None:
            msg = (
                f"Invalid expression: {expression!r}\n"
                "Expected: 'COMPUTED_METRIC <NAME> <arithmetic_expression>'"
            )
            raise ValueError(msg)

        name = match.group(1)
        body = match.group(2)

        # Column label uses Python's .capitalize() so 'RETURN_ON_ASSETS'
        # becomes 'Return_on_assets'.
        column_label = name.capitalize()

        fm_refs: list[FinancialMetric] = []
        cm_refs: list[str] = []

        def _sub_fm(match: re.Match[str]) -> str:
            metric_token = match.group(1)
            fiscal_year = int(match.group(2))
            fiscal_quarter = int(match.group(3))
            try:
                metric_enum = Metric[metric_token]
            except KeyError as err:
                msg = (
                    f"Unknown metric: {metric_token!r}. "
                    f"Available metrics: {sorted(m.name for m in Metric)}"
                )
                raise ValueError(msg) from err
            fm = FinancialMetric(
                metric=metric_enum,
                fiscal_year=fiscal_year,
                fiscal_quarter=fiscal_quarter,
            )
            placeholder = f"_v{len(fm_refs)}"
            fm_refs.append(fm)
            return placeholder

        def _sub_cm(match: re.Match[str]) -> str:
            # The referenced column was created by an earlier
            # ComputedMetricDecorator whose name was passed through
            # .capitalize() — do the same here for lookup.
            ref_column = match.group(1).capitalize()
            placeholder = f"_c{len(cm_refs)}"
            cm_refs.append(ref_column)
            return placeholder

        # Substitute FINANCIAL_METRIC first; it has more arguments
        # so substituting it first won't affect the CM regex.
        substituted = cls._FM_RE.sub(_sub_fm, body)
        substituted = cls._CM_REF_RE.sub(_sub_cm, substituted)

        try:
            tree = ast.parse(substituted, mode="eval")
        except SyntaxError as err:
            msg = (
                f"Invalid arithmetic expression: {body!r}\n"
                f"After substitution: {substituted!r}\n"
                f"Parser error: {err.msg}"
            )
            raise ValueError(msg) from err

        return column_label, fm_refs, cm_refs, tree

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

    def build(self) -> GT:
        upstream_table = self._build_upstream()
        upstream_df: pl.DataFrame = upstream_table._tbl_data
        tickers: list[str] = upstream_df[cols.TICKER].to_list()

        # Verify all upstream column references exist before doing work.
        missing = [c for c in self._cm_references if c not in upstream_df.columns]
        if missing:
            msg = (
                f"COMPUTED_METRIC reference(s) {missing!r} not found in "
                f"upstream table. Available columns: {upstream_df.columns}"
            )
            raise ValueError(msg)

        # Build a lookup: ticker -> {placeholder_name -> value}
        env_per_ticker: dict[str, dict[str, float | None]] = {t: {} for t in tickers}

        # 1. Populate _v placeholders from a single financials API call.
        if self._fm_references:
            metrics_table = build_financials_table(
                tickers=tickers,
                metrics=self._fm_references,
            )
            metrics_df: pl.DataFrame = metrics_table._tbl_data

            for row in metrics_df.iter_rows(named=True):
                ticker = row[cols.TICKER]
                env = env_per_ticker.setdefault(ticker, {})
                for i, fm in enumerate(self._fm_references):
                    raw: Any = row.get(fm.label)
                    env[f"_v{i}"] = float(raw) if raw is not None else None

        # 2. Populate _c placeholders from the upstream DataFrame.
        if self._cm_references:
            for row in upstream_df.iter_rows(named=True):
                ticker = row[cols.TICKER]
                env = env_per_ticker.setdefault(ticker, {})
                for i, ref_col in enumerate(self._cm_references):
                    raw = row.get(ref_col)
                    env[f"_c{i}"] = float(raw) if raw is not None else None

        # Evaluate the expression once per ticker.
        computed: list[float | None] = []
        for ticker in tickers:
            env = env_per_ticker.get(ticker, {})
            try:
                value = self._evaluate(self._ast_tree, env)
            except TypeError, ZeroDivisionError, OverflowError:
                value = None
            computed.append(value)

        logger.info(
            "Computed %r for %d tickers (%d non-null)",
            self._metric_label,
            len(computed),
            sum(1 for v in computed if v is not None),
        )

        # Append the new column to the upstream DataFrame.
        result_df = upstream_df.with_columns(
            pl.Series(name=self._metric_label, values=computed, dtype=pl.Float64)
        )

        return (
            GT(result_df)
            .tab_header(
                title=f"With Computed Metric: {self._metric_label}",
                subtitle=f"{result_df.height} tickers",
            )
            .sub_missing(missing_text="—")
        )


_NUM_PROCESSING_LEVELS = 3


def decorator_builder(commands: Sequence[Command]) -> TableDecorator:
    """Build a chained :class:`TableDecorator` from a sequence of commands.

    Commands are processed in three logical phases so that ordering in
    the input is irrelevant:

    * **Level 0** — ``STOCKS``, ``SAME_SIC_CATEGORY_AS``
      (set up the base table and any filtering)
    * **Level 1** — ``FINANCIAL_METRIC``
      (accumulated into a single :class:`FinancialsDecorator` so the
      financial-statement endpoints are called once per period)
    * **Level 2** — ``COMPUTED_METRIC``
      (each command becomes its own :class:`ComputedMetricDecorator`,
      chained on top of the previous one)

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
    """Handle FINANCIAL_METRIC commands, accumulating into one decorator."""
    match command[0]:
        case "FINANCIAL_METRIC":
            if fin_decorator is None:
                if decorator is None:
                    msg = "FINANCIAL_METRIC requires an upstream (use STOCKS first)"
                    raise ProcessingLevelError(msg)
                fin_decorator = FinancialsDecorator(upstream=decorator)
                decorator = fin_decorator
            try:
                metric_enum = Metric[command[1]]
            except KeyError as err:
                msg = (
                    f"Unknown FINANCIAL_METRIC name: {command[1]!r}. "
                    f"Available: {sorted(m.name for m in Metric)}"
                )
                raise ProcessingLevelError(msg) from err
            fin_decorator.add_metric(
                FinancialMetric(
                    metric=metric_enum,
                    fiscal_year=int(command[2]),
                    fiscal_quarter=int(command[3]),
                )
            )
            return 1, decorator, fin_decorator
        case _:
            return 0, decorator, fin_decorator


def command_processor_level_two(
    command: Command,
    decorator: TableDecorator | None,
) -> tuple[int, TableDecorator | None]:
    """Handle COMPUTED_METRIC commands, each as its own chained decorator."""
    match command[0]:
        case "COMPUTED_METRIC":
            if decorator is None:
                msg = "COMPUTED_METRIC requires an upstream (use STOCKS first)"
                raise ProcessingLevelError(msg)
            # The original whitespace-separated expression is rebuilt
            # by joining the tokens with single spaces.  The DSL parser
            # is tolerant of extra spaces (uses ``\s+``).
            expression = " ".join(command)
            return 1, ComputedMetricDecorator(upstream=decorator, expression=expression)
        case _:
            return 0, decorator
