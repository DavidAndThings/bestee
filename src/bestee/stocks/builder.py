"""DSL → :class:`TableDecorator` chain builder.

The DSL is a small whitespace-tokenised language for declaring a
pipeline of named operations.  Every statement starts with a keyword
followed by a user-supplied *name* (so later statements can refer to
earlier ones unambiguously):

* ``STOCKS`` / ``SAME_SIC_CATEGORY_AS <ticker>``
  — set up the base table.
* ``FINANCIAL_METRIC <name> <metric> <fiscal_year> <fiscal_quarter>``
  — add one column from the financials endpoint.
* ``COMPUTED_METRIC <name> <expression>``
  — add a derived column whose expression references earlier names.
* ``TIME_SERIES <name> <field> <start> <end> <span> <multiplier>``
  — declare an OHLC series (cached, fetched on demand).
* ``TIME_SERIES_DERIVED <name> <expression>``
  — define a new series as arithmetic over existing ones.
* ``TIME_SERIES_METRIC <name> <metric> <ts_name>``
  — add a scalar column by applying a registered reducer to a series.

The builder runs commands in four ordered phases so ordering across
phases in the input is irrelevant; ordering *within* the time-series
phase still matters because references resolve by name.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence

from bestee.stocks.decorators import (
    ComputedMetricDecorator,
    FinancialsDecorator,
    SameSICategoryDecorator,
    TableDecorator,
    TickerSummaryDecorator,
    TimeSeriesCacheDecorator,
    TimeSeriesDerivedDecorator,
    TimeSeriesMetricDecorator,
)
from bestee.stocks.models import (
    FinancialMetric,
    Metric,
    TimeSeriesDef,
    TimeSeriesName,
    TimeSeriesSpan,
)

type Command = Sequence[str]


class ProcessingLevelError(Exception):
    """Raised for any DSL parsing / dispatch failure."""


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
