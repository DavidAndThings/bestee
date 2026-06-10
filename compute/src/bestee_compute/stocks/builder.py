"""DSL → :class:`TableDecorator` chain builder.

The DSL is a small whitespace-tokenised language for declaring a
pipeline of named operations.  Every statement starts with a keyword
followed by a user-supplied *name* (so later statements can refer to
earlier ones unambiguously):

* ``ASSET_SCOPE <ticker_type>``
  — set up the base table (one per call; multiple unioned).
* ``SAME_SIC_CATEGORY_AS <ticker>`` / ``PICK_TICKERS <ticker> [<ticker>...]``
  — filter the base table.
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
* ``STOCHASTIC_OSCILLATOR <name> <ts_high> <ts_low> <ts_close>
  <k_period> <d_period>``
  — add ``<name>_k`` / ``<name>_d`` columns holding each ticker's
  latest stochastic-oscillator %K and %D values.
* ``RSI <name> <ts_close> <period>``
  — add a ``<name>`` column holding each ticker's latest Wilder's
  Relative Strength Index value.
* ``SMA <name> <ts> <period>``
  — add a ``<name>`` column holding each ticker's latest simple
  moving average over the trailing ``period`` bars of ``<ts>``.

Pipeline assembly happens as a stack of higher-order
``build_<phase>_decorators`` wrappers around :func:`master_decorator_builder`.
Each wrapper takes the upstream builder, asks it for the chain so
far, and prepends its own phase's decorators on top.  Stacking them
with ``@``-decorator syntax keeps the phase ordering declarative.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Sequence

from bestee_compute.stocks import expressions
from bestee_compute.stocks.decorators import (
    AssetScopeDecorator,
    ComputedMetricDecorator,
    FinancialMetricCacheDecorator,
    FinancialsDecorator,
    KnowledgeBaseDecorator,
    MergeDecorator,
    PickTickersDecorator,
    RelativeStrengthIndexDecorator,
    SameSICategoryDecorator,
    SimpleMovingAverageDecorator,
    StochasticOscillatorDecorator,
    TimeSeriesCacheDecorator,
    TimeSeriesDerivedDecorator,
    TimeSeriesMetricDecorator,
)
from bestee_compute.stocks.models import (
    CommandHeader,
    FinancialMetric,
    Metric,
    TimeSeriesDef,
    TimeSeriesName,
    TimeSeriesSpan,
)

type Command = Sequence[str]
type DecoratorBuilder = Callable[[Sequence[Command]], Sequence[KnowledgeBaseDecorator]]


class ProcessingLevelError(Exception):
    """Raised for any DSL parsing / dispatch failure."""


# DSL-friendly ASSET_SCOPE names → Massive API ``ticker_type`` codes.
# Keeps the DSL readable ("STOCKS") while passing the API its expected
# enum value ("CS").
_ASSET_SCOPE_ALIASES: dict[str, str] = {
    "STOCKS": "CS",
    "CS": "CS",
    "ETF": "ETF",
    "OTC_STOCKS": "OS",
}


# ── Phase 1: subsetting (SAME_SIC, PICK_TICKERS) ─────────────────────


def build_subsetting_decorators(upstream: DecoratorBuilder) -> DecoratorBuilder:
    """Wrap *upstream* with ``SAME_SIC_CATEGORY_AS`` / ``PICK_TICKERS``.

    Each such command produces a parallel filter on the upstream
    table; if any are present, the filtered tables are unioned via
    :class:`AppendTablesDecorator`.  If none are present, the
    upstream is passed through unchanged.
    """

    def decorator_builder(cmds: Sequence[Command]) -> Sequence[KnowledgeBaseDecorator]:
        upstream_decorators = upstream(cmds)
        filters: list[KnowledgeBaseDecorator] = []

        for c in cmds:
            match c[0]:
                case CommandHeader.SAME_SIC_CATEGORY_AS:
                    if len(c) != 2:
                        msg = "SAME_SIC_CATEGORY_AS requires 1 argument: <ticker>"
                        raise ProcessingLevelError(msg)
                    filters.append(SameSICategoryDecorator(c[1], *upstream_decorators))
                case CommandHeader.PICK_TICKERS:
                    if len(c) < 2:
                        msg = "PICK_TICKERS requires at least one <ticker>"
                        raise ProcessingLevelError(msg)
                    filters.append(
                        PickTickersDecorator(list(c[1:]), *upstream_decorators)
                    )
                case _:
                    pass

        if not filters:
            return upstream_decorators
        return [MergeDecorator(*filters)]

    return decorator_builder


# ── Phase 2: financial metrics ───────────────────────────────────────


def build_financials_decorators(upstream: DecoratorBuilder) -> DecoratorBuilder:
    """Wrap *upstream* with ``FINANCIAL_METRIC`` handling.

    Each ``FINANCIAL_METRIC`` line becomes a config-only
    :class:`FinancialMetricCacheDecorator`.  If any are present, a
    single :class:`FinancialsDecorator` is constructed with the
    upstream's table-providing decorators alongside all cache markers
    — :class:`FinancialsDecorator` then batches every metric into one
    ``build_financials_df`` call.  If no ``FINANCIAL_METRIC`` lines
    are present, the upstream's decorators are returned unchanged.
    """

    def decorator_builder(cmds: Sequence[Command]) -> Sequence[KnowledgeBaseDecorator]:
        upstream_decorators = upstream(cmds)
        caches: list[FinancialMetricCacheDecorator] = []
        seen_names: set[str] = set()

        for c in cmds:
            if c[0] != CommandHeader.FINANCIAL_METRIC:
                continue
            if len(c) != 5:
                msg = (
                    "FINANCIAL_METRIC requires 4 arguments: "
                    "<name> <metric> <fiscal_year> <fiscal_quarter>"
                )
                raise ProcessingLevelError(msg)
            name, metric_token, year_token, quarter_token = c[1], c[2], c[3], c[4]
            if name in seen_names:
                msg = f"FINANCIAL_METRIC name {name!r} is already defined"
                raise ProcessingLevelError(msg)
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
            seen_names.add(name)
            caches.append(
                FinancialMetricCacheDecorator(
                    name,
                    FinancialMetric(
                        metric=metric_enum,
                        fiscal_year=fiscal_year,
                        fiscal_quarter=fiscal_quarter,
                    ),
                )
            )

        if not caches:
            return upstream_decorators
        return [FinancialsDecorator(*upstream_decorators, *caches)]

    return decorator_builder


# ── Phase 3: computed metrics ────────────────────────────────────────


def build_computed_metric_decorators(upstream: DecoratorBuilder) -> DecoratorBuilder:
    """Wrap *upstream* with ``COMPUTED_METRIC`` handling.

    Each ``COMPUTED_METRIC`` line chains a
    :class:`ComputedMetricDecorator` on top of the current chain head
    — references in its expression resolve against columns added
    earlier (by ``FINANCIAL_METRIC`` or an earlier ``COMPUTED_METRIC``).
    """

    def decorator_builder(cmds: Sequence[Command]) -> Sequence[KnowledgeBaseDecorator]:
        current: Sequence[KnowledgeBaseDecorator] = upstream(cmds)
        for c in cmds:
            if c[0] != CommandHeader.COMPUTED_METRIC:
                continue
            if len(c) < 3:
                msg = (
                    "COMPUTED_METRIC requires at least 2 arguments: <name> <expression>"
                )
                raise ProcessingLevelError(msg)
            name = c[1]
            # Tokens 2.. are the whitespace-tokenised expression; rejoin
            # with single spaces — ast.parse is tolerant of whitespace.
            expression = " ".join(c[2:])
            try:
                next_stage = ComputedMetricDecorator(name, expression, *current)
            except ValueError as err:
                raise ProcessingLevelError(str(err)) from err
            current = [next_stage]
        return current

    return decorator_builder


# ── Phase 4: time-series declarations, derivations, and metrics ──────


def _process_time_series(
    c: Command,
    current: Sequence[KnowledgeBaseDecorator],
    ts_defs: dict[str, TimeSeriesDef],
) -> tuple[Sequence[KnowledgeBaseDecorator], dict[str, TimeSeriesDef]]:
    """Handle a single ``TIME_SERIES`` command.

    Form: ``TIME_SERIES <name> <field> <start> <end> <span> <multiplier>``.
    """
    if len(c) != 7:
        msg = (
            "TIME_SERIES requires 6 arguments: "
            "<name> <ohlc_field> <start> <end> <span> <multiplier>"
        )
        raise ProcessingLevelError(msg)
    name, field, start, end, span_token, multiplier_token = c[1:7]
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
        msg = f"TIME_SERIES multiplier must be an integer, got {multiplier_token!r}"
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
    return [TimeSeriesCacheDecorator(name, ts_def, *current)], ts_defs


def _process_time_series_derived(
    c: Command,
    current: Sequence[KnowledgeBaseDecorator],
    ts_defs: dict[str, TimeSeriesDef],
) -> tuple[Sequence[KnowledgeBaseDecorator], dict[str, TimeSeriesDef]]:
    """Handle a single ``TIME_SERIES_DERIVED`` command.

    Form: ``TIME_SERIES_DERIVED <name> <expression>``.  *expression*
    may contain spaces (it's the whitespace-tokenised tail joined back
    with single spaces — ``ast.parse`` is tolerant of whitespace).
    """
    if len(c) < 3:
        msg = "TIME_SERIES_DERIVED requires at least 2 arguments: <name> <expression>"
        raise ProcessingLevelError(msg)
    derived_name = c[1]
    expression = " ".join(c[2:])
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
    ref_names = sorted(expressions.collect_names(parsed))
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
    # Synthesise a TimeSeriesDef for the derived series.  Shape is
    # copied from the first operand; operands must broadcast at build
    # time, and numpy raises a clear error if they don't.
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
        next_stage = TimeSeriesDerivedDecorator(
            derived_name, expression, list(ts_defs), derived_ts, *current
        )
    except ValueError as err:
        raise ProcessingLevelError(str(err)) from err
    return [next_stage], ts_defs


def _process_time_series_metric(
    c: Command,
    current: Sequence[KnowledgeBaseDecorator],
    ts_defs: dict[str, TimeSeriesDef],
) -> Sequence[KnowledgeBaseDecorator]:
    """Handle a single ``TIME_SERIES_METRIC`` command.

    Form: ``TIME_SERIES_METRIC <name> <metric> <ts_name>``.
    """
    if len(c) != 4:
        msg = "TIME_SERIES_METRIC requires 3 arguments: <name> <metric> <ts_name>"
        raise ProcessingLevelError(msg)
    column_name, metric_name, ts_name_ref = c[1], c[2], c[3]
    if ts_name_ref not in ts_defs:
        msg = (
            f"TIME_SERIES_METRIC refers to undefined name {ts_name_ref!r}. "
            f"Defined: {sorted(ts_defs)}"
        )
        raise ProcessingLevelError(msg)
    try:
        next_stage = TimeSeriesMetricDecorator(
            column_name, ts_name_ref, metric_name, *current
        )
    except ValueError as err:
        # Surface unknown-metric errors at pipeline-build time with
        # the same exception type as other DSL failures.
        raise ProcessingLevelError(str(err)) from err
    return [next_stage]


def _derive_ts_def(source: TimeSeriesDef, tag: str) -> TimeSeriesDef:
    """Clone *source*'s shape onto a fresh ``TimeSeriesDef`` with *tag*.

    Indicator outputs (RSI, SMA, %K, %D) and derived series share the
    bar grid of their source close series, but need a distinct identity
    so the registry can disambiguate them.  ``tag`` becomes the DSL
    alias name (e.g. ``"rsi14"``).
    """
    return TimeSeriesDef(
        name=source.name,
        span=source.span,
        start=source.start,
        end=source.end,
        multiplier=source.multiplier,
        tag=tag,
    )


def _process_stochastic_oscillator(
    c: Command,
    current: Sequence[KnowledgeBaseDecorator],
    ts_defs: dict[str, TimeSeriesDef],
) -> tuple[Sequence[KnowledgeBaseDecorator], dict[str, TimeSeriesDef]]:
    """Handle a single ``STOCHASTIC_OSCILLATOR`` command.

    Form: ``STOCHASTIC_OSCILLATOR <name> <ts_high> <ts_low> <ts_close>
    <k_period> <d_period>``.  Produces two aliases ``<name>_k`` and
    ``<name>_d``, each carrying a full series (and a latest scalar)
    downstream stages can reference like any other ``TIME_SERIES``.
    """
    if len(c) != 7:
        msg = (
            "STOCHASTIC_OSCILLATOR requires 6 arguments: "
            "<name> <ts_high> <ts_low> <ts_close> <k_period> <d_period>"
        )
        raise ProcessingLevelError(msg)
    name = c[1]
    ts_refs = {"<ts_high>": c[2], "<ts_low>": c[3], "<ts_close>": c[4]}
    for role, ref in ts_refs.items():
        if ref not in ts_defs:
            msg = (
                f"STOCHASTIC_OSCILLATOR {role} refers to undefined name "
                f"{ref!r}. Defined: {sorted(ts_defs)}"
            )
            raise ProcessingLevelError(msg)
    high_alias, low_alias, close_alias = c[2], c[3], c[4]
    close_def = ts_defs[close_alias]
    try:
        k_period = int(c[5])
        d_period = int(c[6])
    except ValueError as err:
        msg = (
            "STOCHASTIC_OSCILLATOR k_period and d_period must be "
            f"integers, got k={c[5]!r} d={c[6]!r}"
        )
        raise ProcessingLevelError(msg) from err
    k_alias, d_alias = f"{name}_k", f"{name}_d"
    for produced in (k_alias, d_alias):
        if produced in ts_defs:
            msg = (
                f"STOCHASTIC_OSCILLATOR would produce alias {produced!r}, "
                "which is already defined"
            )
            raise ProcessingLevelError(msg)
    try:
        decorator = StochasticOscillatorDecorator(
            name,
            high_alias,
            low_alias,
            close_alias,
            k_period,
            d_period,
            close_def,
            *current,
        )
    except ValueError as err:
        raise ProcessingLevelError(str(err)) from err
    ts_defs = {
        **ts_defs,
        k_alias: _derive_ts_def(close_def, k_alias),
        d_alias: _derive_ts_def(close_def, d_alias),
    }
    return [decorator], ts_defs


def _process_rsi(
    c: Command,
    current: Sequence[KnowledgeBaseDecorator],
    ts_defs: dict[str, TimeSeriesDef],
) -> tuple[Sequence[KnowledgeBaseDecorator], dict[str, TimeSeriesDef]]:
    """Handle a single ``RSI`` command.

    Form: ``RSI <name> <ts_close> <period>``.  Produces alias ``<name>``
    holding a full Wilder-smoothed RSI series (and a latest scalar);
    registered in ``ts_defs`` so downstream ``TIME_SERIES_DERIVED`` /
    indicators can chain off it.
    """
    if len(c) != 4:
        msg = "RSI requires 3 arguments: <name> <ts_close> <period>"
        raise ProcessingLevelError(msg)
    name, ts_close_ref, period_token = c[1], c[2], c[3]
    if name in ts_defs:
        msg = f"RSI alias {name!r} is already defined"
        raise ProcessingLevelError(msg)
    ts_close_def = ts_defs.get(ts_close_ref)
    if ts_close_def is None:
        msg = (
            f"RSI <ts_close> refers to undefined name {ts_close_ref!r}. "
            f"Defined: {sorted(ts_defs)}"
        )
        raise ProcessingLevelError(msg)
    try:
        period = int(period_token)
    except ValueError as err:
        msg = f"RSI period must be an integer, got {period_token!r}"
        raise ProcessingLevelError(msg) from err
    try:
        decorator = RelativeStrengthIndexDecorator(
            name, ts_close_ref, period, ts_close_def, *current
        )
    except ValueError as err:
        raise ProcessingLevelError(str(err)) from err
    ts_defs = {**ts_defs, name: _derive_ts_def(ts_close_def, name)}
    return [decorator], ts_defs


def _process_sma(
    c: Command,
    current: Sequence[KnowledgeBaseDecorator],
    ts_defs: dict[str, TimeSeriesDef],
) -> tuple[Sequence[KnowledgeBaseDecorator], dict[str, TimeSeriesDef]]:
    """Handle a single ``SMA`` command.

    Form: ``SMA <name> <ts> <period>``.  Produces alias ``<name>``
    holding a full SMA series (and a latest scalar); registered in
    ``ts_defs`` so downstream stages can chain off it.
    """
    if len(c) != 4:
        msg = "SMA requires 3 arguments: <name> <ts> <period>"
        raise ProcessingLevelError(msg)
    name, ts_ref, period_token = c[1], c[2], c[3]
    if name in ts_defs:
        msg = f"SMA alias {name!r} is already defined"
        raise ProcessingLevelError(msg)
    ts_def = ts_defs.get(ts_ref)
    if ts_def is None:
        msg = (
            f"SMA <ts> refers to undefined name {ts_ref!r}. Defined: {sorted(ts_defs)}"
        )
        raise ProcessingLevelError(msg)
    try:
        period = int(period_token)
    except ValueError as err:
        msg = f"SMA period must be an integer, got {period_token!r}"
        raise ProcessingLevelError(msg) from err
    try:
        decorator = SimpleMovingAverageDecorator(name, ts_ref, period, ts_def, *current)
    except ValueError as err:
        raise ProcessingLevelError(str(err)) from err
    ts_defs = {**ts_defs, name: _derive_ts_def(ts_def, name)}
    return [decorator], ts_defs


def build_time_series_decorators(upstream: DecoratorBuilder) -> DecoratorBuilder:
    """Wrap *upstream* with ``TIME_SERIES`` / ``TIME_SERIES_DERIVED`` /
    ``TIME_SERIES_METRIC`` / ``STOCHASTIC_OSCILLATOR`` / ``RSI`` / ``SMA``
    handling.

    All commands share a per-pass ``ts_defs`` registry.  Commands run
    in two sub-passes:

    1. All raw ``TIME_SERIES`` declarations first (in input order) —
       so DERIVED / METRIC / indicator lines may sit anywhere in the
       input, including before their referenced ``TIME_SERIES``.
    2. Then ``TIME_SERIES_DERIVED``, ``TIME_SERIES_METRIC``,
       ``STOCHASTIC_OSCILLATOR``, ``RSI``, and ``SMA`` lines, in input
       order.  Indicator and DERIVED stages register their output
       aliases in ``ts_defs`` as they're processed, so a later
       ``TIME_SERIES_DERIVED smoothed sma20 - rsi14`` works as long as
       the operands appear earlier in input order.  Cycles aren't
       possible — the registry only ever grows.
    """

    def decorator_builder(cmds: Sequence[Command]) -> Sequence[KnowledgeBaseDecorator]:
        current: Sequence[KnowledgeBaseDecorator] = upstream(cmds)
        ts_defs: dict[str, TimeSeriesDef] = {}

        # Pass 1: register every TIME_SERIES declaration and chain its
        # cache decorator.
        for c in cmds:
            if c[0] == CommandHeader.TIME_SERIES:
                current, ts_defs = _process_time_series(c, current, ts_defs)

        # Pass 2: build derived series, metric columns, and indicators;
        # all may reference any name registered above.  Indicator
        # commands (RSI / SMA / STOCHASTIC_OSCILLATOR) extend ts_defs
        # with their output alias(es) so subsequent TIME_SERIES_DERIVED
        # / TIME_SERIES_METRIC / indicator lines can chain off them —
        # source must appear earlier in input order, since the registry
        # grows as we go.
        for c in cmds:
            match c[0]:
                case CommandHeader.TIME_SERIES_DERIVED:
                    current, ts_defs = _process_time_series_derived(c, current, ts_defs)
                case CommandHeader.TIME_SERIES_METRIC:
                    current = _process_time_series_metric(c, current, ts_defs)
                case CommandHeader.STOCHASTIC_OSCILLATOR:
                    current, ts_defs = _process_stochastic_oscillator(
                        c, current, ts_defs
                    )
                case CommandHeader.RSI:
                    current, ts_defs = _process_rsi(c, current, ts_defs)
                case CommandHeader.SMA:
                    current, ts_defs = _process_sma(c, current, ts_defs)
                case _:
                    pass

        return current

    return decorator_builder


# ── Phase 0 + top-level entry points ─────────────────────────────────


@build_time_series_decorators
@build_computed_metric_decorators
@build_financials_decorators
@build_subsetting_decorators
def master_decorator_builder(
    cmds: Sequence[Command],
) -> Sequence[KnowledgeBaseDecorator]:
    """Innermost builder — handles ``ASSET_SCOPE`` lines.

    Each ``ASSET_SCOPE`` line produces an :class:`AssetScopeDecorator`;
    multiple are unioned via :class:`MergeDecorator`.  At least one
    ``ASSET_SCOPE`` is required — every other phase chains on top of
    this base.

    The ``@build_*`` wrappers stacked on this function add the rest of
    the DSL on top, in order: SUBSETTING → FINANCIALS → COMPUTED →
    TIME SERIES.
    """
    sources: list[KnowledgeBaseDecorator] = []
    for c in cmds:
        match c[0]:
            case CommandHeader.ASSET_SCOPE:
                if len(c) != 2:
                    msg = "ASSET_SCOPE requires 1 argument: <ticker_type>"
                    raise ProcessingLevelError(msg)
                alias = c[1]
                ticker_type = _ASSET_SCOPE_ALIASES.get(alias)
                if ticker_type is None:
                    msg = (
                        f"Unknown ASSET_SCOPE {alias!r}. "
                        f"Known: {sorted(_ASSET_SCOPE_ALIASES)}."
                    )
                    raise ProcessingLevelError(msg)
                sources.append(AssetScopeDecorator(ticker_type))
            case _:
                pass

    if not sources:
        msg = (
            "Pipeline must declare at least one ASSET_SCOPE before any "
            "downstream command."
        )
        raise ProcessingLevelError(msg)
    # Always wrap in a MergeDecorator so the downstream invariant ("one
    # chain head") holds whether the user declared one ASSET_SCOPE or many.
    return [MergeDecorator(*sources)]


def decorator_builder(cmds: Sequence[Command]) -> KnowledgeBaseDecorator:
    """Public entry point — build one chained decorator from *cmds*.

    Wraps :func:`master_decorator_builder` (which returns a single-
    element sequence by invariant) and unwraps the single decorator.

    Raises:
        ProcessingLevelError: For any DSL parsing failure, including
            an unrecognised command keyword.
    """
    # Surface unknown keywords with a clear error before we hand off
    # to the phase pipeline (which would silently skip them).
    known_keywords = {h.value for h in CommandHeader}
    unknown = sorted({c[0] for c in cmds if c[0] not in known_keywords})
    if unknown:
        msg = f"Unrecognized command keyword(s): {unknown}"
        raise ProcessingLevelError(msg)

    results = list(master_decorator_builder(cmds))
    if len(results) != 1:
        msg = f"decorator_builder expected exactly one chain head, got {len(results)}"
        raise ProcessingLevelError(msg)
    return results[0]
