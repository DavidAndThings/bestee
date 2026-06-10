"""Build a financial-metrics comparison table from the Massive API."""

import logging
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

import polars as pl
from great_tables import GT
from massive import RESTClient

from bestee_compute.client import get_client
from bestee_compute.stocks.models import FinancialMetric, StatementType

logger = logging.getLogger(__name__)

# ── Types used as grouping keys ──────────────────────────────────────

# (statement_type, fiscal_year, fiscal_quarter) — uniquely identifies
# a single API call for financial-statement endpoints.
type _PeriodKey = tuple[StatementType, int, int]

# Ratios are point-in-time and don't support fiscal-period filters, so
# all ratio metrics share one API call regardless of the period the user
# attached to them.
_RATIOS_KEY: _PeriodKey = (StatementType.RATIOS, 0, 0)

# ── Endpoint registry ────────────────────────────────────────────────

# Maps each statement type to the RESTClient method name and the raw API
# query-parameter used to batch-filter by multiple tickers.
#
# NOTE: The SDK's keyword arguments (e.g. ``tickers_any_of``) do NOT
# translate to the correct ``tickers.any_of`` query parameter.  We work
# around this by passing the filter via the ``params`` dict instead.
_STATEMENT_FETCHERS: dict[
    StatementType,
    tuple[str, str],  # (method_name, raw_api_ticker_param)
] = {
    StatementType.INCOME_STATEMENT: (
        "list_financials_income_statements",
        "tickers.any_of",
    ),
    StatementType.BALANCE_SHEET: (
        "list_financials_balance_sheets",
        "tickers.any_of",
    ),
    StatementType.CASH_FLOW: (
        "list_financials_cash_flow_statements",
        "tickers.any_of",
    ),
    StatementType.RATIOS: (
        "list_financials_ratios",
        "ticker.any_of",
    ),
}


# ── Helper functions ─────────────────────────────────────────────────


def _group_metrics(
    metrics: list[FinancialMetric],
) -> dict[_PeriodKey, list[FinancialMetric]]:
    """Group metrics by the API call they require.

    Financial-statement metrics are keyed by
    ``(statement_type, fiscal_year, fiscal_quarter)`` so that metrics
    from the same statement *and* period share a single call.

    All ratio metrics are collapsed into one group because the ratios
    endpoint does not support fiscal-period filtering.
    """
    groups: dict[_PeriodKey, list[FinancialMetric]] = defaultdict(list)
    for m in metrics:
        if m.statement == StatementType.RATIOS:
            groups[_RATIOS_KEY].append(m)
        else:
            key: _PeriodKey = (m.statement, m.fiscal_year, m.fiscal_quarter)
            groups[key].append(m)
    logger.debug("Grouped %d metrics into %d API call(s)", len(metrics), len(groups))
    return dict(groups)


def _resolve_ticker(result: object, requested: set[str]) -> str | None:
    """Return the first requested ticker found in *result*.

    Financial-statement models expose ``tickers`` (a list), whereas ratio
    models expose ``ticker`` (a single string).
    """
    tickers_attr: list[str] | None = getattr(result, "tickers", None)
    if tickers_attr is not None:
        for t in tickers_attr:
            if t in requested:
                return t
        return None

    ticker_attr: str | None = getattr(result, "ticker", None)
    if ticker_attr is not None and ticker_attr in requested:
        return ticker_attr
    return None


def _period_sort_key(result: object) -> str:
    """Return a date string usable for sorting results newest-first.

    Financial statements have ``period_end``; ratios have ``date``.
    """
    for attr in ("period_end", "date"):
        val = getattr(result, attr, None)
        if val is not None:
            return str(val)
    return ""


def _fetch_statement(
    client: RESTClient,
    key: _PeriodKey,
    tickers_csv: str,
) -> Iterator[Any]:
    """Call the appropriate SDK method and return the result iterator.

    For financial-statement endpoints the ``fiscal_year``,
    ``fiscal_quarter``, and ``timeframe="quarterly"`` filters are sent.
    For ratios, no period filter is applied (the endpoint does not
    support it); the caller is responsible for picking the latest row.
    """
    statement, fiscal_year, fiscal_quarter = key
    method_name, ticker_param = _STATEMENT_FETCHERS[statement]
    method = getattr(client, method_name)

    params: dict[str, str] = {ticker_param: tickers_csv}
    kwargs: dict[str, Any] = {"params": params}

    if statement != StatementType.RATIOS:
        kwargs["fiscal_year"] = fiscal_year
        kwargs["fiscal_quarter"] = fiscal_quarter
        kwargs["timeframe"] = "quarterly"

    logger.info("Calling %s for %s", method_name, tickers_csv)
    return method(**kwargs)


# ── Public API ───────────────────────────────────────────────────────


def build_financials_df(
    tickers: list[str],
    metrics: list[FinancialMetric],
    *,
    api_key: str | None = None,
) -> pl.DataFrame:
    """Build the raw Polars DataFrame of financial metrics.

    Same semantics as :func:`build_financials_table` but returns the
    unstyled DataFrame.  Used by the decorator pipeline to chain
    transformations without round-tripping through Great Tables.

    See :func:`build_financials_table` for argument and behavior
    documentation.
    """
    logger.info(
        "Building financials DataFrame for %d tickers with %d metrics",
        len(tickers),
        len(metrics),
    )
    client = get_client(api_key)
    groups = _group_metrics(metrics)
    logger.info("Metrics grouped into %d API call(s)", len(groups))
    requested = set(tickers)
    tickers_csv = ",".join(tickers)

    # ticker -> column_label -> value
    data: dict[str, dict[str, float | None]] = {
        t: {m.label: None for m in metrics} for t in tickers
    }

    for key, group_metrics in groups.items():
        results = list(_fetch_statement(client, key, tickers_csv))
        logger.debug("Statement %s returned %d results", key, len(results))
        # Sort newest-first so the first hit per ticker is the latest.
        results.sort(key=_period_sort_key, reverse=True)

        # For a given API call we only want one row per ticker (the most
        # recent match).
        seen: set[str] = set()

        for result in results:
            ticker = _resolve_ticker(result, requested)
            if ticker is None or ticker in seen:
                continue
            seen.add(ticker)
            logger.debug("Extracted data for ticker %s from %s", ticker, key)

            for m in group_metrics:
                value = getattr(result, m.field, None)
                if value is not None:
                    data[ticker][m.label] = float(value)

    # Build a Polars DataFrame preserving the caller's ticker order.
    rows = [{"Ticker": t, **data[t]} for t in tickers]
    return pl.DataFrame(rows)


def build_financials_table(
    tickers: list[str],
    metrics: list[FinancialMetric],
    *,
    api_key: str | None = None,
) -> GT:
    """Build a comparison table of financial metrics for the given tickers.

    API calls are grouped by ``(statement_type, fiscal_year,
    fiscal_quarter)``.  Metrics from the same statement *and* period are
    fetched in a **single call** with all tickers batched via
    ``tickers.any_of``.  Statement types with no requested metrics are
    never called.

    Requesting the same metric for two different quarters results in two
    separate columns and (at most) two API calls — one per period.

    Args:
        tickers: Ticker symbols to include as rows (e.g.
            ``["AAPL", "MSFT", "GOOG"]``).
        metrics: The financial metrics (with fiscal periods) to include
            as columns.
        api_key: Massive API key.  Falls back to the ``MASSIVE_API_KEY``
            environment variable when *None*.

    Returns:
        A :class:`great_tables.GT` display table with *tickers* on the
        rows and *metrics* on the columns.

    Raises:
        RuntimeError: If no API key is available.
    """
    df = build_financials_df(tickers, metrics, api_key=api_key)
    metric_cols = [m.label for m in metrics]
    periods = sorted({f"FY{m.fiscal_year} Q{m.fiscal_quarter}" for m in metrics})

    logger.info(
        "Built financials GT table: %d tickers × %d metrics", len(tickers), len(metrics)
    )
    return (
        GT(df)
        .tab_header(
            title="Financial Metrics Comparison",
            subtitle=(
                f"{len(tickers)} tickers · {len(metrics)} metrics · "
                + ", ".join(periods)
            ),
        )
        .fmt_number(
            columns=metric_cols,
            compact=True,
            decimals=2,
        )
        .sub_missing(missing_text="—")
        .cols_align(align="right", columns=metric_cols)
    )
