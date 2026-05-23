"""Market-wide snapshot data from the Massive API."""

import datetime as dt
import logging
from collections.abc import Sequence
from typing import Literal, cast

import polars as pl
from great_tables import GT
from massive.rest.models import Agg, GroupedDailyAgg, TickerSnapshot
from massive.rest.models.common import Sort

from bestee.client import get_client
from bestee.stocks.models import TimeSeriesDef, TimeSeriesName

logger = logging.getLogger(__name__)

_US_EASTERN = dt.timezone(dt.timedelta(hours=-4))

_BASE_FIELDS = ["Open", "High", "Low", "Close", "Volume", "VWAP", "Transactions"]

# Massive supports the full Polygon-style set of bar timespans.  Keep
# this in sync with the ``timespan`` parameter that the upstream API
# documents — invalid values produce a 400 from the server but the
# Literal catches typos at type-check time too.
Timespan = Literal[
    "second", "minute", "hour", "day", "week", "month", "quarter", "year"
]
SortDirection = Literal["asc", "desc"]


def get_market_snapshot(
    date: str,
    *,
    api_key: str | None = None,
    adjusted: bool = True,
    include_otc: bool = False,
) -> GT:
    """Return the full market snapshot for a specific trading day.

    Calls the Massive ``get_grouped_daily_aggs`` endpoint which returns
    OHLCV data for every traded ticker on the given date in a **single
    API call**.

    Every data column is suffixed with the date (e.g.
    ``Open (2025-07-11)``) so that tables from different dates can be
    merged on the ``Ticker`` column without name collisions.

    Args:
        date: Trading date in ``"YYYY-MM-DD"`` format (e.g.
            ``"2025-07-11"``).  Must be a trading day — weekends and
            market holidays return no data.
        api_key: Massive API key.  Falls back to the ``MASSIVE_API_KEY``
            environment variable when *None*.
        adjusted: Whether the results are adjusted for splits.  Defaults
            to *True*.
        include_otc: Include OTC securities.  Defaults to *False*.

    Returns:
        A :class:`great_tables.GT` display table with one row per ticker
        and date-stamped columns for Open, High, Low, Close, Volume,
        VWAP, and Transactions.

    Raises:
        RuntimeError: If no API key is available.
    """
    logger.info(
        "Fetching market snapshot for %s (adjusted=%s, include_otc=%s)",
        date,
        adjusted,
        include_otc,
    )
    client = get_client(api_key)

    results = client.get_grouped_daily_aggs(
        date,
        adjusted=adjusted,
        include_otc=include_otc,
    )

    # Column names carry the date so two snapshots can be merged.
    def _col(name: str) -> str:
        return f"{name} ({date})"

    rows: list[dict[str, str | float | int | None]] = []
    for bar in results:
        if not isinstance(bar, GroupedDailyAgg):
            continue
        rows.append(
            {
                "Ticker": bar.ticker,
                _col("Open"): bar.open,
                _col("High"): bar.high,
                _col("Low"): bar.low,
                _col("Close"): bar.close,
                _col("Volume"): bar.volume,
                _col("VWAP"): bar.vwap,
                _col("Transactions"): bar.transactions,
            }
        )

    logger.info("Collected %d tickers for %s", len(rows), date)

    if not rows:
        schema: dict[str, type[pl.DataType]] = {"Ticker": pl.Utf8}  # type: ignore[assignment]
        for name in _BASE_FIELDS:
            if name in ("Volume", "Transactions"):
                schema[_col(name)] = pl.Int64  # type: ignore[assignment]
            else:
                schema[_col(name)] = pl.Float64  # type: ignore[assignment]
        df = pl.DataFrame(schema=schema)
    else:
        df = pl.DataFrame(rows).sort("Ticker")

    price_cols = [_col(c) for c in ("Open", "High", "Low", "Close", "VWAP")]
    volume_cols = [_col(c) for c in ("Volume", "Transactions")]

    logger.info("Built market snapshot GT table for %s", date)
    return (
        GT(df)
        .tab_header(
            title="Market Snapshot",
            subtitle=f"{date} · {len(rows)} tickers",
        )
        .fmt_number(columns=price_cols, decimals=2)
        .fmt_integer(columns=volume_cols)
        .sub_missing(missing_text="—")
        .cols_align(align="left", columns=["Ticker"])
        .cols_align(align="right", columns=[*price_cols, *volume_cols])
    )


def _trading_date_from_snapshots(
    snapshots: list[TickerSnapshot],  # type: ignore[type-arg]
) -> str:
    """Derive the trading date from the first snapshot's *updated* field.

    The ``updated`` field is a nanosecond-epoch timestamp.  We convert it
    to a date string in US Eastern time (the timezone US markets operate
    in).

    Raises:
        RuntimeError: If no snapshot in the input carries a usable
            timestamp.  In that case the trading date is unknowable and
            date-stamped column names would be meaningless.
    """
    for snap in snapshots:
        if isinstance(snap, TickerSnapshot) and snap.updated is not None:
            ts = dt.datetime.fromtimestamp(int(snap.updated) / 1e9, tz=_US_EASTERN)
            logger.debug(
                "Derived trading date %s from snapshot timestamp",
                ts.strftime("%Y-%m-%d"),
            )
            return ts.strftime("%Y-%m-%d")
    msg = "Could not derive trading date from snapshots (no usable timestamps)"
    raise RuntimeError(msg)


def get_latest_market_snapshot(
    *,
    api_key: str | None = None,
    include_otc: bool = False,
) -> GT:
    """Return the most recent market-wide snapshot (day bar only).

    Calls the Massive ``get_snapshot_all`` endpoint which returns the
    current-day OHLCV bar for every actively traded ticker in a
    **single API call**.

    Every data column is suffixed with the trading date (e.g.
    ``Open (2025-07-18)``) so that this table can be merged with
    historical snapshots on the ``Ticker`` column.

    Args:
        api_key: Massive API key.  Falls back to the ``MASSIVE_API_KEY``
            environment variable when *None*.
        include_otc: Include OTC securities.  Defaults to *False*.

    Returns:
        A :class:`great_tables.GT` display table with one row per ticker
        and date-stamped columns for Open, High, Low, Close, Volume,
        and VWAP.

    Raises:
        RuntimeError: If no API key is available, or if the snapshot
            response contains no usable timestamps (the trading date
            cannot be derived).
    """
    logger.info("Fetching latest market snapshot (include_otc=%s)", include_otc)
    client = get_client(api_key)

    result = client.get_snapshot_all(
        "stocks",
        include_otc=include_otc,
    )
    snapshots: list[TickerSnapshot] = result if isinstance(result, list) else []
    logger.info("Received %d snapshots from API", len(snapshots))

    trading_date = _trading_date_from_snapshots(snapshots)
    logger.info("Trading date: %s", trading_date)

    def _col(name: str) -> str:
        return f"{name} ({trading_date})"

    rows: list[dict[str, str | float | None]] = []
    for snap in snapshots:
        if not isinstance(snap, TickerSnapshot):
            continue
        day = snap.day
        if day is None:
            continue
        rows.append(
            {
                "Ticker": snap.ticker,
                _col("Open"): day.open,
                _col("High"): day.high,
                _col("Low"): day.low,
                _col("Close"): day.close,
                _col("Volume"): day.volume,
                _col("VWAP"): day.vwap,
            }
        )

    logger.info("Collected %d tickers with day data", len(rows))

    if not rows:
        df = pl.DataFrame(
            schema={
                "Ticker": pl.Utf8,
                _col("Open"): pl.Float64,
                _col("High"): pl.Float64,
                _col("Low"): pl.Float64,
                _col("Close"): pl.Float64,
                _col("Volume"): pl.Float64,
                _col("VWAP"): pl.Float64,
            }
        )
    else:
        df = pl.DataFrame(rows).sort("Ticker")

    price_cols = [_col(c) for c in ("Open", "High", "Low", "Close", "VWAP")]
    volume_col = _col("Volume")

    logger.info("Built latest market snapshot GT table")
    return (
        GT(df)
        .tab_header(
            title="Latest Market Snapshot",
            subtitle=f"{trading_date} \u00b7 {len(rows)} tickers",
        )
        .fmt_number(columns=price_cols, decimals=2)
        .fmt_number(columns=[volume_col], compact=True, decimals=0)
        .sub_missing(missing_text="\u2014")
        .cols_align(align="left", columns=["Ticker"])
        .cols_align(align="right", columns=[*price_cols, volume_col])
    )


# -- OHLC bars for one ticker over a date range -----------------------


def get_ohlc(
    ticker: str,
    from_: str | dt.date | dt.datetime,
    to: str | dt.date | dt.datetime,
    *,
    timespan: Timespan = "day",
    multiplier: int = 1,
    adjusted: bool = True,
    sort: SortDirection = "asc",
    limit: int | None = None,
    api_key: str | None = None,
) -> pl.DataFrame:
    """Return OHLC bars for *ticker* over a date range.

    Calls the Massive ``get_aggs`` endpoint, which returns aggregated
    bars at any window size: a 1-day bar, a 5-minute bar, a 4-hour bar,
    a 1-week bar, etc.  The window size is *multiplier \u00d7 timespan*,
    e.g. ``timespan="minute", multiplier=15`` for 15-minute bars.

    Args:
        ticker: Ticker symbol (case-insensitive on Massive's side).
        from_: Start of the window.  Accepts ``"YYYY-MM-DD"``,
            :class:`datetime.date`, :class:`datetime.datetime`, or a
            Unix-millisecond integer (forwarded as-is to Massive).
        to: End of the window.  Same accepted forms as *from_*.
        timespan: Base unit.  One of ``"second"``, ``"minute"``,
            ``"hour"``, ``"day"``, ``"week"``, ``"month"``,
            ``"quarter"``, ``"year"``.  Defaults to ``"day"``.
        multiplier: How many *timespan* units per bar.  Defaults to 1.
        adjusted: Adjust prices for splits.  Defaults to *True*.
        sort: ``"asc"`` (oldest first, default) or ``"desc"``.
        limit: Cap on the number of base aggregates Massive queries to
            build the result (Massive default is 5000, max 50000).
            Defaults to *None* (use Massive's default).
        api_key: Massive API key.  Falls back to ``MASSIVE_API_KEY``.

    Returns:
        A :class:`polars.DataFrame` with one row per bar and columns:

        * ``Timestamp`` (Datetime, UTC) \u2014 bar's start time
        * ``Open``, ``High``, ``Low``, ``Close`` (Float64)
        * ``Volume`` (Float64) \u2014 Massive reports float for fractional
          share trades
        * ``VWAP`` (Float64, nullable \u2014 not all bars carry one)
        * ``Transactions`` (Int64, nullable)

        Rows are sorted in the requested direction.  An empty range
        (e.g. weekend with daily bars) yields an empty frame with the
        same schema.

    Raises:
        RuntimeError: If no API key is available.
    """
    upper = ticker.upper()
    logger.info(
        "Fetching OHLC for %s: %dx %s from %s to %s (adjusted=%s, sort=%s, limit=%s)",
        upper,
        multiplier,
        timespan,
        from_,
        to,
        adjusted,
        sort,
        limit,
    )
    client = get_client(api_key)

    sort_value = Sort.ASC if sort == "asc" else Sort.DESC
    if limit is None:
        results = client.get_aggs(
            upper,
            multiplier,
            timespan,
            from_,
            to,
            adjusted=adjusted,
            sort=sort_value,
        )
    else:
        results = client.get_aggs(
            upper,
            multiplier,
            timespan,
            from_,
            to,
            adjusted=adjusted,
            sort=sort_value,
            limit=limit,
        )

    rows: list[dict[str, object]] = []
    for bar in results:
        if not isinstance(bar, Agg):
            continue
        ts = (
            dt.datetime.fromtimestamp(bar.timestamp / 1000, tz=dt.UTC)
            if bar.timestamp is not None
            else None
        )
        rows.append(
            {
                "Timestamp": ts,
                "Open": bar.open,
                "High": bar.high,
                "Low": bar.low,
                "Close": bar.close,
                "Volume": bar.volume,
                "VWAP": bar.vwap,
                "Transactions": bar.transactions,
            }
        )

    logger.info("Collected %d bars for %s", len(rows), upper)

    schema = {
        "Timestamp": pl.Datetime("ms", time_zone="UTC"),
        "Open": pl.Float64,
        "High": pl.Float64,
        "Low": pl.Float64,
        "Close": pl.Float64,
        "Volume": pl.Float64,
        "VWAP": pl.Float64,
        "Transactions": pl.Int64,
    }
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema)


def get_ohlc_table(
    ticker: str,
    from_: str | dt.date | dt.datetime,
    to: str | dt.date | dt.datetime,
    *,
    timespan: Timespan = "day",
    multiplier: int = 1,
    adjusted: bool = True,
    sort: SortDirection = "asc",
    limit: int | None = None,
    api_key: str | None = None,
) -> GT:
    """Return :func:`get_ohlc` wrapped in a styled GT for display."""
    df = get_ohlc(
        ticker,
        from_,
        to,
        timespan=timespan,
        multiplier=multiplier,
        adjusted=adjusted,
        sort=sort,
        limit=limit,
        api_key=api_key,
    )
    bar_label = f"{multiplier} {timespan}{'s' if multiplier != 1 else ''}"
    subtitle = f"{df.height} bars \u00b7 {bar_label} \u00b7 {from_} \u2192 {to}"
    return (
        GT(df)
        .tab_header(title=f"{ticker.upper()} OHLC", subtitle=subtitle)
        .fmt_number(columns=["Open", "High", "Low", "Close", "VWAP"], decimals=2)
        .fmt_number(columns=["Volume"], compact=True, decimals=0)
        .fmt_integer(columns=["Transactions"])
        .sub_missing(missing_text="\u2014")
        .cols_align(align="left", columns=["Timestamp"])
        .cols_align(
            align="right",
            columns=[
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "VWAP",
                "Transactions",
            ],
        )
    )


_TIME_SERIES_FIELDS: dict[TimeSeriesName, str] = {
    TimeSeriesName.OPEN_PRICE: "Open",
    TimeSeriesName.HIGH_PRICE: "High",
    TimeSeriesName.LOW_PRICE: "Low",
    TimeSeriesName.CLOSE_PRICE: "Close",
}


def get_time_series(
    ticker: str,
    ts_def: TimeSeriesDef,
    *,
    api_key: str | None = None,
) -> Sequence[float]:
    """Return a single OHLC field as a sequence of floats.

    Convenience wrapper for the common case of "give me just the close
    prices" (or open, high, low) over a date range, without the rest
    of the OHLCV frame.  Internally fetches bars via :func:`get_ohlc`
    and extracts the field named by ``ts_def.name``.

    Args:
        ticker: Ticker symbol.
        ts_def: Bundle of bar-window unit (``ts_def.span``), multiplier
            (``ts_def.multiplier``), date range
            (``ts_def.start``/``ts_def.end``), and which OHLC field to
            extract (``ts_def.name``).
        api_key: Massive API key.  Falls back to ``MASSIVE_API_KEY``.

    Returns:
        A list of floats, one per bar, in chronological order.  Bars
        with a null value for the requested field are dropped so the
        result has no ``None``s.  Caller wanting timestamp alignment
        should use :func:`get_ohlc` directly.

    Raises:
        RuntimeError: If no API key is available.
        KeyError: If ``ts_def.name`` isn't one of the OPEN/HIGH/LOW/
            CLOSE_PRICE members of :class:`TimeSeriesName`.
    """
    column = _TIME_SERIES_FIELDS[ts_def.name]
    df = get_ohlc(
        ticker,
        ts_def.start,
        ts_def.end,
        timespan=cast(Timespan, ts_def.span.value),
        multiplier=ts_def.multiplier,
        api_key=api_key,
    )
    return df[column].drop_nulls().to_list()
