"""Market-wide snapshot data from the Massive API."""

import polars as pl
from great_tables import GT
from massive.rest.models import GroupedDailyAgg

from bestee.client import get_client

_BASE_FIELDS = ["Open", "High", "Low", "Close", "Volume", "VWAP", "Transactions"]


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
