import datetime as dt
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import reduce
from typing import Literal, cast

import numpy as np
import polars as pl
import polars.selectors as cs
from massive.rest.models import Agg
from massive.rest.models.aggs import GroupedDailyAgg
from massive.rest.models.common import Sort
from sklearn.decomposition import PCA

from bestee_compute.client import get_client
from bestee_compute.stocks.models import OHLCHeader

logger = logging.getLogger(__name__)

Timespan = Literal[
    "second", "minute", "hour", "day", "week", "month", "quarter", "year"
]
SortDirection = Literal["asc", "desc"]
# Which slice of ``tickers`` a method should operate on, by the liquidity split.
TickerClass = Literal["all", "liquid", "illiquid"]
# Which residual model a residualized analysis fits (rolling PCA vs. OLS market
# model), and how a residual correlation is turned into a clustering affinity.
ResidualMethod = Literal["pca", "ols"]
SimilarityMetric = Literal["aff", "corr"]
# How the RRG reference is built, and the per-asset regime features modeled.
ReferenceType = Literal["mean", "ticker"]
RegimeFeature = Literal["residual", "log_vol", "abs_residual"]

_LOG_RETURN_COLUMN_NAME = "LogReturn"
_PCA_RESIDUAL_COLUMN_NAME = "PCA_Residual"
_OLS_RESIDUAL_COLUMN_NAME = "OLS_Residual"

# OHLC columns that carry a price (safe to gap-fill); Volume / Transactions are
# counts and must keep their non-trading-day nulls, else a filled price would
# fabricate turnover and corrupt the liquidity median.
_PRICE_COLUMNS = frozenset(
    {
        OHLCHeader.OPEN,
        OHLCHeader.HIGH,
        OHLCHeader.LOW,
        OHLCHeader.CLOSE,
        OHLCHeader.VWAP,
    }
)

# Concurrency for the grouped-daily fan-out (one request per trading day).
_GROUPED_DAILY_MAX_WORKERS = 16
# OHLCHeader -> the matching attribute on a GroupedDailyAgg bar.
_GROUPED_DAILY_FIELD: dict[OHLCHeader, str] = {
    OHLCHeader.OPEN: "open",
    OHLCHeader.HIGH: "high",
    OHLCHeader.LOW: "low",
    OHLCHeader.CLOSE: "close",
    OHLCHeader.VOLUME: "volume",
    OHLCHeader.VWAP: "vwap",
    OHLCHeader.TRANSACTIONS: "transactions",
}


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
@dataclass
class AnalysisConfig:
    """Every input to the workflow, consolidated into one object.

    Merges what used to be three classes -- the fetch window, the universe plus
    per-analysis knobs, and the fetch infrastructure plus caches -- so every
    function takes a single ``config``. Only the universe and fetch window are
    required; each analysis' own settings are optional and grouped (the
    analysis-specific ones behind an ``<analysis>_`` prefix, the shared
    residualization knobs together). Fields an analysis does not set are
    validated where they are read, not at construction, so each analysis pays
    only for what it uses.
    """

    # Universe and fetch window (required).
    tickers: Sequence[str]
    start_date: str | dt.date | dt.datetime
    end_date: str | dt.date | dt.datetime

    # Fetch-window options.
    timespan: Timespan = "day"
    bar_multiplier: int = 1
    adjusted: bool = True
    sort: SortDirection = "asc"
    limit: int | None = None
    api_key: str | None = None

    # Price field and benchmark (read by every analysis).
    ohlc_column: OHLCHeader = OHLCHeader.CLOSE
    benchmark_ticker: str | None = None

    # Residualization knobs (PCA / OLS analyses: clustering, regimes). Optional
    # so an analysis that does not residualize need not set a window.
    residualization_method: ResidualMethod = "pca"
    residualization_window: int | None = None
    normalization_window: int | None = None
    n_components: int = 1

    # Spectral-clustering knobs (read by run_spectral_clustering only).
    clustering_similarity_metric: SimilarityMetric = "aff"
    clustering_min_num_clusters: int = 2
    clustering_max_num_clusters: int = 50
    clustering_assign_illiquid: bool = False

    # Relative-rotation-graph knobs (read by the rrg functions only).
    rrg_reference_type: ReferenceType = "mean"
    rrg_momentum_lookback: int | None = None
    rrg_smoothing_span: int | None = None

    # Per-asset regime-detection knobs (read by run_regime_analysis only).
    regime_hmm_hidden_states: int = 2
    regime_hmm_lag: int = 1
    regime_features: list[RegimeFeature] = field(
        default_factory=lambda: ["residual", "log_vol"]
    )
    regime_vol_halflife: int = 20
    regime_covariance_type: Literal["diag", "full"] = "diag"
    regime_sticky: float = 1.0
    regime_auto_select_states: bool = False
    regime_state_range: tuple[int, int] = (2, 4)
    regime_n_init: int = 5
    regime_n_iter: int = 100
    regime_min_covar: float = 1e-4
    regime_min_obs_per_param: int = 10
    regime_random_state: int = 0

    # Fetch infrastructure and liquidity / coverage screening.
    max_workers: int = _GROUPED_DAILY_MAX_WORKERS
    min_coverage: float = 0.95
    min_dollar_volume: float = 0.0
    # An optional pre-built clean panel; when set, columns are sliced from it
    # rather than fetched.
    injected_panel: pl.DataFrame | None = field(default=None, repr=False, compare=False)

    # Lazily-built caches (not constructor arguments).
    _dollar_volume: dict[str, float] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _resolved_panel: pl.DataFrame | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        # Validation the old pydantic Fields/validators enforced; the window,
        # cluster and regime invariants are the ones worth checking up front.
        if not self.tickers:
            raise ValueError("tickers must be non-empty")
        if self.residualization_window is not None and self.residualization_window < 2:
            raise ValueError("residualization_window must be >= 2")
        if self.normalization_window is not None and self.normalization_window < 2:
            raise ValueError("normalization_window must be >= 2")
        if self.n_components < 1:
            raise ValueError("n_components must be >= 1")
        if self.residualization_window is not None:
            # PCA needs n_components <= min(n_samples, n_features) = the window
            # length and the basket size.
            limit = min(self.residualization_window, len(self.tickers))
            if self.n_components > limit:
                raise ValueError(
                    f"n_components ({self.n_components}) cannot exceed "
                    f"min(residualization_window, n_tickers) = {limit}"
                )
        if not self.regime_features:
            raise ValueError("regime_features must list at least one feature")
        lo, hi = self.regime_state_range
        if self.regime_auto_select_states and not (2 <= lo <= hi):
            raise ValueError(
                f"regime_state_range must satisfy 2 <= lo <= hi, got {(lo, hi)}"
            )


# ----------------------------------------------------------------------------
# Date helpers
# ----------------------------------------------------------------------------
def _to_date(value: str | dt.date | dt.datetime) -> dt.date:
    """Coerce a start/end bound (ISO string, date, or datetime) to a date."""
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])


def _weekday_dates(start: dt.date, end: dt.date) -> list[dt.date]:
    """Mon-Fri dates in ``[start, end]`` (holidays return empty bulk payloads)."""
    if end < start:
        start, end = end, start
    days: list[dt.date] = []
    day = start
    while day <= end:
        if day.weekday() < 5:
            days.append(day)
        day += dt.timedelta(days=1)
    return days


# ----------------------------------------------------------------------------
# Massive fetch primitives
# ----------------------------------------------------------------------------
def get_ohlc(
    ticker: str,
    config: AnalysisConfig,
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
        config.bar_multiplier,
        config.timespan,
        config.start_date,
        config.end_date,
        config.adjusted,
        config.sort,
        config.limit,
    )
    client = get_client(config.api_key)

    sort_value = Sort.ASC if config.sort == "asc" else Sort.DESC
    if config.limit is None:
        results = client.get_aggs(
            upper,
            config.bar_multiplier,
            config.timespan,
            config.start_date,
            config.end_date,
            adjusted=config.adjusted,
            sort=sort_value,
        )
    else:
        results = client.get_aggs(
            upper,
            config.bar_multiplier,
            config.timespan,
            config.start_date,
            config.end_date,
            adjusted=config.adjusted,
            sort=sort_value,
            limit=config.limit,
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
                OHLCHeader.TIMESTAMP: ts,
                OHLCHeader.OPEN: bar.open,
                OHLCHeader.HIGH: bar.high,
                OHLCHeader.LOW: bar.low,
                OHLCHeader.CLOSE: bar.close,
                OHLCHeader.VOLUME: bar.volume,
                OHLCHeader.VWAP: bar.vwap,
                OHLCHeader.TRANSACTIONS: bar.transactions,
            }
        )

    logger.info("Collected %d bars for %s", len(rows), upper)

    schema: dict[str, pl.DataType | type[pl.DataType]] = {
        OHLCHeader.TIMESTAMP: pl.Datetime("ms", time_zone="UTC"),
        OHLCHeader.OPEN: pl.Float64,
        OHLCHeader.HIGH: pl.Float64,
        OHLCHeader.LOW: pl.Float64,
        OHLCHeader.CLOSE: pl.Float64,
        OHLCHeader.VOLUME: pl.Float64,
        OHLCHeader.VWAP: pl.Float64,
        OHLCHeader.TRANSACTIONS: pl.Int64,
    }
    frame = (
        pl.DataFrame(schema=schema) if not rows else pl.DataFrame(rows, schema=schema)
    )
    if config.timespan == "day" and config.bar_multiplier == 1:
        # Daily bars: stamp the trading day at 00:00 UTC. get_aggs marks daily
        # bars at midnight ET while the grouped-daily (bulk) endpoint marks them
        # mid-session, so normalizing both to the day keeps every fetch path --
        # per-ticker, benchmark, and bulk -- join-aligned on Timestamp.
        frame = frame.with_columns(pl.col(OHLCHeader.TIMESTAMP).dt.truncate("1d"))
    return frame


def get_grouped_daily_column(
    config: AnalysisConfig,
    tickers: Sequence[str],
    ohlc_column: OHLCHeader,
    *,
    drop_missing: bool = True,
    max_workers: int = _GROUPED_DAILY_MAX_WORKERS,
) -> pl.DataFrame:
    """Wide ``{ticker}_{ohlc_column}`` daily panel via the grouped-daily endpoint.

    Pulls the whole market once per trading day (Massive
    ``get_grouped_daily_aggs``) and keeps only *tickers*, so it costs ~one
    request per trading day instead of one per ticker -- the efficient path when
    tickers outnumber the trading days in the window. Daily bars only.

    Returns a ``Timestamp`` column (Datetime, UTC, truncated to the trading day
    to match :func:`get_ohlc`) plus one ``{ticker}_{ohlc_column}`` column per
    ticker.

    Args:
        drop_missing: When *True* (default) drop any day on which a requested
            ticker has no bar, so the panel is inner-join aligned exactly like
            the per-ticker path. Set *False* to keep the null-preserving panel
            (e.g. to screen each ticker's history coverage before filling gaps).
        max_workers: Concurrency for the one-request-per-trading-day fan-out.
    """
    wanted = set(tickers)
    field = _GROUPED_DAILY_FIELD[ohlc_column]
    dates = _weekday_dates(_to_date(config.start_date), _to_date(config.end_date))
    client = get_client(config.api_key)
    logger.info(
        "Fetching grouped-daily %s for %d tickers across %d weekdays",
        ohlc_column,
        len(wanted),
        len(dates),
    )

    def fetch(date: dt.date) -> list[tuple[str, int, float]]:
        results = client.get_grouped_daily_aggs(
            date.isoformat(), adjusted=config.adjusted
        )
        return [
            (bar.ticker, bar.timestamp, getattr(bar, field))
            for bar in results
            if isinstance(bar, GroupedDailyAgg)
            and bar.ticker in wanted
            and bar.timestamp is not None
            and getattr(bar, field) is not None
        ]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        rows = [row for day_rows in pool.map(fetch, dates) for row in day_rows]

    value_columns = [f"{ticker}_{ohlc_column}" for ticker in tickers]
    if not rows:
        empty: dict[str, pl.DataType | type[pl.DataType]] = {
            OHLCHeader.TIMESTAMP: pl.Datetime("ms", time_zone="UTC")
        }
        empty.update({column: pl.Float64 for column in value_columns})
        return pl.DataFrame(schema=empty)

    wide = pl.DataFrame(
        rows,
        schema={
            "Ticker": pl.Utf8,
            OHLCHeader.TIMESTAMP: pl.Int64,
            "Value": pl.Float64,
        },
        orient="row",
    ).pivot(
        on="Ticker",
        index=OHLCHeader.TIMESTAMP,
        values="Value",
        aggregate_function="first",
    )
    # Reattach any ticker that never traded as an all-null column, so the column
    # set (and the inner-join semantics via drop_nulls below) match the
    # per-ticker path -- a fully-absent ticker empties the result, as an empty
    # inner join would.
    missing = [ticker for ticker in tickers if ticker not in wide.columns]
    if missing:
        wide = wide.with_columns(
            [pl.lit(None, dtype=pl.Float64).alias(ticker) for ticker in missing]
        )
    panel = (
        wide.rename({ticker: f"{ticker}_{ohlc_column}" for ticker in tickers})
        .with_columns(
            pl.from_epoch(pl.col(OHLCHeader.TIMESTAMP), time_unit="ms")
            .dt.replace_time_zone("UTC")
            .dt.truncate("1d")
            # from_epoch yields microsecond precision regardless of the input
            # unit; normalize to the declared millisecond dtype so this bulk
            # path stays join-aligned with the per-ticker get_ohlc fetch.
            .dt.cast_time_unit("ms")
            .alias(OHLCHeader.TIMESTAMP)
        )
        .select(OHLCHeader.TIMESTAMP, *value_columns)
        .sort(OHLCHeader.TIMESTAMP, descending=config.sort == "desc")
    )
    return panel.drop_nulls() if drop_missing else panel


# ----------------------------------------------------------------------------
# Clean-panel construction and resolution
# ----------------------------------------------------------------------------
def get_clean_panel(
    config: AnalysisConfig,
    *ohlc_columns: OHLCHeader,
) -> pl.DataFrame:
    """Gap-free multi-OHLC panel for the names that traded the window.

    Keeps screening *out* of the analysis pipeline: build this once, upstream,
    and inject it via ``config.injected_panel``. Fetches each requested column
    null-preserving (so there's no inner-join collapse), keeps names that
    traded on >= *min_coverage* of the window's days (dropping recent IPOs
    and mid-window delistings), and -- for price columns only -- nulls
    non-positive prints and forward/back-fills the gaps. Volume and
    transaction counts keep their non-trading-day nulls so a filled price
    never fabricates turnover. Daily bars only.

    Args:
        config: The universe + fetch window.
        config: Screening thresholds and concurrency.
        ohlc_columns: Columns to include (default ``(Close, Volume)``).
            Coverage is always screened on ``Close``.

    Returns:
        A wide ``Timestamp`` + ``{ticker}_{column}`` panel.

    Raises:
        ValueError: If no ticker clears the coverage threshold.
    """
    columns = list(ohlc_columns) or [OHLCHeader.CLOSE, OHLCHeader.VOLUME]
    close = get_grouped_daily_column(
        config,
        config.tickers,
        OHLCHeader.CLOSE,
        drop_missing=False,
        max_workers=config.max_workers,
    )
    height = close.height
    covered = [
        ticker
        for ticker in config.tickers
        if height > 0
        and 1.0 - close[f"{ticker}_{OHLCHeader.CLOSE}"].null_count() / height
        >= config.min_coverage
    ]
    if not covered:
        raise ValueError(
            f"No tickers cleared the {config.min_coverage:.0%} coverage screen."
        )
    logger.info(
        "Screened %d/%d tickers at >=%.0f%% coverage over %d days",
        len(covered),
        len(config.tickers),
        config.min_coverage * 100,
        height,
    )
    frames = [_clean_panel_column(config, ohlc, covered, close) for ohlc in columns]
    return reduce(
        lambda left, right: left.join(right, on=OHLCHeader.TIMESTAMP, how="inner"),
        frames,
    ).sort(OHLCHeader.TIMESTAMP)


def _clean_panel_column(
    config: AnalysisConfig,
    ohlc_column: OHLCHeader,
    covered: Sequence[str],
    close: pl.DataFrame,
) -> pl.DataFrame:
    """One screened OHLC column for *covered*; fill prices, keep volume gaps."""
    raw = (
        close
        if ohlc_column == OHLCHeader.CLOSE
        else get_grouped_daily_column(
            config,
            covered,
            ohlc_column,
            drop_missing=False,
            max_workers=config.max_workers,
        )
    )
    wanted = [f"{ticker}_{ohlc_column}" for ticker in covered]
    frame = raw.select(OHLCHeader.TIMESTAMP, *wanted)
    if ohlc_column in _PRICE_COLUMNS:
        # Null non-positive prints, then carry the last good price across the
        # gaps so the residual pipeline sees a rectangular panel.
        frame = frame.with_columns(
            pl.when(pl.col(c) > 0).then(pl.col(c)).otherwise(None).alias(c)
            for c in wanted
        ).with_columns(pl.col(wanted).forward_fill().backward_fill())
    return frame


def _prefer_bulk_fetch(config: AnalysisConfig) -> bool:
    """True when the grouped-daily bulk endpoint is the cheaper fetch.

    It serves daily bars only and costs ~one request per trading day, versus
    one per ticker for the per-ticker path -- so it wins for daily bars when
    there are more tickers than trading days in the window.
    """
    if config.timespan != "day" or config.bar_multiplier != 1:
        return False
    trading_days = len(
        _weekday_dates(_to_date(config.start_date), _to_date(config.end_date))
    )
    return len(config.tickers) > trading_days


def _resolve(config: AnalysisConfig) -> pl.DataFrame | None:
    """The panel to slice from: injected, or a lazily-built clean panel.

    With no injected ``panel`` but a ragged universe large enough for the
    bulk endpoint, the first access builds (and caches) a coverage-screened,
    gap-filled :func:`get_clean_panel`, so the modules scale to the universe
    with no manual injection. Small baskets / non-daily bars return ``None``
    and use the per-ticker fetch path (no collapse for full-history baskets).
    """
    if config.injected_panel is not None:
        return config.injected_panel
    if _prefer_bulk_fetch(config):
        if config._resolved_panel is None:
            config._resolved_panel = get_clean_panel(config, OHLCHeader.CLOSE)
        return config._resolved_panel
    return None


# ----------------------------------------------------------------------------
# Universe selection and liquidity
# ----------------------------------------------------------------------------
def _available_tickers(config: AnalysisConfig) -> list[str]:
    """Input ``tickers`` that have data, in original order.

    Every ticker for the per-ticker path or an injected panel covering them
    all; the coverage-screened subset (those that cleared ``min_coverage``)
    once the clean panel is built -- so ``get_tickers`` and ``column`` always
    agree on the surviving universe.
    """
    panel = _resolve(config)
    if panel is None:
        return list(config.tickers)
    columns = set(panel.columns)
    return [t for t in config.tickers if f"{t}_{OHLCHeader.CLOSE}" in columns]


def get_tickers(
    config: AnalysisConfig,
    ticker_class: TickerClass = "all",
) -> list[str]:
    """The ``ticker_class`` slice of the surviving universe.

    ``"all"`` is every available ticker; ``"liquid"`` / ``"illiquid"`` split
    those by whether their median dollar volume clears ``min_dollar_volume``.
    """
    match ticker_class:
        case "all":
            return _available_tickers(config)
        case "liquid":
            return liquid_tickers(config)
        case "illiquid":
            return illiquid_tickers(config)
        case _:
            raise ValueError(f"Invalid ticker class: {ticker_class}")


def liquid_tickers(config: AnalysisConfig) -> list[str]:
    liquidity = get_median_dollar_volume(config)
    return [
        t
        for t in _available_tickers(config)
        if liquidity[t] >= config.min_dollar_volume
    ]


def illiquid_tickers(config: AnalysisConfig) -> list[str]:
    liquidity = get_median_dollar_volume(config)
    return [
        t for t in _available_tickers(config) if liquidity[t] < config.min_dollar_volume
    ]


def get_median_dollar_volume(config: AnalysisConfig) -> dict[str, float]:
    """Median daily dollar volume (Close x Volume) per ticker (cached).

    Computed independently per ticker -- no cross-ticker alignment -- so it
    tolerates heterogeneous histories. ``median`` ignores missing bars, and
    a ticker with no data over the window maps to ``0.0``. Uses ``Close`` and
    ``Volume`` regardless of ``ohlc_column`` (dollar volume is a price x size
    measure). The result is memoized so repeated ``get_tickers`` calls don't
    refetch.
    """
    if config._dollar_volume is None:
        config._dollar_volume = _compute_median_dollar_volume(config)
    return config._dollar_volume


def _compute_median_dollar_volume(config: AnalysisConfig) -> dict[str, float]:
    tickers = _available_tickers(config)
    panel = _resolve(config)
    close_cols = [f"{t}_{OHLCHeader.CLOSE}" for t in tickers]
    volume_cols = [f"{t}_{OHLCHeader.VOLUME}" for t in tickers]
    if panel is not None and all(
        col in panel.columns for col in (*close_cols, *volume_cols)
    ):
        # A panel already carrying Close+Volume -- no fetch. Its Volume keeps
        # its non-trading-day nulls, so filled prices on those days multiply
        # out to null and ``median`` ignores them.
        return _dollar_volume_from(
            panel.select(OHLCHeader.TIMESTAMP, *close_cols),
            panel.select(OHLCHeader.TIMESTAMP, *volume_cols),
            tickers,
        )
    if _prefer_bulk_fetch(config):
        # Close from the resolved panel; Volume fetched null-preserving so
        # the inner join keeps the window rather than collapsing on gaps.
        close = column(config, OHLCHeader.CLOSE, "all")
        volume = get_grouped_daily_column(
            config,
            tickers,
            OHLCHeader.VOLUME,
            drop_missing=False,
            max_workers=config.max_workers,
        )
        return _dollar_volume_from(close, volume, tickers)
    result: dict[str, float] = {}
    for ticker in tickers:
        bars = get_ohlc(ticker, config)
        dollar_volume = (bars[OHLCHeader.CLOSE] * bars[OHLCHeader.VOLUME]).median()
        result[ticker] = (
            float(cast(float, dollar_volume)) if dollar_volume is not None else 0.0
        )
    return result


def _dollar_volume_from(
    close: pl.DataFrame, volume: pl.DataFrame, tickers: Sequence[str]
) -> dict[str, float]:
    """Median (Close x Volume) per ticker from aligned close/volume panels."""
    medians = (
        close.join(volume, on=OHLCHeader.TIMESTAMP, how="inner")
        .select(
            (
                pl.col(f"{ticker}_{OHLCHeader.CLOSE}")
                * pl.col(f"{ticker}_{OHLCHeader.VOLUME}")
            )
            .median()
            .alias(ticker)
            for ticker in tickers
        )
        .row(0)
    )
    return {
        ticker: float(median) if median is not None else 0.0
        for ticker, median in zip(tickers, medians, strict=True)
    }


# ----------------------------------------------------------------------------
# Column access
# ----------------------------------------------------------------------------
def _ticker_column(
    config: AnalysisConfig, ticker: str, ohlc_column: OHLCHeader
) -> pl.DataFrame:
    """``Timestamp`` + a single renamed ``{ticker}_{ohlc_column}`` (one fetch)."""
    renamed = f"{ticker}_{ohlc_column}"
    return (
        get_ohlc(ticker, config)
        .rename({ohlc_column: renamed})
        .select(OHLCHeader.TIMESTAMP, renamed)
    )


def column(
    config: AnalysisConfig,
    ohlc_column: OHLCHeader,
    ticker_class: TickerClass = "all",
    *,
    drop_missing: bool = True,
) -> pl.DataFrame:
    """Wide ``{ticker}_{ohlc_column}`` frame for the *ticker_class* slice.

    Slices the resolved panel (an injected one, or the lazily-built clean
    panel -- see :func:`_resolve`) when it carries every requested column;
    otherwise fetches -- the grouped-daily bulk endpoint when tickers
    outnumber the window's trading days, else a per-ticker join. Every path
    returns the same wide, time-aligned frame: a ``Timestamp`` column plus one
    column per ticker.

    With ``drop_missing=False`` the fetch paths keep every trading day *any*
    ticker traded (a full outer union, null where a ticker has no bar) instead
    of inner-joining down to the days they all share -- so a single sparse
    ticker can't collapse the panel. Use it for per-ticker analyses (e.g.
    Fama-French) that align each ticker to shared exogenous data rather than to
    one another; the inner-join default stays for analyses that need a common
    grid across tickers.
    """
    tickers = get_tickers(config, ticker_class)
    wanted = [f"{ticker}_{ohlc_column}" for ticker in tickers]
    panel = _resolve(config)
    if panel is not None and all(c in panel.columns for c in wanted):
        return panel.select(OHLCHeader.TIMESTAMP, *wanted)
    if _prefer_bulk_fetch(config):
        return get_grouped_daily_column(
            config,
            tickers,
            ohlc_column,
            drop_missing=drop_missing,
            max_workers=config.max_workers,
        )
    how: Literal["inner", "full"] = "inner" if drop_missing else "full"
    return reduce(
        lambda left, right: left.join(
            right, on=OHLCHeader.TIMESTAMP, how=how, coalesce=True
        ),
        [_ticker_column(config, ticker, ohlc_column) for ticker in tickers],
    )


def benchmark_column(
    config: AnalysisConfig,
    ohlc_column: OHLCHeader,
) -> pl.DataFrame:
    """``Timestamp`` + ``{benchmark_ticker}_{ohlc_column}`` (panel or fetch)."""
    if config.benchmark_ticker is None:
        raise ValueError("benchmark_ticker is not set")
    wanted = f"{config.benchmark_ticker}_{ohlc_column}"
    panel = _resolve(config)
    if panel is not None and wanted in panel.columns:
        return panel.select(OHLCHeader.TIMESTAMP, wanted)
    return _ticker_column(config, config.benchmark_ticker, ohlc_column)


# ----------------------------------------------------------------------------
# OHLC column and log-return accessors
# ----------------------------------------------------------------------------
def get_ohlc_column_for_tickers(
    config: AnalysisConfig,
    ticker_class: TickerClass = "all",
    *,
    drop_missing: bool = True,
) -> pl.DataFrame:
    return column(config, config.ohlc_column, ticker_class, drop_missing=drop_missing)


def get_ohlc_column_for_benchmark(config: AnalysisConfig) -> pl.DataFrame:
    return benchmark_column(config, config.ohlc_column)


def get_log_return_for_tickers(
    config: AnalysisConfig,
    ticker_class: TickerClass = "all",
) -> pl.DataFrame:
    tickers = get_tickers(config, ticker_class)
    return (
        get_ohlc_column_for_tickers(config, ticker_class)
        .select(
            OHLCHeader.TIMESTAMP,
            *[
                (
                    pl.col(f"{ticker}_{config.ohlc_column}").log()
                    - pl.col(f"{ticker}_{config.ohlc_column}").log().shift(1)
                ).alias(f"{ticker}_{_LOG_RETURN_COLUMN_NAME}")
                for ticker in tickers
            ],
        )
        .drop_nulls()
    )


def get_log_return_for_benchmark(config: AnalysisConfig) -> pl.DataFrame:
    benchmark = config.benchmark_ticker
    if benchmark is None:
        raise ValueError("benchmark_ticker is not set")
    return get_ohlc_column_for_benchmark(config).select(
        OHLCHeader.TIMESTAMP,
        (
            pl.col(f"{benchmark}_{config.ohlc_column}").log()
            - pl.col(f"{benchmark}_{config.ohlc_column}").log().shift(1)
        ).alias(f"{benchmark}_{_LOG_RETURN_COLUMN_NAME}"),
    )


# ----------------------------------------------------------------------------
# Residualization
# ----------------------------------------------------------------------------
def rolling_pca_residuals(
    returns: np.ndarray, window: int, n_components: int
) -> np.ndarray:
    """Rolling rank-``n_components`` PCA reconstruction-error residuals.

    *returns* is a ``(T, N)`` array (rows = bars, columns = assets). On each
    trailing window of ``window`` rows the top ``n_components`` principal
    components of the basket are its common (market + leading systematic)
    factors, so each asset's residual is the part of its return they leave
    unexplained -- the rank-``n_components`` reconstruction error of the
    most-recent bar.

    Returns a ``(T, N)`` array whose first ``window - 1`` rows are NaN (warmup,
    before a full window exists).
    """
    residuals = np.full(returns.shape, np.nan)
    for end in range(window, returns.shape[0] + 1):
        block = returns[end - window : end]
        pca = PCA(n_components=n_components)
        reconstructed = pca.inverse_transform(pca.fit_transform(block))
        residuals[end - 1] = (block - reconstructed)[-1]
    return residuals


def _require_window(window: int | None, field: str) -> int:
    """Return *window*, or raise if this analysis left the optional field unset."""
    if window is None:
        raise ValueError(f"AnalysisConfig.{field} must be set for this analysis")
    return window


def get_pca_residuals(
    config: AnalysisConfig,
    ticker_class: TickerClass = "all",
) -> pl.DataFrame:
    """Rolling PCA reconstruction-error residuals per ticker.

    Removes the leading ``n_components`` principal components (default 1, the
    market factor) of the basket over each trailing ``residualization_window``;
    raising ``n_components`` past ~1-2 also strips the sector modes clustering
    relies on.
    """
    window = _require_window(config.residualization_window, "residualization_window")
    tickers = get_tickers(config, ticker_class)
    log_returns = get_log_return_for_tickers(config, ticker_class)
    return_columns = [f"{t}_{_LOG_RETURN_COLUMN_NAME}" for t in tickers]
    returns = log_returns.select(return_columns).to_numpy()
    residuals = rolling_pca_residuals(returns, window, config.n_components)
    residual_columns = [
        pl.Series(f"{ticker}_{_PCA_RESIDUAL_COLUMN_NAME}", residuals[:, index])
        for index, ticker in enumerate(tickers)
    ]
    return (
        log_returns.select(OHLCHeader.TIMESTAMP)
        .with_columns(residual_columns)
        .filter(pl.col(f"{tickers[0]}_{_PCA_RESIDUAL_COLUMN_NAME}").is_not_nan())
    )


def get_normalized_pca_residuals(
    config: AnalysisConfig,
    ticker_class: TickerClass = "all",
) -> pl.DataFrame:
    window = _require_window(config.normalization_window, "normalization_window")
    tickers = get_tickers(config, ticker_class)
    return get_pca_residuals(config, ticker_class).select(
        OHLCHeader.TIMESTAMP,
        *z_score_normalize(
            window=window,
            columns_to_normalize=[
                f"{ticker}_{_PCA_RESIDUAL_COLUMN_NAME}" for ticker in tickers
            ],
        ),
    )


def get_ols_residuals(
    config: AnalysisConfig,
    ticker_class: TickerClass = "all",
) -> pl.DataFrame:
    benchmark = config.benchmark_ticker
    assert benchmark is not None, "benchmark_ticker is required for OLS residuals"
    tickers = get_tickers(config, ticker_class)

    # get_log_return_for_tickers covers only ``tickers``; join the
    # benchmark's return so both are computed and time-aligned in one frame.
    log_returns = (
        get_log_return_for_tickers(config, ticker_class)
        .join(
            get_log_return_for_benchmark(config),
            on=OHLCHeader.TIMESTAMP,
            how="inner",
        )
        .drop_nulls()
    )
    ticker_columns = [f"{t}_{_LOG_RETURN_COLUMN_NAME}" for t in tickers]
    ticker_returns = log_returns.select(ticker_columns).to_numpy()
    benchmark_returns = log_returns[f"{benchmark}_{_LOG_RETURN_COLUMN_NAME}"].to_numpy()
    window = _require_window(config.residualization_window, "residualization_window")

    # Rolling OLS (market-model) residualization: regress each ticker's
    # return on the benchmark's (with an intercept) over each trailing
    # window; the residual is the part of the ticker's most-recent return
    # the benchmark does not explain.
    residuals = np.full(ticker_returns.shape, np.nan)
    for end in range(window, ticker_returns.shape[0] + 1):
        design = np.column_stack(
            [np.ones(window), benchmark_returns[end - window : end]]
        )
        target = ticker_returns[end - window : end]
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        residuals[end - 1] = target[-1] - design[-1] @ coefficients

    residual_columns = [
        pl.Series(f"{ticker}_{_OLS_RESIDUAL_COLUMN_NAME}", residuals[:, index])
        for index, ticker in enumerate(tickers)
    ]
    return (
        log_returns.select(OHLCHeader.TIMESTAMP)
        .with_columns(residual_columns)
        .filter(pl.col(f"{tickers[0]}_{_OLS_RESIDUAL_COLUMN_NAME}").is_not_nan())
    )


def get_normalized_ols_residuals(
    config: AnalysisConfig,
    ticker_class: TickerClass = "all",
) -> pl.DataFrame:
    window = _require_window(config.normalization_window, "normalization_window")
    tickers = get_tickers(config, ticker_class)
    return get_ols_residuals(config, ticker_class).select(
        OHLCHeader.TIMESTAMP,
        *z_score_normalize(
            window=window,
            columns_to_normalize=[
                f"{ticker}_{_OLS_RESIDUAL_COLUMN_NAME}" for ticker in tickers
            ],
        ),
    )


# ----------------------------------------------------------------------------
# Similarity and normalization utilities
# ----------------------------------------------------------------------------
def compute_similarity_matrix(data: pl.DataFrame, sim_metric: str) -> pl.DataFrame:
    """Pairwise asset similarity matrix from a wide residual/return frame.

    Non-numeric columns (e.g. ``Timestamp``) are dropped and rows with any
    missing value are removed, so every pair is measured on the same aligned
    (complete-case) sample.

    Args:
        data: Wide frame of one numeric column per asset (plus an ignored
            ``Timestamp``), e.g. normalized PCA/OLS residuals.
        sim_metric: ``"corr"`` for the signed Pearson correlation R, or
            ``"aff"`` for the affinity ``0.5 * (1 + R)`` mapped to ``[0, 1]``
            (identical -> 1, uncorrelated -> 0.5, opposite -> 0) -- a valid
            non-negative similarity for graph methods like spectral clustering.

    Returns:
        A square :class:`polars.DataFrame` whose columns/rows are the asset
        columns in input order.

    Raises:
        ValueError: If *sim_metric* is unknown, *data* has no numeric columns,
            or fewer than two aligned observations remain.
    """
    values = data.select(cs.numeric()).drop_nulls()
    if values.width == 0:
        raise ValueError("data has no numeric asset columns to compare.")
    if values.height < 2:
        raise ValueError(
            "Need at least 2 aligned observations across all assets to "
            "compute correlations."
        )
    corr = values.corr().to_numpy()
    match sim_metric:
        case "corr":
            similarity = corr
        case "aff":
            similarity = 0.5 * (1.0 + corr)
        case _:
            raise ValueError(f"Unknown similarity metric: {sim_metric}")
    return pl.DataFrame(similarity, schema=values.columns, orient="row")


def z_score_normalize(
    window: int, columns_to_normalize: Sequence[str]
) -> list[pl.Expr]:
    # Rolling standardization. A longer window gives a slower-moving baseline,
    # which adds far fewer spurious direction changes than a short one -- and
    # measured smoother than an EWM mean, which (being more responsive) acts as
    # a high-pass filter and adds jaggedness. A locally-flat window has
    # rolling_std == 0 (0 / 0 = NaN); fill with 0 so drop_nulls() (which
    # ignores NaN) does not let it leak through.
    return [
        (
            (pl.col(col) - pl.col(col).rolling_mean(window_size=window))
            / pl.col(col).rolling_std(window_size=window)
        )
        .fill_nan(0.0)
        .alias(col)
        for col in columns_to_normalize
    ]
