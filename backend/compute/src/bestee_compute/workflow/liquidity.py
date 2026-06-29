"""Federal Reserve Net Liquidity from FRED, free of publication-lag bias.

Net Liquidity (in millions of USD) is::

    WALCL  -  TGA (WTREGEN)  -  RRP (RRPONTSYD)

* ``WALCL`` — Fed total assets (Wednesday level, millions, weekly).
* ``WTREGEN`` — Treasury General Account (week average, millions, weekly).
* ``RRPONTSYD`` — Overnight reverse repos (billions, daily).

Each component is fetched from FRED, shifted forward by its true
publication lag so a given trading session only ever sees data that was
already released, aligned onto a business-day calendar via an as-of
(last-known-value) join, then combined into the level series and a
stationary rate-of-change feature suitable for a regime model (e.g. HMM).
"""

import logging
import os
from dataclasses import dataclass
from datetime import date

import httpx
import polars as pl
from dotenv import find_dotenv, load_dotenv

logger = logging.getLogger(__name__)

FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"

# Output column names.
COL_DATE = "date"
COL_NET_LIQUIDITY = "net_liquidity"
COL_LIQUIDITY_ROC = "liquidity_roc"


@dataclass(frozen=True)
class LiquidityComponent:
    """A single FRED series feeding the net-liquidity equation.

    Attributes:
        series_id: FRED series identifier (e.g. ``"WALCL"``).
        publication_lag_days: Calendar days between a series' observation
            date and the first trading session that may act on it, given
            the Fed's release schedule.  Shifting by this lag is what
            removes look-ahead bias.
        unit_scale: Multiplier that brings the series into a common unit
            (millions of USD).  ``1.0`` for series already in millions;
            ``1_000.0`` for series reported in billions.
    """

    series_id: str
    publication_lag_days: int
    unit_scale: float = 1.0


def _resolve_api_key(api_key: str | None) -> str:
    """Return *api_key* or fall back to the ``FRED_API_KEY`` env var.

    Loads ``.env`` and ``.env.local`` (the latter is where this project
    keeps ``FRED_API_KEY``) before reading the environment, mirroring
    :func:`bestee_compute.client.get_client`.
    """
    if api_key:
        return api_key
    load_dotenv()
    local = find_dotenv(".env.local", usecwd=True)
    if local:
        load_dotenv(local)
    key = os.environ.get("FRED_API_KEY")
    if not key:
        msg = (
            "No FRED API key provided. Pass 'api_key' or set the "
            "FRED_API_KEY environment variable (e.g. in compute/.env.local)."
        )
        logger.error("No FRED API key found in parameter or FRED_API_KEY env var")
        raise RuntimeError(msg)
    return key


class FederalNetLiquidityPipeline:
    """Compute Fed Net Liquidity from FRED without look-ahead bias.

    Args:
        api_key: FRED API key.  Falls back to the ``FRED_API_KEY``
            environment variable when *None*.
        components: Override the FRED series / publication lags / unit
            scales.  Must define the ``"WALCL"``, ``"TGA"`` and ``"RRP"``
            keys used by the equation.  Defaults to
            :attr:`DEFAULT_COMPONENTS`.
        observation_start: Earliest observation date to request from
            FRED.  *None* fetches full history.
        roc_window: Look-back window, in business days, for the
            ``liquidity_roc`` feature.  Defaults to ``5`` (one trading
            week); e.g. ``21`` ≈ one month.
        log_returns: If *True* (default), ``liquidity_roc`` is the log
            return ``ln(NLₜ / NLₜ₋ₙ)``; if *False* it is the simple
            percentage change ``NLₜ / NLₜ₋ₙ − 1``.  Log returns are
            additive across time and tend to suit Gaussian-emission HMMs.
        timeout: Per-request timeout in seconds.

    Raises:
        RuntimeError: If no API key is available.
        ValueError: If *components* omits a required key or *roc_window*
            is not positive.
    """

    # WALCL & WTREGEN are Wednesday levels released Thursday ~16:30 ET, so
    # the first fully tradable session is Friday (+2 calendar days).
    # RRPONTSYD is daily and usable the next session (+1 calendar day).
    DEFAULT_COMPONENTS: dict[str, LiquidityComponent] = {
        "WALCL": LiquidityComponent("WALCL", publication_lag_days=2),
        "TGA": LiquidityComponent("WTREGEN", publication_lag_days=2),
        "RRP": LiquidityComponent(
            "RRPONTSYD", publication_lag_days=1, unit_scale=1_000.0
        ),
    }
    _REQUIRED_KEYS = ("WALCL", "TGA", "RRP")

    def __init__(
        self,
        api_key: str | None = None,
        *,
        components: dict[str, LiquidityComponent] | None = None,
        observation_start: date | None = None,
        roc_window: int = 5,
        log_returns: bool = True,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = _resolve_api_key(api_key)
        self.components = (
            dict(components) if components else dict(self.DEFAULT_COMPONENTS)
        )
        missing = [k for k in self._REQUIRED_KEYS if k not in self.components]
        if missing:
            msg = f"components must define {self._REQUIRED_KEYS}; missing {missing}"
            raise ValueError(msg)
        if roc_window < 1:
            raise ValueError("roc_window must be a positive integer")
        self.observation_start = observation_start
        self.roc_window = roc_window
        self.log_returns = log_returns
        self.timeout = timeout

    def _fetch_series(
        self,
        name: str,
        component: LiquidityComponent,
        client: httpx.Client,
    ) -> pl.DataFrame:
        """Fetch one FRED series as a publication-shifted ``[date, name]`` frame.

        Args:
            name: Internal component name, used as the value column label.
            component: The series' id, publication lag and unit scale.
            client: An open :class:`httpx.Client`.

        Returns:
            A Polars DataFrame sorted ascending by ``date`` (the first
            session each value is tradable) with the scaled value stored
            under *name* (in millions of USD).

        Raises:
            httpx.HTTPError: If the FRED request fails.
            ValueError: If FRED returns no usable observations.
        """
        params = {
            "series_id": component.series_id,
            "api_key": self.api_key,
            "file_type": "json",
        }
        if self.observation_start is not None:
            params["observation_start"] = self.observation_start.isoformat()

        logger.info("Fetching FRED series %s", component.series_id)
        response = client.get(FRED_OBSERVATIONS_URL, params=params)
        response.raise_for_status()
        observations = response.json().get("observations", [])

        frame = (
            pl.DataFrame(
                {
                    "raw_date": [o["date"] for o in observations],
                    "raw_value": [o.get("value") for o in observations],
                },
                schema={"raw_date": pl.Utf8, "raw_value": pl.Utf8},
            )
            .with_columns(
                pl.col("raw_date").str.to_date(),
                # FRED encodes missing values as "."; cast(strict=False)
                # turns those into null, and drop_nulls removes them so
                # only genuine observations define the step function.
                pl.col("raw_value").cast(pl.Float64, strict=False),
            )
            .drop_nulls()
            .select(
                (
                    pl.col("raw_date")
                    + pl.duration(days=component.publication_lag_days)
                ).alias(COL_DATE),
                (pl.col("raw_value") * component.unit_scale).alias(name),
            )
            .sort(COL_DATE)
        )

        if frame.is_empty():
            msg = f"FRED returned no usable observations for {component.series_id}"
            raise ValueError(msg)
        logger.info(
            "Fetched %d observations for %s (lag=+%dd)",
            frame.height,
            component.series_id,
            component.publication_lag_days,
        )
        return frame

    def _roc_expr(self) -> pl.Expr:
        """Rate-of-change expression for ``net_liquidity``.

        Log return ``ln(NLₜ / NLₜ₋ₙ)`` by default (additive across time
        and well-suited to Gaussian-emission HMMs), or simple percentage
        change ``NLₜ / NLₜ₋ₙ − 1`` when ``log_returns`` is *False*.  Net
        liquidity is positive by construction, so the log is defined.
        """
        net_liquidity = pl.col(COL_NET_LIQUIDITY)
        if self.log_returns:
            return net_liquidity.log().diff(n=self.roc_window)
        return net_liquidity.pct_change(n=self.roc_window)

    def generate_liquidity_anchor(self) -> pl.DataFrame:
        """Build the look-ahead-safe net-liquidity feature frame.

        Returns:
            A Polars DataFrame with one row per business day and columns:

            * ``date`` — the trading session (business day).
            * ``net_liquidity`` — WALCL − TGA − RRP, in millions of USD.
            * ``liquidity_roc`` — net liquidity's ``roc_window``-session
              change: a log return by default, or simple percentage
              change when ``log_returns=False`` (a stationary feature).

        Raises:
            httpx.HTTPError: If any FRED request fails.
            ValueError: If a series returns no usable observations.
        """
        with httpx.Client(timeout=self.timeout) as client:
            frames = {
                name: self._fetch_series(name, component, client)
                for name, component in self.components.items()
            }

        # Calendar bounds span *every* component (the original only looked
        # at two of the three, clipping data when TGA led or lagged).
        all_dates = pl.concat([f.select(COL_DATE) for f in frames.values()])[COL_DATE]
        start, end = all_dates.min(), all_dates.max()
        # Non-null because every frame is guaranteed non-empty above.
        assert isinstance(start, date) and isinstance(end, date)

        # Business-day calendar (Mon–Fri), matching pandas freq="B".
        calendar = (
            pl.date_range(start, end, interval="1d", eager=True)
            .alias(COL_DATE)
            .to_frame()
            .filter(pl.col(COL_DATE).dt.weekday() <= 5)
        )

        # As-of join each component: every session takes the most recent
        # already-published value. This forward-fills without manufacturing
        # weekend rows or dropping weekend-shifted observations (a bug in
        # the fixed-Timedelta-shift + business-day reindex approach).
        master = calendar
        for name in self.components:
            master = master.join_asof(frames[name], on=COL_DATE, strategy="backward")

        master = (
            # Drop warm-up sessions before all three components exist.
            master.drop_nulls()
            .with_columns(
                (pl.col("WALCL") - pl.col("TGA") - pl.col("RRP")).alias(
                    COL_NET_LIQUIDITY
                )
            )
            .with_columns(self._roc_expr().alias(COL_LIQUIDITY_ROC))
            .select(COL_DATE, COL_NET_LIQUIDITY, COL_LIQUIDITY_ROC)
            .drop_nulls()
        )
        logger.info("Built net-liquidity frame with %d rows", master.height)
        return master
