"""Options analytics built on the Massive API.

Exposes the delta-based normalized implied-volatility skew (a.k.a. risk
reversal), ``(Call_IV - Put_IV) / Denominator_IV``, in two modes:

* **snapshot** — greeks and IV straight from Massive's real-time option
  chain snapshot (fast, current).
* **historical** — point-in-time IV reconstructed from the option's daily
  close on the quote date via Black–Scholes inversion (see
  :mod:`bestee_compute.securities.black_scholes`), using a risk-free rate
  from FRED (:func:`bestee_compute.securities.rates.get_risk_free_rate`)
  and a configurable dividend yield.  This avoids the look-ahead bias of
  reading today's greeks for a past date.

The legs can be taken from the single nearest contract/expiration or
**interpolated** to the exact target delta (across the smile) and exact
target days-to-expiration (across the term structure, in total variance).
Results are returned as a :class:`VolatilitySkewResult`; a batch helper
(:func:`compute_volatility_skew_frame`) and a Great Tables display
(:func:`volatility_skew_table`) are provided for cross-sections.
"""

import datetime
import logging
import math
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

import polars as pl
from great_tables import GT
from massive import RESTClient
from massive.rest.models.aggs import Agg
from massive.rest.models.contracts import OptionsContract
from massive.rest.models.snapshot import OptionContractSnapshot

from bestee_compute.client import get_client
from bestee_compute.securities.black_scholes import (
    OptionType,
    bs_delta,
    implied_volatility,
)
from bestee_compute.securities.rates import get_risk_free_rate

logger = logging.getLogger(__name__)

Mode = Literal["auto", "snapshot", "historical"]

_DAYS_PER_YEAR = 365.0
_MAX_WORKERS = 10


# ── Result types ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class OptionLeg:
    """One leg (call / put / denominator) of the skew calculation.

    Attributes:
        ticker: The option contract used, or *None* when the leg is
            interpolated (synthetic) rather than a single contract.
        contract_type: ``"call"`` or ``"put"``.
        target_delta: The requested delta.
        delta: The leg's actual delta (equals ``target_delta`` when
            interpolated).
        implied_volatility: The leg's implied volatility (decimal).
    """

    ticker: str | None
    contract_type: str
    target_delta: float
    delta: float
    implied_volatility: float


@dataclass(frozen=True)
class VolatilitySkewResult:
    """Full output of a normalized-volatility-skew computation."""

    underlying_ticker: str
    quote_date: datetime.date
    target_expiration: datetime.date
    expiration: datetime.date
    days_to_expiration: int
    underlying_price: float | None
    call: OptionLeg
    put: OptionLeg
    denominator: OptionLeg
    skew: float
    mode: str
    interpolated: bool


@dataclass(frozen=True)
class _Quote:
    """A usable (delta, IV) point on one expiration's smile."""

    contract_type: str
    ticker: str
    delta: float
    implied_volatility: float


# ── Helpers: option type, time ───────────────────────────────────────


def _as_option_type(value: str) -> OptionType:
    """Narrow a raw contract-type string to the typed literal."""
    if value == "call":
        return "call"
    if value == "put":
        return "put"
    raise ValueError(f"Unexpected contract type {value!r}.")


def _year_fraction(start: datetime.date, end: datetime.date) -> float:
    """Years between two dates (floored at one day to stay positive)."""
    return max((end - start).days, 1) / _DAYS_PER_YEAR


def _resolve_mode(
    mode: Mode, quote_date: datetime.date
) -> Literal["snapshot", "historical"]:
    if mode != "auto":
        return mode
    return "historical" if quote_date < datetime.date.today() else "snapshot"


# ── Expiration resolution ────────────────────────────────────────────


def _bracketing_expirations(
    client: RESTClient,
    underlying_ticker: str,
    target: datetime.date,
    *,
    as_of: datetime.date | None,
    expired: bool | None,
) -> tuple[datetime.date | None, datetime.date | None]:
    """Return (latest expiration <= target, earliest expiration >= target)."""

    def _first(
        order: str,
        *,
        gte: datetime.date | None = None,
        lte: datetime.date | None = None,
    ) -> datetime.date | None:
        results = client.list_options_contracts(
            underlying_ticker=underlying_ticker,
            expiration_date_gte=gte,
            expiration_date_lte=lte,
            as_of=as_of,
            expired=expired,
            sort="expiration_date",
            order=order,
            limit=1,
        )
        for contract in results:
            if isinstance(contract, OptionsContract) and contract.expiration_date:
                return datetime.date.fromisoformat(contract.expiration_date)
        return None

    backward = _first("desc", lte=target)
    forward = _first("asc", gte=target)
    return backward, forward


def _nearest_active_expiration(
    client: RESTClient,
    underlying_ticker: str,
    target: datetime.date,
) -> datetime.date:
    """Return the active expiration date closest to *target*."""
    backward, forward = _bracketing_expirations(
        client, underlying_ticker, target, as_of=None, expired=False
    )
    candidates = [d for d in (backward, forward) if d is not None]
    if not candidates:
        raise ValueError(f"No active option contracts found for {underlying_ticker!r}.")
    return min(candidates, key=lambda d: (abs((d - target).days), -d.toordinal()))


def _resolve_expirations(
    client: RESTClient,
    underlying_ticker: str,
    target: datetime.date,
    *,
    as_of: datetime.date | None,
    expired: bool | None,
    interpolate: bool,
) -> list[datetime.date]:
    """Pick the expirations to evaluate: nearest one, or the bracket."""
    backward, forward = _bracketing_expirations(
        client, underlying_ticker, target, as_of=as_of, expired=expired
    )
    present = [d for d in (backward, forward) if d is not None]
    if not present:
        raise ValueError(
            f"No option contracts found for {underlying_ticker!r} near {target}."
        )
    if not interpolate:
        nearest = min(present, key=lambda d: (abs((d - target).days), -d.toordinal()))
        return [nearest]
    # Distinct, in ascending order (may be a single expiration if target
    # coincides with one or only one side exists).
    return sorted(set(present))


# ── Surface builders (mode-specific) ─────────────────────────────────


def _spread_ok(snap: OptionContractSnapshot, max_relative_spread: float) -> bool:
    quote = snap.last_quote
    if quote is None or quote.bid is None or quote.ask is None:
        return False  # cannot assess liquidity -> exclude when filtering
    mid = 0.5 * (quote.bid + quote.ask)
    if mid <= 0.0:
        return False
    return (quote.ask - quote.bid) / mid <= max_relative_spread


def _snapshot_surface(
    client: RESTClient,
    underlying_ticker: str,
    expiration: datetime.date,
    *,
    min_open_interest: int,
    max_relative_spread: float | None,
) -> tuple[list[_Quote], float | None]:
    """Build one expiration's smile from the real-time chain snapshot."""
    quotes: list[_Quote] = []
    underlying_price: float | None = None
    snapshots = client.list_snapshot_options_chain(
        underlying_ticker,
        params={"expiration_date": expiration.isoformat(), "limit": 250},
    )
    for snap in snapshots:
        if not isinstance(snap, OptionContractSnapshot):
            continue
        details, greeks = snap.details, snap.greeks
        if (
            details is None
            or details.ticker is None
            or details.contract_type is None
            or greeks is None
            or greeks.delta is None
            or snap.implied_volatility is None
        ):
            continue
        if snap.underlying_asset and snap.underlying_asset.price is not None:
            underlying_price = snap.underlying_asset.price
        if min_open_interest > 0 and (
            snap.open_interest is None or snap.open_interest < min_open_interest
        ):
            continue
        if max_relative_spread is not None and not _spread_ok(
            snap, max_relative_spread
        ):
            continue
        quotes.append(
            _Quote(
                contract_type=details.contract_type,
                ticker=details.ticker,
                delta=greeks.delta,
                implied_volatility=snap.implied_volatility,
            )
        )
    return quotes, underlying_price


def _daily_bar(
    client: RESTClient, ticker: str, day: datetime.date
) -> tuple[float, float] | None:
    """Return ``(close, volume)`` for *ticker* on *day*, or *None*."""
    for agg in client.get_aggs(ticker, 1, "day", day, day):
        if isinstance(agg, Agg) and agg.close is not None:
            return agg.close, (agg.volume or 0.0)
    return None


def _historical_surface(
    client: RESTClient,
    underlying_ticker: str,
    expiration: datetime.date,
    quote_date: datetime.date,
    *,
    rate: float,
    dividend_yield: float,
    strike_band: float,
    min_volume: float,
) -> tuple[list[_Quote], float | None]:
    """Reconstruct one expiration's smile point-in-time via BS inversion."""
    underlying_bar = _daily_bar(client, underlying_ticker, quote_date)
    if underlying_bar is None:
        raise ValueError(
            f"No daily price for {underlying_ticker!r} on {quote_date}; cannot "
            "reconstruct historical implied volatility."
        )
    spot = underlying_bar[0]
    t_years = _year_fraction(quote_date, expiration)

    contracts: list[OptionsContract] = []
    listing = client.list_options_contracts(
        underlying_ticker=underlying_ticker,
        expiration_date=expiration,
        as_of=quote_date,
        strike_price_gte=spot * (1.0 - strike_band),
        strike_price_lte=spot * (1.0 + strike_band),
        expired=None,
        limit=250,
    )
    for contract in listing:
        if (
            isinstance(contract, OptionsContract)
            and contract.ticker
            and contract.contract_type
            and contract.strike_price is not None
        ):
            contracts.append(contract)

    def _price(
        contract: OptionsContract,
    ) -> tuple[OptionsContract, tuple[float, float] | None]:
        assert contract.ticker is not None
        return contract, _daily_bar(client, contract.ticker, quote_date)

    quotes: list[_Quote] = []
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        for contract, bar in pool.map(_price, contracts):
            if bar is None or contract.ticker is None or contract.strike_price is None:
                continue
            price, volume = bar
            if volume < min_volume:
                continue
            option_type = _as_option_type(str(contract.contract_type))
            iv = implied_volatility(
                price,
                spot,
                contract.strike_price,
                t_years,
                rate,
                dividend_yield,
                option_type,
            )
            if iv is None:
                continue
            delta = bs_delta(
                spot,
                contract.strike_price,
                t_years,
                rate,
                dividend_yield,
                iv,
                option_type,
            )
            quotes.append(
                _Quote(
                    contract_type=option_type,
                    ticker=contract.ticker,
                    delta=delta,
                    implied_volatility=iv,
                )
            )
    return quotes, spot


# ── Leg selection & interpolation ────────────────────────────────────


def _interp_iv_in_delta(pool: list[_Quote], target_delta: float) -> float:
    """Linearly interpolate IV at *target_delta* across a sorted smile."""
    deltas = [q.delta for q in pool]
    ivs = [q.implied_volatility for q in pool]
    if target_delta <= deltas[0]:
        return ivs[0]
    if target_delta >= deltas[-1]:
        return ivs[-1]
    for i in range(1, len(pool)):
        if deltas[i] >= target_delta:
            d0, d1 = deltas[i - 1], deltas[i]
            v0, v1 = ivs[i - 1], ivs[i]
            weight = (target_delta - d0) / (d1 - d0)
            return v0 + weight * (v1 - v0)
    return ivs[-1]


def _select_leg(
    quotes: list[_Quote],
    contract_type: str,
    target_delta: float,
    *,
    interpolate: bool,
) -> OptionLeg:
    """Choose (or interpolate) the leg of *contract_type* at *target_delta*."""
    pool = sorted(
        (q for q in quotes if q.contract_type == contract_type),
        key=lambda q: q.delta,
    )
    if not pool:
        raise ValueError(
            f"No {contract_type} contracts with greeks/IV to match "
            f"delta {target_delta:+.2f}."
        )
    if not interpolate:
        best = min(pool, key=lambda q: abs(q.delta - target_delta))
        return OptionLeg(
            ticker=best.ticker,
            contract_type=contract_type,
            target_delta=target_delta,
            delta=best.delta,
            implied_volatility=best.implied_volatility,
        )
    iv = _interp_iv_in_delta(pool, target_delta)
    return OptionLeg(
        ticker=None,
        contract_type=contract_type,
        target_delta=target_delta,
        delta=target_delta,
        implied_volatility=iv,
    )


def _term_interp_iv(
    iv_lo: float, t_lo: float, iv_hi: float, t_hi: float, t_target: float
) -> float:
    """Interpolate IV across expirations, linear in total variance vs time."""
    if t_hi == t_lo:
        return iv_lo
    w_lo = iv_lo * iv_lo * t_lo
    w_hi = iv_hi * iv_hi * t_hi
    weight = (t_target - t_lo) / (t_hi - t_lo)
    w_target = max(w_lo + weight * (w_hi - w_lo), 1e-12)
    return math.sqrt(w_target / t_target)


def _denominator_type(denominator_delta: float) -> str:
    return "call" if denominator_delta >= 0.0 else "put"


# ── Public API ───────────────────────────────────────────────────────


def compute_normalized_volatility_skew_result(
    underlying_ticker: str,
    quote_date: datetime.date,
    lookahead_days: int,
    put_delta: float,
    call_delta: float,
    denominator_delta: float,
    *,
    mode: Mode = "auto",
    interpolate: bool = True,
    min_open_interest: int = 0,
    max_relative_spread: float | None = None,
    risk_free_rate: float | None = None,
    dividend_yield: float = 0.0,
    strike_band: float = 0.5,
    min_volume: float = 1.0,
    api_key: str | None = None,
    fred_api_key: str | None = None,
) -> VolatilitySkewResult:
    """Compute the delta-based normalized IV skew with full detail.

    The skew is ``(Call_IV - Put_IV) / Denominator_IV`` for options
    targeting ``quote_date + lookahead_days``.  See the
    module docstring for the snapshot vs historical modes and the
    interpolation behaviour.

    Args:
        underlying_ticker: Underlying equity/ETF symbol, e.g. ``"AAPL"``.
        quote_date: Quote/anchor date.
        lookahead_days: Calendar days to the target expiration (> 0).
        put_delta: Target (signed, negative) delta for the put leg.
        call_delta: Target (signed, positive) delta for the call leg.
        denominator_delta: Target delta for the normalizing leg; its sign
            selects call (``>= 0``) vs put.
        mode: ``"auto"`` (historical iff the quote date is in the past),
            ``"snapshot"``, or ``"historical"``.
        interpolate: If *True*, interpolate IV to the exact target delta
            (smile) and target days-to-expiration (term structure); if
            *False*, use the single nearest contract and expiration.
        min_open_interest: Drop snapshot contracts below this open
            interest (``0`` disables).
        max_relative_spread: Drop snapshot contracts whose ``(ask-bid)/mid``
            exceeds this (``None`` disables).
        risk_free_rate: Historical mode only; the continuously compounded
            rate as a decimal.  *None* fetches it from FRED as of the
            quote date.
        dividend_yield: Historical mode only; continuous dividend yield.
        strike_band: Historical mode only; fraction of spot around which
            to consider strikes (``0.5`` -> ±50%).
        min_volume: Historical mode only; minimum daily contract volume.
        api_key: Massive API key.  Falls back to ``MASSIVE_API_KEY``.
        fred_api_key: FRED API key for the risk-free rate.  Falls back to
            ``FRED_API_KEY``.

    Returns:
        A :class:`VolatilitySkewResult`.

    Raises:
        ValueError: On bad inputs, missing contracts/snapshots/prices, or
            a zero denominator IV.
        RuntimeError: If a required API key is unavailable.
    """
    if lookahead_days <= 0:
        raise ValueError("lookahead_days must be a positive integer")
    if call_delta <= 0:
        raise ValueError("call_delta must be positive (calls have delta in (0, 1]).")
    if put_delta >= 0:
        raise ValueError("put_delta must be negative (puts have delta in [-1, 0)).")

    client = get_client(api_key)
    resolved_mode = _resolve_mode(mode, quote_date)
    valuation = datetime.date.today() if resolved_mode == "snapshot" else quote_date
    target = quote_date + datetime.timedelta(days=lookahead_days)

    rate = 0.0
    if resolved_mode == "historical":
        rate = (
            risk_free_rate
            if risk_free_rate is not None
            else get_risk_free_rate(quote_date, api_key=fred_api_key)
        )

    expirations = _resolve_expirations(
        client,
        underlying_ticker,
        target,
        as_of=None if resolved_mode == "snapshot" else quote_date,
        expired=False if resolved_mode == "snapshot" else None,
        interpolate=interpolate,
    )
    logger.info(
        "Vol-skew %s (%s): target %s, evaluating expirations %s",
        underlying_ticker,
        resolved_mode,
        target,
        [e.isoformat() for e in expirations],
    )

    denom_type = _denominator_type(denominator_delta)
    per_expiration: list[
        tuple[datetime.date, float, tuple[OptionLeg, OptionLeg, OptionLeg]]
    ] = []
    underlying_price: float | None = None
    for expiration in expirations:
        if resolved_mode == "snapshot":
            quotes, price = _snapshot_surface(
                client,
                underlying_ticker,
                expiration,
                min_open_interest=min_open_interest,
                max_relative_spread=max_relative_spread,
            )
        else:
            quotes, price = _historical_surface(
                client,
                underlying_ticker,
                expiration,
                quote_date,
                rate=rate,
                dividend_yield=dividend_yield,
                strike_band=strike_band,
                min_volume=min_volume,
            )
        if price is not None:
            underlying_price = price
        if not quotes:
            continue
        legs = (
            _select_leg(quotes, "call", call_delta, interpolate=interpolate),
            _select_leg(quotes, "put", put_delta, interpolate=interpolate),
            _select_leg(quotes, denom_type, denominator_delta, interpolate=interpolate),
        )
        per_expiration.append((expiration, _year_fraction(valuation, expiration), legs))

    if not per_expiration:
        raise ValueError(
            f"No usable option data for {underlying_ticker!r} near {target} "
            f"({resolved_mode} mode)."
        )

    if len(per_expiration) == 1:
        expiration, _t, (call_leg, put_leg, denom_leg) = per_expiration[0]
        used_expiration = expiration
        interpolated = interpolate
    else:
        (_, t_lo, legs_lo), (_, t_hi, legs_hi) = (
            per_expiration[0],
            per_expiration[1],
        )
        t_target = _year_fraction(valuation, target)

        def _termed(lo: OptionLeg, hi: OptionLeg, ctype: str, tgt: float) -> OptionLeg:
            iv = _term_interp_iv(
                lo.implied_volatility, t_lo, hi.implied_volatility, t_hi, t_target
            )
            return OptionLeg(
                ticker=None,
                contract_type=ctype,
                target_delta=tgt,
                delta=tgt,
                implied_volatility=iv,
            )

        call_leg = _termed(legs_lo[0], legs_hi[0], "call", call_delta)
        put_leg = _termed(legs_lo[1], legs_hi[1], "put", put_delta)
        denom_leg = _termed(legs_lo[2], legs_hi[2], denom_type, denominator_delta)
        used_expiration = target
        interpolated = True

    if denom_leg.implied_volatility == 0.0:
        raise ValueError("Denominator option has zero implied volatility.")

    skew = (
        call_leg.implied_volatility - put_leg.implied_volatility
    ) / denom_leg.implied_volatility

    result = VolatilitySkewResult(
        underlying_ticker=underlying_ticker,
        quote_date=quote_date,
        target_expiration=target,
        expiration=used_expiration,
        days_to_expiration=(used_expiration - valuation).days,
        underlying_price=underlying_price,
        call=call_leg,
        put=put_leg,
        denominator=denom_leg,
        skew=skew,
        mode=resolved_mode,
        interpolated=interpolated,
    )
    logger.info(
        "Vol-skew %s: call IV %.4f, put IV %.4f, denom IV %.4f -> skew %.4f",
        underlying_ticker,
        call_leg.implied_volatility,
        put_leg.implied_volatility,
        denom_leg.implied_volatility,
        skew,
    )
    return result


def compute_normalized_volatility_skew(
    underlying_ticker: str,
    quote_date: datetime.date,
    lookahead_days: int,
    put_delta: float,
    call_delta: float,
    denominator_delta: float,
    *,
    mode: Mode = "auto",
    interpolate: bool = True,
    min_open_interest: int = 0,
    max_relative_spread: float | None = None,
    risk_free_rate: float | None = None,
    dividend_yield: float = 0.0,
    strike_band: float = 0.5,
    min_volume: float = 1.0,
    api_key: str | None = None,
    fred_api_key: str | None = None,
) -> float:
    """Return only the normalized volatility skew as a float.

    Thin wrapper over :func:`compute_normalized_volatility_skew_result`;
    see it for full argument documentation.
    """
    return compute_normalized_volatility_skew_result(
        underlying_ticker,
        quote_date,
        lookahead_days,
        put_delta,
        call_delta,
        denominator_delta,
        mode=mode,
        interpolate=interpolate,
        min_open_interest=min_open_interest,
        max_relative_spread=max_relative_spread,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        strike_band=strike_band,
        min_volume=min_volume,
        api_key=api_key,
        fred_api_key=fred_api_key,
    ).skew


def _result_row(result: VolatilitySkewResult) -> dict[str, object]:
    return {
        "Ticker": result.underlying_ticker,
        "Quote Date": result.quote_date.isoformat(),
        "Expiration": result.expiration.isoformat(),
        "DTE": result.days_to_expiration,
        "Underlying": result.underlying_price,
        "Call Delta": result.call.delta,
        "Call IV": result.call.implied_volatility,
        "Put Delta": result.put.delta,
        "Put IV": result.put.implied_volatility,
        "Denom IV": result.denominator.implied_volatility,
        "Skew": result.skew,
        "Mode": result.mode,
        "Interpolated": result.interpolated,
    }


_FRAME_SCHEMA: dict[str, pl.DataType] = {
    "Ticker": pl.Utf8(),
    "Quote Date": pl.Utf8(),
    "Expiration": pl.Utf8(),
    "DTE": pl.Int64(),
    "Underlying": pl.Float64(),
    "Call Delta": pl.Float64(),
    "Call IV": pl.Float64(),
    "Put Delta": pl.Float64(),
    "Put IV": pl.Float64(),
    "Denom IV": pl.Float64(),
    "Skew": pl.Float64(),
    "Mode": pl.Utf8(),
    "Interpolated": pl.Boolean(),
}


def results_to_frame(results: Sequence[VolatilitySkewResult]) -> pl.DataFrame:
    """Collect skew results into a Polars DataFrame (one row each)."""
    return pl.DataFrame([_result_row(r) for r in results], schema=_FRAME_SCHEMA)


def compute_volatility_skew_frame(
    underlying_tickers: Sequence[str],
    quote_date: datetime.date,
    lookahead_days: int,
    put_delta: float,
    call_delta: float,
    denominator_delta: float,
    *,
    mode: Mode = "auto",
    interpolate: bool = True,
    min_open_interest: int = 0,
    max_relative_spread: float | None = None,
    risk_free_rate: float | None = None,
    dividend_yield: float = 0.0,
    strike_band: float = 0.5,
    min_volume: float = 1.0,
    api_key: str | None = None,
    fred_api_key: str | None = None,
) -> pl.DataFrame:
    """Compute the skew for many underlyings, one row per success.

    Tickers that error (no contracts, no data, etc.) are logged and
    skipped rather than aborting the batch.  See
    :func:`compute_normalized_volatility_skew_result` for the arguments.
    """
    results: list[VolatilitySkewResult] = []
    for ticker in underlying_tickers:
        try:
            results.append(
                compute_normalized_volatility_skew_result(
                    ticker,
                    quote_date,
                    lookahead_days,
                    put_delta,
                    call_delta,
                    denominator_delta,
                    mode=mode,
                    interpolate=interpolate,
                    min_open_interest=min_open_interest,
                    max_relative_spread=max_relative_spread,
                    risk_free_rate=risk_free_rate,
                    dividend_yield=dividend_yield,
                    strike_band=strike_band,
                    min_volume=min_volume,
                    api_key=api_key,
                    fred_api_key=fred_api_key,
                )
            )
        except ValueError, RuntimeError:
            logger.warning(
                "Skipping %s: skew computation failed", ticker, exc_info=True
            )
    logger.info(
        "Computed skew for %d/%d underlyings",
        len(results),
        len(underlying_tickers),
    )
    return results_to_frame(results)


def volatility_skew_table(frame: pl.DataFrame) -> GT:
    """Render a skew DataFrame (from :func:`compute_volatility_skew_frame`)."""
    return (
        GT(frame)
        .tab_header(
            title="Normalized Volatility Skew",
            subtitle=f"{frame.height} underlying(s)",
        )
        .fmt_currency(columns=["Underlying"], decimals=2)
        .fmt_percent(columns=["Call IV", "Put IV", "Denom IV"], decimals=2)
        .fmt_number(columns=["Call Delta", "Put Delta", "Skew"], decimals=4)
        .sub_missing(missing_text="—")
        .cols_align(align="center", columns=["Mode", "Interpolated", "DTE"])
    )
