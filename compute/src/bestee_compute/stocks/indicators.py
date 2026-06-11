"""TA-Lib-backed technical-indicator primitives.

Thin wrappers around `TA-Lib <https://ta-lib.org>`_ that adapt its output
to the shape the decorator pipeline expects: full-length ``list``\\s with
``None`` over each indicator's warmup, a :class:`ValueError` for an
out-of-range period, and ``[]`` for empty input.  The actual numerics
(Wilder smoothing, rolling extrema, moving averages) are delegated to
TA-Lib rather than hand-rolled here.

These are consumed by the indicator decorators in
:mod:`bestee_compute.stocks.decorators` (``StochasticOscillatorDecorator``,
``RelativeStrengthIndexDecorator``, ``SimpleMovingAverageDecorator``) but
are kept dependency-free of the pipeline so they can be used standalone.
"""

from collections.abc import Sequence

import numpy as np
import talib

# ── Stochastic oscillator ────────────────────────────────────────────


def stochastic_oscillator(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    k_period: int,
    d_period: int,
) -> tuple[list[float | None], list[float | None]]:
    """Compute the classic stochastic-oscillator %K and %D series via TA-Lib.

    Delegates the maths to TA-Lib: :func:`talib.STOCHF` for the fast %K and
    :func:`talib.SMA` for the %D smoothing.  STOCHF is asked for an
    *unsmoothed* %K (``fastd_period=1``) so %K keeps its natural
    ``k_period - 1`` warmup; folding %D's lookback in (STOCHF's default)
    would otherwise push %K's first value out to ``k_period + d_period - 2``.
    %D is then a ``d_period`` SMA of that %K, so its first value lands at
    ``k_period + d_period - 2`` exactly as before.

    TA-Lib emits ``NaN`` over each series' warmup (surfaced here as ``None``
    to keep the full bar grid) and ``0.0`` for a flat, zero-range window.
    """
    if k_period < 1 or d_period < 1:
        msg = (
            f"stochastic_oscillator periods must be >= 1; "
            f"got k_period={k_period}, d_period={d_period}"
        )
        raise ValueError(msg)
    n = min(len(highs), len(lows), len(closes))
    if n == 0:
        return [], []
    highs_arr = np.asarray(highs[:n], dtype=np.float64)
    lows_arr = np.asarray(lows[:n], dtype=np.float64)
    closes_arr = np.asarray(closes[:n], dtype=np.float64)
    # fastd_period=1 leaves %K unsmoothed so it keeps its own k_period-1
    # lookback; the SMA below applies the d_period smoothing for %D.
    fast_k, _ = talib.STOCHF(
        highs_arr,
        lows_arr,
        closes_arr,
        fastk_period=k_period,
        fastd_period=1,
    )
    k_values: list[float | None] = [None if np.isnan(v) else float(v) for v in fast_k]
    if d_period == 1:
        # SMA period 1 is the identity (TA-Lib rejects timeperiod < 2).
        d_values: list[float | None] = list(k_values)
    else:
        d_raw = talib.SMA(fast_k, timeperiod=d_period)
        d_values = [None if np.isnan(v) else float(v) for v in d_raw]
    return k_values, d_values


# ── Relative Strength Index (Wilder's smoothing) ─────────────────────


def relative_strength_index(
    closes: Sequence[float],
    period: int,
) -> list[float | None]:
    """Compute Wilder's Relative Strength Index series via TA-Lib.

    Delegates to :func:`talib.RSI`, which uses Wilder's smoothing.  TA-Lib
    emits ``NaN`` for the ``period``-bar warmup (surfaced here as ``None``)
    and ``0.0`` for a wholly flat series (no price movement at all).  A
    series shorter than the warmup yields all ``None``.
    """
    if period < 1:
        msg = f"RSI period must be >= 1; got {period}"
        raise ValueError(msg)
    n = len(closes)
    if n == 0:
        return []
    arr = np.asarray(closes, dtype=np.float64)
    rsi = talib.RSI(arr, timeperiod=period)
    return [None if np.isnan(v) else float(v) for v in rsi]


# ── Simple Moving Average ────────────────────────────────────────────


def simple_moving_average(
    values: Sequence[float],
    period: int,
) -> list[float | None]:
    """Compute the simple moving average series via TA-Lib.

    Delegates to :func:`talib.SMA`.  ``period`` 1 is the identity, which
    TA-Lib won't compute (it requires ``timeperiod >= 2``), so it's handled
    directly.  TA-Lib emits ``NaN`` over the ``period - 1`` warmup, surfaced
    here as ``None``.

    TA-Lib skips a *leading* run of ``NaN``s, so stacking SMA on another
    indicator's output (whose warmup is encoded as ``NaN``) still yields a
    usable scalar in the tail -- e.g. ``SMA smoothed rsi14 5`` on top of an
    RSI series.  A ``NaN`` in the *interior* of the input propagates to the
    end of the output, per TA-Lib's convention.
    """
    if period < 1:
        msg = f"SMA period must be >= 1; got {period}"
        raise ValueError(msg)
    n = len(values)
    if n == 0:
        return []
    arr = np.asarray(values, dtype=np.float64)
    # SMA over a 1-bar window is the identity; TA-Lib rejects timeperiod < 2.
    sma = arr if period == 1 else talib.SMA(arr, timeperiod=period)
    return [None if np.isnan(v) else float(v) for v in sma]


def rsquared_trend(series: Sequence[float]) -> float | None:
    """R² of a least-squares linear fit of *series* against its index.

    Equivalent to the squared Pearson correlation between the bar index
    (0, 1, 2, …) and the value at that bar.  A linear trend scores 1.0;
    a flat or noisy series scores near 0.  Returns ``None`` for series
    too short (< 2 valid points after dropping NaNs) or with zero
    variance.  NaN-padded warmup from upstream indicators is dropped so
    R² can be applied to an RSI or SMA series cleanly.
    """
    arr = np.asarray(series, dtype=np.float64)
    valid = ~np.isnan(arr)
    if int(valid.sum()) < 2:
        return None
    xs = np.arange(arr.size, dtype=np.float64)[valid]
    ys = arr[valid]
    if float(ys.var()) == 0.0:
        return None
    r = np.corrcoef(xs, ys)[0, 1]
    if not np.isfinite(r):
        return None
    return float(r * r)
