"""Merton (continuous dividend-yield) Black–Scholes option analytics.

Pure pricing math with no network access and no I/O: every function takes
scalar floats and returns a float (implied volatility returns ``None`` when
a market price cannot be inverted).

The model is Black–Scholes–Merton with a continuous dividend yield ``q``::

    d1 = (ln(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    call = S * exp(-q * T) * N(d1) - K * exp(-r * T) * N(d2)
    put  = K * exp(-r * T) * N(-d2) - S * exp(-q * T) * N(-d1)

where ``S`` is spot, ``K`` strike, ``T`` time to expiry in years, ``r`` the
risk-free rate, ``sigma`` the annualized volatility, and ``N`` the standard
normal CDF.  Call delta is ``exp(-q * T) * N(d1)`` and put delta is
``exp(-q * T) * (N(d1) - 1)``.

:func:`implied_volatility` inverts the price→sigma map with bisection (the
price is monotone increasing in ``sigma``) and returns ``None`` for prices
that violate the no-arbitrage bounds or otherwise fail to converge.
"""

import logging
import math
from typing import Literal

logger = logging.getLogger(__name__)

OptionType = Literal["call", "put"]

_SQRT2 = math.sqrt(2.0)
_MIN_SIGMA = 1e-6
_MAX_SIGMA = 5.0
_SIGMA_BRACKET_TOL = 1e-9


def _norm_cdf(x: float) -> float:
    """Return the standard normal CDF ``N(x)``.

    Args:
        x: Point at which to evaluate the CDF.

    Returns:
        ``0.5 * (1 + erf(x / sqrt(2)))``, a value in ``[0, 1]``.
    """
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def _validate_inputs(
    spot: float,
    strike: float,
    t_years: float,
    sigma: float,
) -> None:
    """Reject non-positive pricing inputs (programming errors).

    Args:
        spot: Current price of the underlying.
        strike: Option strike price.
        t_years: Time to expiry in years.
        sigma: Annualized volatility.

    Raises:
        ValueError: If any of ``spot``, ``strike``, ``t_years`` or
            ``sigma`` is not strictly positive.
    """
    for name, value in (
        ("spot", spot),
        ("strike", strike),
        ("t_years", t_years),
        ("sigma", sigma),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive, got {value!r}.")


def _d1_d2(
    spot: float,
    strike: float,
    t_years: float,
    rate: float,
    dividend_yield: float,
    sigma: float,
) -> tuple[float, float]:
    """Return the Black–Scholes ``d1`` and ``d2`` terms.

    Args:
        spot: Current price of the underlying.
        strike: Option strike price.
        t_years: Time to expiry in years.
        rate: Continuously compounded risk-free rate.
        dividend_yield: Continuous dividend yield ``q``.
        sigma: Annualized volatility.

    Returns:
        The ``(d1, d2)`` pair.
    """
    vol_sqrt_t = sigma * math.sqrt(t_years)
    d1 = (
        math.log(spot / strike)
        + (rate - dividend_yield + 0.5 * sigma * sigma) * t_years
    ) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def bs_price(
    spot: float,
    strike: float,
    t_years: float,
    rate: float,
    dividend_yield: float,
    sigma: float,
    option_type: OptionType,
) -> float:
    """Price a European option with the Black–Scholes–Merton formula.

    Args:
        spot: Current price of the underlying (> 0).
        strike: Strike price (> 0).
        t_years: Time to expiry in years (> 0).
        rate: Continuously compounded risk-free rate.
        dividend_yield: Continuous dividend yield ``q``.
        sigma: Annualized volatility (> 0).
        option_type: ``"call"`` or ``"put"``.

    Returns:
        The option's theoretical price.

    Raises:
        ValueError: If ``spot``, ``strike``, ``t_years`` or ``sigma`` is
            not strictly positive.
    """
    _validate_inputs(spot, strike, t_years, sigma)
    d1, d2 = _d1_d2(spot, strike, t_years, rate, dividend_yield, sigma)
    discounted_spot = spot * math.exp(-dividend_yield * t_years)
    discounted_strike = strike * math.exp(-rate * t_years)
    if option_type == "call":
        return discounted_spot * _norm_cdf(d1) - discounted_strike * _norm_cdf(d2)
    return discounted_strike * _norm_cdf(-d2) - discounted_spot * _norm_cdf(-d1)


def bs_delta(
    spot: float,
    strike: float,
    t_years: float,
    rate: float,
    dividend_yield: float,
    sigma: float,
    option_type: OptionType,
) -> float:
    """Compute the Black–Scholes–Merton delta of a European option.

    Args:
        spot: Current price of the underlying (> 0).
        strike: Strike price (> 0).
        t_years: Time to expiry in years (> 0).
        rate: Continuously compounded risk-free rate.
        dividend_yield: Continuous dividend yield ``q``.
        sigma: Annualized volatility (> 0).
        option_type: ``"call"`` or ``"put"``.

    Returns:
        The option delta: call delta in ``(0, 1)`` and put delta in
        ``(-1, 0)`` (both scaled by ``exp(-q * t_years)``).

    Raises:
        ValueError: If ``spot``, ``strike``, ``t_years`` or ``sigma`` is
            not strictly positive.
    """
    _validate_inputs(spot, strike, t_years, sigma)
    d1, _ = _d1_d2(spot, strike, t_years, rate, dividend_yield, sigma)
    discount = math.exp(-dividend_yield * t_years)
    if option_type == "call":
        return discount * _norm_cdf(d1)
    return discount * (_norm_cdf(d1) - 1.0)


def implied_volatility(
    price: float,
    spot: float,
    strike: float,
    t_years: float,
    rate: float,
    dividend_yield: float,
    option_type: OptionType,
    *,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float | None:
    """Recover the volatility implied by an observed option price.

    Inverts ``sigma -> bs_price`` with bisection over ``sigma`` in roughly
    ``[1e-6, 5.0]``.  The price is monotone increasing in ``sigma``, so the
    bracket is reliable.  Unlike :func:`bs_price`/:func:`bs_delta`, this
    function never raises on bad market data — it returns ``None`` instead.

    Args:
        price: Observed option price to invert.
        spot: Current price of the underlying.
        strike: Strike price.
        t_years: Time to expiry in years.
        rate: Continuously compounded risk-free rate.
        dividend_yield: Continuous dividend yield ``q``.
        option_type: ``"call"`` or ``"put"``.
        tol: Absolute price tolerance for convergence.
        max_iter: Maximum number of bisection iterations.

    Returns:
        The implied volatility, or ``None`` when the inputs are degenerate
        (non-positive ``price``, ``spot``, ``strike`` or ``t_years``), the
        price violates the model's no-arbitrage bounds for the bracket, or
        the search fails to converge within ``max_iter`` iterations.
    """
    if price <= 0.0 or spot <= 0.0 or strike <= 0.0 or t_years <= 0.0:
        logger.debug(
            "implied_volatility: degenerate inputs "
            "(price=%r spot=%r strike=%r t_years=%r)",
            price,
            spot,
            strike,
            t_years,
        )
        return None

    def price_at(sigma: float) -> float:
        return bs_price(spot, strike, t_years, rate, dividend_yield, sigma, option_type)

    low, high = _MIN_SIGMA, _MAX_SIGMA
    price_low = price_at(low)
    price_high = price_at(high)
    if price < price_low - tol or price > price_high + tol:
        logger.debug(
            "implied_volatility: price %r outside invertible range [%r, %r]",
            price,
            price_low,
            price_high,
        )
        return None

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        diff = price_at(mid) - price
        if abs(diff) <= tol or (high - low) <= _SIGMA_BRACKET_TOL:
            return mid
        if diff > 0.0:
            high = mid
        else:
            low = mid

    logger.debug(
        "implied_volatility: no convergence after %d iterations (price=%r)",
        max_iter,
        price,
    )
    return None
