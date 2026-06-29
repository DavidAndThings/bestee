"""Security-level analytics on the Massive API.

Submodules:

* :mod:`bestee_compute.securities.options` — options analytics: the
  delta-based normalized implied-volatility skew (snapshot or historical,
  with smile/term interpolation), plus batch and display helpers.
* :mod:`bestee_compute.securities.black_scholes` — Black–Scholes–Merton
  pricing, delta, and implied-volatility inversion.
* :mod:`bestee_compute.securities.rates` — point-in-time risk-free rate
  from FRED.
"""

from bestee_compute.securities import black_scholes, options, rates
from bestee_compute.securities.options import (
    OptionLeg,
    VolatilitySkewResult,
    compute_normalized_volatility_skew,
    compute_normalized_volatility_skew_result,
    compute_volatility_skew_frame,
    results_to_frame,
    volatility_skew_table,
)
from bestee_compute.securities.rates import get_risk_free_rate

__all__ = [
    "OptionLeg",
    "VolatilitySkewResult",
    "black_scholes",
    "compute_normalized_volatility_skew",
    "compute_normalized_volatility_skew_result",
    "compute_volatility_skew_frame",
    "get_risk_free_rate",
    "options",
    "rates",
    "results_to_frame",
    "volatility_skew_table",
]
