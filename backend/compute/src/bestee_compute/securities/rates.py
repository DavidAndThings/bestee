"""Point-in-time risk-free rate from FRED.

Fetches a Treasury constant-maturity yield series from FRED (default
``DGS3MO`` — the 3-Month Treasury, reported in percent) and returns the
most recent observation on or before a given as-of date as a decimal
fraction (e.g. a FRED value of ``5.25`` becomes ``0.0525``).

The point-in-time value is resolved server-side: the request pins
``observation_end`` to the as-of date and sorts observations descending,
so the first non-missing value FRED returns is the latest valid yield
that would have been known on that date.  Missing readings (which FRED
encodes as the string ``"."``) are skipped, which carries the last valid
value backward in time.
"""

import datetime
import logging
import os

import httpx
from dotenv import find_dotenv, load_dotenv

logger = logging.getLogger(__name__)

FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"

# FRED encodes a missing observation as this sentinel string.
_FRED_MISSING_VALUE = "."


def _resolve_api_key(api_key: str | None) -> str:
    """Return *api_key* or fall back to the ``FRED_API_KEY`` env var.

    Loads ``.env`` and ``.env.local`` (the latter is where this project
    keeps ``FRED_API_KEY``) before reading the environment, mirroring
    :mod:`bestee_compute.macro.liquidity`.

    Args:
        api_key: An explicit API key, or *None* to read the environment.

    Returns:
        The resolved FRED API key.

    Raises:
        RuntimeError: If no key is provided and ``FRED_API_KEY`` is unset.
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


def get_risk_free_rate(
    as_of: datetime.date,
    *,
    series_id: str = "DGS3MO",
    api_key: str | None = None,
    timeout: float = 30.0,
) -> float:
    """Return the point-in-time risk-free rate as a decimal fraction.

    Fetches the FRED series *series_id* (default ``DGS3MO``, the 3-Month
    Treasury Constant Maturity yield reported in percent) and returns the
    most recent observation dated on or before *as_of*, converted from
    percent to a decimal fraction (e.g. ``5.25`` -> ``0.0525``).

    Point-in-time selection is delegated to FRED: the request pins
    ``observation_end`` to *as_of* and requests descending order, so the
    first non-missing value is the latest valid yield on or before *as_of*.
    Missing readings (FRED's ``"."``) are skipped, carrying the last valid
    value backward in time.

    Args:
        as_of: Date to value the rate as of. The returned rate is the
            latest valid observation on or before this date.
        series_id: FRED series identifier. Defaults to ``"DGS3MO"``.
        api_key: FRED API key. Falls back to the ``FRED_API_KEY``
            environment variable when *None*.
        timeout: Per-request timeout in seconds.

    Returns:
        The risk-free rate as a decimal fraction (e.g. ``0.0525``).

    Raises:
        RuntimeError: If no API key is available.
        httpx.HTTPError: If the FRED request fails.
        ValueError: If FRED returns no usable observation on or before
            *as_of*.
    """
    key = _resolve_api_key(api_key)
    params = {
        "series_id": series_id,
        "api_key": key,
        "file_type": "json",
        "observation_end": as_of.isoformat(),
        "sort_order": "desc",
    }

    logger.info("Fetching FRED series %s as of %s", series_id, as_of.isoformat())
    with httpx.Client(timeout=timeout) as client:
        response = client.get(FRED_OBSERVATIONS_URL, params=params)
        response.raise_for_status()
        observations = response.json().get("observations", [])

    # The list is sorted newest-first and bounded at *as_of*, so the first
    # genuine value is the latest valid yield on or before the as-of date.
    for observation in observations:
        raw_value = observation.get("value")
        if not raw_value or raw_value == _FRED_MISSING_VALUE:
            continue
        try:
            percent = float(raw_value)
        except ValueError:
            continue
        rate = percent / 100.0
        observation_date = observation.get("date")
        logger.info(
            "Resolved %s risk-free rate %.6f (%.4f%%) from observation %s",
            series_id,
            rate,
            percent,
            observation_date,
        )
        return rate

    msg = (
        f"FRED returned no usable observation for {series_id} on or before "
        f"{as_of.isoformat()}"
    )
    raise ValueError(msg)
