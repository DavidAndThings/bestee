"""Fama-French 6-factor regression variables.

Assembles the left- and right-hand-side variables for a Fama-French 6-factor
model (the 2018 five-factor model augmented with momentum):

    R_i - R_f = a + b1*(Mkt-RF) + b2*SMB + b3*HML
                  + b4*RMW + b5*CMA + b6*Mom + e

The six factors and the risk-free rate are the canonical series from Kenneth
French's data library (daily, value-weighted, in percent), which is the standard
source -- reconstructing them requires the full CRSP/Compustat universe with
portfolio sorts. Each ticker's excess return is built from its own (adjusted)
close. Returns are daily simple returns to match the factor convention.

Note: Massive's adjusted close is split-adjusted, not dividend-adjusted, so the
per-ticker return is a close-to-close (price) return; the factors are total
returns. For dividend payers this understates total return slightly.
"""

import datetime as dt
import io
import logging
import math
import re
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import httpx
import numpy as np
import polars as pl
from pydantic import BaseModel

from bestee_compute.workflow.tools import (
    AnalysisConfig,
    OHLCHeader,
    get_ohlc_column_for_tickers,
    get_tickers,
)

logger = logging.getLogger(__name__)

# Kenneth French data library (daily). The 5-factor file also carries RF.
_KEN_FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FF5_DAILY_URL = f"{_KEN_FRENCH_BASE}F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
MOMENTUM_DAILY_URL = f"{_KEN_FRENCH_BASE}F-F_Momentum_Factor_daily_CSV.zip"

# Column names in the Ken French files (in order, after the date column).
_FF5_FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
_MOMENTUM_FACTOR = "Mom"
# The six right-hand-side regressors of the FF6 model.
FAMA_FRENCH_6_FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
# ``factors_to_use`` -> that model's regressors: FF3 (1993), the Carhart
# 4-factor (1997, + momentum), FF5 (2015, + profitability/investment), and
# FF6 (2018, all six).
_FACTOR_MODELS: dict[int, list[str]] = {
    3: ["Mkt-RF", "SMB", "HML"],
    4: ["Mkt-RF", "SMB", "HML", "Mom"],
    5: ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
    6: ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"],
}
_RISK_FREE_COLUMN = "RF"
_DATE_COLUMN = "date"
_RETURN_COLUMN_NAME = "Return"
_EXCESS_RETURN_COLUMN_NAME = "ExcessReturn"

# A data row is ``YYYYMMDD,v1,v2,...``; the preamble, header, blank trailer and
# copyright lines never start with an 8-digit date, so this selects data only.
_DATA_ROW = re.compile(r"^\s*(\d{8})\s*,(.*)$")
# Ken French encodes missing observations as -99.99 or -999 (in percent).
_MISSING_THRESHOLD = -99.0
_USER_AGENT = "bestee-compute/1.0"
_TICKER_COLUMN = "Ticker"
_SPECIFICATION_COLUMN = "Specification"
_RESIDUAL_COLUMN = "Residual"
_ZSCORE_COLUMN = "ZScore"
_PVALUE_COLUMN = "PValue"
_OOS_DATE_COLUMN = "Date"


class FamaFrenchSpecification(BaseModel):
    """The choices that define a Fama-French regression: the estimation window
    ``[start_date, end_date]`` (inclusive ISO ``YYYY-MM-DD``) and which model to
    use via ``factors_to_use`` -- 3 (FF3), 4 (Carhart), 5 (FF5) or 6 (FF6).
    """

    start_date: str
    end_date: str
    factors_to_use: int

    @property
    def name(self) -> str:
        """Compact label identifying the model and its estimation window."""
        return f"FF{self.factors_to_use}_{self.start_date}_{self.end_date}"

    def verify_out_of_sample(self, date: dt.date) -> None:
        """Raise unless *date* is strictly after this model's window end.

        Residuals are only meaningful out of sample, so the evaluation date must
        come after ``end_date`` -- the last day used to fit the model.

        Raises:
            ValueError: If *date* is on or before ``end_date``.
        """
        end = dt.date.fromisoformat(self.end_date[:10])
        if date <= end:
            raise ValueError(
                f"date {date} must be after the model window end {end} "
                "(residuals are out-of-sample)"
            )


class FamaFrenchResult(BaseModel):
    """One ticker's fitted Fama-French factor regression.

    Coefficients are daily (the model is fit on daily excess returns); annualize
    ``alpha`` by ~252. ``t_stats`` and ``std_errors`` are classical
    (homoskedastic) OLS values keyed by ``"alpha"`` and each factor name -- for
    daily returns, robust (Newey-West) errors would be more conservative.
    """

    ticker: str
    factors: list[str]
    n_observations: int
    start_date: dt.date
    end_date: dt.date
    alpha: float
    betas: dict[str, float]
    t_stats: dict[str, float]
    std_errors: dict[str, float]
    r_squared: float
    adj_r_squared: float
    residual_std: float

    def summary(self) -> pl.DataFrame:
        """Per-parameter coefficient, standard error and t-stat as a frame."""
        names = ["alpha", *self.factors]
        coefficients = [self.alpha, *self.betas.values()]
        return pl.DataFrame(
            {
                "Parameter": names,
                "Coefficient": coefficients,
                "StdError": [self.std_errors[name] for name in names],
                "tStat": [self.t_stats[name] for name in names],
            }
        )


def get_variables(config: AnalysisConfig) -> Mapping[str, pl.DataFrame]:
    """Regression-ready Fama-French 6-factor variables, one frame per ticker.

    Returns a ``ticker -> DataFrame`` mapping. Each frame has a ``Timestamp``
    column, the ticker's daily excess return ``ExcessReturn`` (the dependent
    variable, ``R_i - R_f``), and the six shared factor columns ``Mkt-RF``,
    ``SMB``, ``HML``, ``RMW``, ``CMA`` and ``Mom`` (the regressors). All
    values are daily decimals, time-aligned on the trading day, with each
    ticker's warm-up/missing rows dropped independently -- so the model is
    ``ExcessReturn ~ Mkt-RF + SMB + HML + RMW + CMA + Mom`` per frame.

    The factors are fetched once and shared across the per-ticker frames.
    """
    tickers = get_tickers(config)
    # Fetch null-preserving (a full outer union, not the cross-ticker inner
    # join): each ticker is regressed on the shared factors independently, so a
    # sparse name must not drop another's trading days. Sort before pct_change,
    # which is order-dependent and the outer union isn't guaranteed sorted.
    returns = (
        get_ohlc_column_for_tickers(config, drop_missing=False)
        .sort(OHLCHeader.TIMESTAMP)
        .select(
            OHLCHeader.TIMESTAMP,
            pl.col(OHLCHeader.TIMESTAMP).dt.date().alias(_DATE_COLUMN),
            *[
                pl.col(f"{ticker}_{config.ohlc_column}")
                .pct_change()
                .alias(f"{ticker}_{_RETURN_COLUMN_NAME}")
                for ticker in tickers
            ],
        )
    )
    merged = returns.join(fetch_fama_french_factors(), on=_DATE_COLUMN, how="inner")
    return {
        ticker: merged.select(
            OHLCHeader.TIMESTAMP,
            # Excess return R_i - R_f: net out the risk-free rate.
            (
                pl.col(f"{ticker}_{_RETURN_COLUMN_NAME}") - pl.col(_RISK_FREE_COLUMN)
            ).alias(_EXCESS_RETURN_COLUMN_NAME),
            *FAMA_FRENCH_6_FACTORS,
        ).drop_nulls()
        for ticker in tickers
    }


def fit_model(
    config: AnalysisConfig,
    specification: FamaFrenchSpecification,
) -> Mapping[str, FamaFrenchResult]:
    """Fit a Fama-French factor regression per ticker (delegates to `_fit`).

    Regresses each ticker's excess return on ``specification``'s factor set
    over its ``[start_date, end_date]`` window; returns a
    ``ticker -> FamaFrenchResult`` mapping (alpha, betas, t-stats, R^2, ...).
    Tickers with too few in-window observations are skipped (logged).

    Raises:
        ValueError: If ``factors_to_use`` is not 3/4/5/6 or the window is
            inverted.
    """
    return _fit(get_variables(config), specification)


def _fit(
    variables: Mapping[str, pl.DataFrame],
    specification: FamaFrenchSpecification,
) -> dict[str, FamaFrenchResult]:
    """Per-ticker OLS over the spec's window on pre-built *variables*."""
    if specification.factors_to_use not in _FACTOR_MODELS:
        raise ValueError(
            f"factors_to_use must be one of {sorted(_FACTOR_MODELS)}; "
            f"got {specification.factors_to_use}"
        )
    factors = _FACTOR_MODELS[specification.factors_to_use]
    start = dt.date.fromisoformat(specification.start_date[:10])
    end = dt.date.fromisoformat(specification.end_date[:10])
    if end < start:
        raise ValueError(f"end_date {end} precedes start_date {start}")

    results: dict[str, FamaFrenchResult] = {}
    for ticker, frame in variables.items():
        window = frame.filter(
            pl.col(OHLCHeader.TIMESTAMP).dt.date().is_between(start, end)
        )
        # Need more rows than parameters (intercept + factors) for a fit.
        if window.height <= len(factors) + 1:
            logger.warning(
                "Skipping %s: %d obs in [%s, %s] is too few for a %d-factor fit",
                ticker,
                window.height,
                start,
                end,
                specification.factors_to_use,
            )
            continue
        response = window[_EXCESS_RETURN_COLUMN_NAME].to_numpy()
        factor_matrix = window.select(factors).to_numpy()
        try:
            fit = _ordinary_least_squares(response, factor_matrix)
        except np.linalg.LinAlgError as error:
            logger.warning("Skipping %s: OLS failed (%s)", ticker, error)
            continue
        parameters = ["alpha", *factors]
        window_dates = window[OHLCHeader.TIMESTAMP].dt.date()
        results[ticker] = FamaFrenchResult(
            ticker=ticker,
            factors=factors,
            n_observations=window.height,
            start_date=cast(dt.date, window_dates.min()),
            end_date=cast(dt.date, window_dates.max()),
            alpha=float(fit.coef[0]),
            betas={
                factor: float(beta)
                for factor, beta in zip(factors, fit.coef[1:], strict=True)
            },
            t_stats={
                name: float(stat)
                for name, stat in zip(parameters, fit.t_stats, strict=True)
            },
            std_errors={
                name: float(error)
                for name, error in zip(parameters, fit.std_errors, strict=True)
            },
            r_squared=fit.r_squared,
            adj_r_squared=fit.adj_r_squared,
            residual_std=fit.residual_std,
        )
    return results


def get_residuals(
    config: AnalysisConfig,
    dates: str | Sequence[str],
    specifications: Sequence[FamaFrenchSpecification],
    *,
    variables: Mapping[str, pl.DataFrame] | None = None,
) -> pl.DataFrame:
    """Out-of-sample Fama-French residuals across one or more dates.

    For each :class:`FamaFrenchSpecification` and each date in *dates*, fits
    every ticker's model on that spec's estimation window and evaluates the
    part of the realized excess return the model leaves unexplained::

        residual_i = (R_i - R_f) - (alpha_i + sum_k beta_{i,k} * factor_k)

    Every date must be strictly after *every* spec's ``end_date`` -- each spec
    enforces this via :meth:`FamaFrenchSpecification.verify_out_of_sample` --
    and a trading day for which factor data exists (Ken French lags ~1-2
    months).  Dates with no observations for any ticker are skipped with a
    warning; a :exc:`ValueError` is raised only when *no* date yields records.

    Pass pre-built *variables* (from :func:`get_variables`) to skip the
    network fetch; *config* is only used when *variables* is ``None``.

    Returns:
        A long
        ``[Date, Ticker, Specification, Residual, ZScore, PValue]`` frame --
        one row per (date, ticker, specification) that has both a fitted model
        and an observation on that date.  ``ZScore`` standardizes the residual
        by the spec's own fitted idiosyncratic volatility and ``PValue`` is its
        two-sided tail probability; both are **local to each specification**
        and must not be compared across specifications.

    Raises:
        ValueError: If any date is not after some spec's window, or no
            (date, ticker, specification) triple has an observation.
    """
    date_list: list[str] = [dates] if isinstance(dates, str) else list(dates)
    targets = [dt.date.fromisoformat(d[:10]) for d in date_list]
    for specification in specifications:
        for target in targets:
            specification.verify_out_of_sample(target)
    if variables is None:
        variables = get_variables(config)
    # Latest day with both a price and factor data. Ken French factors lag
    # ~1-2 months, so out-of-sample dates past this can't be evaluated -- used
    # to turn the otherwise-cryptic "no observation" failure into actionable
    # guidance.
    covered = [
        cast(dt.date, last)
        for frame in variables.values()
        if frame.height
        and (last := frame[OHLCHeader.TIMESTAMP].dt.date().max()) is not None
    ]
    latest_covered = max(covered) if covered else None
    frames: list[pl.DataFrame] = []
    for date_str, target in zip(date_list, targets):
        records = [
            record
            for specification in specifications
            for record in _residual_records(variables, target, specification)
        ]
        if not records:
            if latest_covered is not None and target > latest_covered:
                logger.warning(
                    "No factor data on %s yet (available through %s; Ken French "
                    "factors lag ~1-2 months); skipping.",
                    target,
                    latest_covered,
                )
            else:
                logger.warning("No ticker had an observation on %s; skipping.", target)
            continue
        frames.append(
            pl.DataFrame(
                records,
                schema={
                    _TICKER_COLUMN: pl.Utf8,
                    _SPECIFICATION_COLUMN: pl.Utf8,
                    _RESIDUAL_COLUMN: pl.Float64,
                    _ZSCORE_COLUMN: pl.Float64,
                    _PVALUE_COLUMN: pl.Float64,
                },
                orient="row",
            ).with_columns(pl.lit(date_str).alias(_OOS_DATE_COLUMN))
        )
    if not frames:
        if latest_covered is not None and all(t > latest_covered for t in targets):
            raise ValueError(
                f"No Fama-French factor data for out-of-sample date(s) "
                f"{date_list}: prices and factors are only available through "
                f"{latest_covered} (Ken French factors lag ~1-2 months). "
                f"Choose an out-of-sample date on or before {latest_covered}."
            )
        raise ValueError(f"No ticker had any observation across dates {date_list}.")
    return pl.concat(frames).select(
        _OOS_DATE_COLUMN,
        _TICKER_COLUMN,
        _SPECIFICATION_COLUMN,
        _RESIDUAL_COLUMN,
        _ZSCORE_COLUMN,
        _PVALUE_COLUMN,
    )


def _residual_records(
    variables: Mapping[str, pl.DataFrame],
    target: dt.date,
    specification: FamaFrenchSpecification,
) -> list[tuple[str, str, float, float, float]]:
    """``(ticker, spec name, residual, z-score, p-value)`` rows for one spec.

    The residual is standardized by *this spec's own* fitted idiosyncratic
    volatility (``residual_std``), so the z-score and p-value are local to
    the specification and must not be compared across specifications.
    """
    records: list[tuple[str, str, float, float, float]] = []
    for ticker, model in _fit(variables, specification).items():
        row = variables[ticker].filter(pl.col(OHLCHeader.TIMESTAMP).dt.date() == target)
        if row.height == 0:
            logger.warning("Skipping %s: no observation on %s", ticker, target)
            continue
        predicted = model.alpha + sum(
            model.betas[factor] * float(row[factor][0]) for factor in model.factors
        )
        residual = float(row[_EXCESS_RETURN_COLUMN_NAME][0]) - predicted
        z_score, p_value = _residual_significance(residual, model.residual_std)
        records.append((ticker, specification.name, residual, z_score, p_value))
    return records


@dataclass(frozen=True)
class _OLSFit:
    """Raw OLS output, coefficients ordered ``[alpha, *betas]``."""

    coef: np.ndarray
    std_errors: np.ndarray
    t_stats: np.ndarray
    r_squared: float
    adj_r_squared: float
    residual_std: float


def _ordinary_least_squares(response: np.ndarray, factors: np.ndarray) -> _OLSFit:
    """OLS of *response* on an intercept plus *factors* (classical SEs).

    *factors* is ``(n, k)``; an intercept column is prepended. Returns the
    coefficients ``[alpha, *betas]`` with homoskedastic standard errors and
    t-stats, plus R^2 / adjusted R^2 and the residual standard deviation.
    """
    n = response.shape[0]
    design = np.column_stack([np.ones(n), factors])
    n_params = design.shape[1]
    coef, _, _, _ = np.linalg.lstsq(design, response, rcond=None)
    residuals = response - design @ coef
    dof = n - n_params
    sigma_squared = float(residuals @ residuals) / dof
    covariance = sigma_squared * np.linalg.inv(design.T @ design)
    std_errors = np.sqrt(np.diag(covariance))
    total = float(((response - response.mean()) ** 2).sum())
    explained = (
        1.0 - float(residuals @ residuals) / total if total > 0 else float("nan")
    )
    return _OLSFit(
        coef=coef,
        std_errors=std_errors,
        t_stats=coef / std_errors,
        r_squared=explained,
        adj_r_squared=1.0 - (1.0 - explained) * (n - 1) / dof,
        residual_std=float(np.sqrt(sigma_squared)),
    )


def _residual_significance(residual: float, sigma: float) -> tuple[float, float]:
    """Standardized residual and two-sided p-value under the model's null.

    Studentizes *residual* by ``sigma`` -- the producing model's own fitted
    idiosyncratic volatility -- so both outputs are local to that model. The
    p-value is the two-sided Normal tail of the z-score, i.e. the large-sample
    limit of the Student-t studentized prediction residual (the leverage term
    ~p/n is omitted as negligible for the multi-year daily windows these models
    use; for very short windows a Student-t with ``n - p`` df would be slightly
    more conservative). A non-positive ``sigma`` yields NaNs.
    """
    if sigma <= 0.0:
        return float("nan"), float("nan")
    z_score = residual / sigma
    return z_score, math.erfc(abs(z_score) / math.sqrt(2.0))


def fetch_fama_french_factors(*, timeout: float = 60.0) -> pl.DataFrame:
    """Fetch the daily FF 6-factor set (+ RF) from the Ken French data library.

    Returns a ``date`` column (``pl.Date``) plus ``Mkt-RF``, ``SMB``, ``HML``,
    ``RMW``, ``CMA``, ``RF`` and ``Mom`` as daily decimals (the source's percent
    values divided by 100), inner-joined so every row has all seven series.

    Raises:
        httpx.HTTPError: If either download fails.
    """
    headers = {"User-Agent": _USER_AGENT}
    with httpx.Client(
        timeout=timeout, headers=headers, follow_redirects=True
    ) as client:
        five_factor = _parse_daily_factor_csv(
            _download_ken_french_csv(FF5_DAILY_URL, client), _FF5_FACTORS
        )
        momentum = _parse_daily_factor_csv(
            _download_ken_french_csv(MOMENTUM_DAILY_URL, client), [_MOMENTUM_FACTOR]
        )
    factors = five_factor.join(momentum, on=_DATE_COLUMN, how="inner")
    logger.info("Fetched %d daily Fama-French factor rows", factors.height)
    return factors


def _download_ken_french_csv(url: str, client: httpx.Client) -> str:
    """Download a Ken French ``*_CSV.zip`` and return its single CSV's text."""
    logger.info("Fetching Fama-French factors from %s", url)
    response = client.get(url)
    response.raise_for_status()
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    return archive.read(archive.namelist()[0]).decode("latin-1")


def _parse_daily_factor_csv(text: str, value_columns: list[str]) -> pl.DataFrame:
    """Parse a daily Ken French factor CSV into ``date`` + decimal factor columns.

    The files wrap the data in a prose preamble and a copyright footer; only the
    ``YYYYMMDD,...`` lines are kept. Source values are percentages, so they are
    divided by 100, and the missing-data sentinels (-99.99 / -999) become null.
    """
    dates: list[str] = []
    rows: list[list[str]] = []
    for line in text.splitlines():
        match = _DATA_ROW.match(line)
        if match is None:
            continue
        values = [cell.strip() for cell in match.group(2).split(",")]
        if len(values) != len(value_columns):
            continue
        dates.append(match.group(1))
        rows.append(values)

    frame = pl.DataFrame(
        {
            _DATE_COLUMN: dates,
            **{col: [row[i] for row in rows] for i, col in enumerate(value_columns)},
        },
        schema={_DATE_COLUMN: pl.Utf8, **{col: pl.Utf8 for col in value_columns}},
    )
    return frame.with_columns(
        pl.col(_DATE_COLUMN).str.to_date("%Y%m%d"),
        *[
            pl.when(pl.col(col).cast(pl.Float64, strict=False) <= _MISSING_THRESHOLD)
            .then(None)
            .otherwise(pl.col(col).cast(pl.Float64, strict=False) / 100.0)
            .alias(col)
            for col in value_columns
        ],
    )
