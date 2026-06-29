"""Per-asset regime detection via an autoregressive Gaussian HMM.

Each asset's *normalized residual* (idiosyncratic, market-neutral) plus a small,
deliberately minimal set of derived features (default: the residual itself and a
trailing **log-volatility** of the *raw* residual) is modeled as a switching
VAR(p): a hidden Markov chain whose emission is an order-``hmm_lag``
vector-autoregression with regime-specific coefficients and innovation
covariance. One HMM is fit **per asset**, so every name gets its own regime
path (e.g. calm vs. turbulent idiosyncratic dynamics).

The residual is market-model (OLS) idiosyncratic return when a
``benchmark_ticker`` is set -- each asset regressed on the benchmark over a
rolling window -- and rolling-PCA reconstruction error over the basket itself
when it is not, so a benchmark is optional rather than required.

The volatility feature is taken from the *raw* residual, not the normalized one:
``get_normalized_*`` z-scores over a rolling window, so the normalized residual
has ~unit rolling variance by construction and its trailing volatility carries
almost no regime signal -- the raw residual's does.
"""

import datetime as dt
import logging
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import reduce
from typing import Any, Literal

import numpy as np
import polars as pl
from hmmlearn.base import BaseHMM
from pydantic import BaseModel, ConfigDict, Field
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state

from bestee_compute.stocks.models import OHLCHeader
from bestee_compute.workflow.tools import (
    AnalysisConfig,
    RegimeFeature,
    TickerClass,
    get_normalized_ols_residuals,
    get_normalized_pca_residuals,
    get_ols_residuals,
    get_pca_residuals,
    get_tickers,
)

logger = logging.getLogger(__name__)

_REGIME_COLUMN = "Regime"
_REGIME_PROB_COLUMN = "Regime_Prob"
_REGIME_PROB_PREFIX = "Regime_Prob"
_FEATURE_SEPARATOR = "__"
# Floor inside ``log`` so a zero-variance window can't produce ``-inf``.
_VOL_EPS = 1e-12


class _GaussianARHMM(BaseHMM):
    """Switching VAR(p): a per-state Gaussian autoregressive emission.

    Observation ``y_t`` in ``R^d``; regressor ``x_t = [1, y_{t-1}, ..., y_{t-p}]``
    (``p = lag``). The emission is
    ``y_t | (state s, x_t) ~ N(coef_[s] @ x_t, covars_[s])`` -- i.e. each regime
    is a VAR(p) with its own coefficients and innovation covariance. Built on
    hmmlearn's EM/forward-backward/Viterbi by supplying only the AR-Gaussian
    emission hooks.

    The first ``lag`` rows have no full lag history, so they get a uniform
    (log-likelihood 0) emission and are excluded from the M-step; callers should
    drop the first ``lag`` decoded labels.
    """

    def __init__(
        self,
        n_components: int,
        lag: int,
        *,
        covariance_type: Literal["diag", "full"] = "diag",
        min_covar: float = 1e-4,
        reg: float = 1e-6,
        n_iter: int = 100,
        tol: float = 1e-4,
        transmat_prior: float | np.ndarray = 1.0,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            n_components=n_components,
            n_iter=n_iter,
            tol=tol,
            # hmmlearn accepts an array prior at runtime; its stub says float.
            transmat_prior=transmat_prior,  # type: ignore[arg-type]
            random_state=random_state,
            verbose=verbose,
            init_params="stac",
            params="stac",
            implementation="log",
        )
        self.lag = lag
        self.covariance_type = covariance_type
        self.min_covar = min_covar
        self.reg = reg

    def _get_n_fit_scalars_per_param(self) -> dict[str, int]:
        # Free scalars per fittable param (hmmlearn's data-sufficiency check).
        k, d = self.n_components, self.n_features
        return {
            "s": k - 1,
            "t": k * (k - 1),
            "a": k * d * (1 + self.lag * d),
            "c": k * d if self.covariance_type == "diag" else k * d * (d + 1) // 2,
        }

    # ---- autoregressive design ------------------------------------------
    def _design(
        self, observations: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(rows, regressors, targets)``: indices ``t >= lag`` with AR design."""
        n, d = observations.shape
        p = self.lag
        if n <= p:
            return np.empty(0, dtype=int), np.empty((0, 1 + p * d)), np.empty((0, d))
        rows = np.arange(p, n)
        columns = [np.ones((rows.size, 1))] + [
            observations[rows - i] for i in range(1, p + 1)
        ]
        return rows, np.concatenate(columns, axis=1), observations[rows]

    def _log_gaussian(self, targets: np.ndarray, regressors: np.ndarray) -> np.ndarray:
        """``(m, K)`` Gaussian log-density of each target under each state."""
        m, d = targets.shape
        log_prob = np.empty((m, self.n_components))
        const = d * np.log(2.0 * np.pi)
        for s in range(self.n_components):
            error = targets - regressors @ self.coef_[s].T
            if self.covariance_type == "diag":
                var = self.covars_[s]
                log_det = np.log(var).sum()
                maha = ((error * error) / var).sum(axis=1)
            else:
                cov = self.covars_[s]
                _, log_det = np.linalg.slogdet(cov)
                solved = np.linalg.solve(cov, error.T).T
                maha = (error * solved).sum(axis=1)
            log_prob[:, s] = -0.5 * (const + log_det + maha)
        return log_prob

    # ---- hmmlearn emission hooks ----------------------------------------
    def _init(
        self,
        X: np.ndarray,  # noqa: N803
        lengths: Sequence[int] | None = None,
    ) -> None:
        super()._init(X, lengths)
        data = np.asarray(X, dtype=float)
        d = data.shape[1]
        self.n_features = d
        k, p = self.n_components, self.lag
        q = 1 + p * d
        self.coef_ = np.zeros((k, d, q))
        self.covars_ = (
            np.ones((k, d))
            if self.covariance_type == "diag"
            else np.tile(np.eye(d), (k, 1, 1))
        )
        rows, regressors, targets = self._design(data)
        if rows.size == 0:
            return
        random = check_random_state(self.random_state)
        if targets.shape[0] >= k:
            labels = KMeans(
                n_clusters=k,
                n_init="auto",
                random_state=random.randint(np.iinfo(np.int32).max),
            ).fit_predict(targets)
        else:
            labels = np.zeros(targets.shape[0], dtype=int)
        for s in range(k):
            self._init_state(s, regressors[labels == s], targets[labels == s])

    def _init_state(self, s: int, regressors: np.ndarray, targets: np.ndarray) -> None:
        d = self.n_features
        q = 1 + self.lag * d
        if regressors.shape[0] >= q:
            coef = np.linalg.lstsq(regressors, targets, rcond=None)[0].T
            error = targets - regressors @ coef.T
            cov = np.atleast_2d(np.cov(error, rowvar=False))
        else:
            coef = np.zeros((d, q))
            if targets.shape[0]:
                coef[:, 0] = targets.mean(axis=0)
            cov = np.eye(d)
        self.coef_[s] = coef
        if self.covariance_type == "diag":
            self.covars_[s] = np.clip(np.diag(cov), self.min_covar, None)
        else:
            self.covars_[s] = cov + self.min_covar * np.eye(d)

    def _compute_log_likelihood(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        # Warmup rows (no full lag history) stay 0 == uniform across states.
        log_prob = np.zeros((X.shape[0], self.n_components))
        rows, regressors, targets = self._design(X)
        if rows.size:
            log_prob[rows] = self._log_gaussian(targets, regressors)
        return log_prob

    def _initialize_sufficient_statistics(self) -> dict:
        stats = super()._initialize_sufficient_statistics()
        k, d = self.n_components, self.n_features
        q = 1 + self.lag * d
        stats["Sxx"] = np.zeros((k, q, q))
        stats["Sxy"] = np.zeros((k, q, d))
        stats["Syy"] = np.zeros((k, d, d))
        stats["Sw"] = np.zeros(k)
        return stats

    def _accumulate_sufficient_statistics(
        self,
        stats,
        X,  # noqa: N803
        lattice,
        posteriors,
        fwdlattice,
        bwdlattice,
    ) -> None:
        super()._accumulate_sufficient_statistics(
            stats, X, lattice, posteriors, fwdlattice, bwdlattice
        )
        rows, regressors, targets = self._design(X)
        if rows.size == 0:
            return
        weights = posteriors[rows]
        for s in range(self.n_components):
            weighted = regressors * weights[:, s : s + 1]
            stats["Sxx"][s] += weighted.T @ regressors
            stats["Sxy"][s] += weighted.T @ targets
            stats["Syy"][s] += (targets * weights[:, s : s + 1]).T @ targets
            stats["Sw"][s] += weights[:, s].sum()

    def _do_mstep(self, stats) -> None:
        super()._do_mstep(stats)  # startprob_, transmat_
        d, q = self.n_features, 1 + self.lag * self.n_features
        for s in range(self.n_components):
            # State-weighted multivariate OLS: coef = Syx @ Sxx^-1 (ridge-stabilized).
            sxx = stats["Sxx"][s] + self.reg * np.eye(q)
            coef = np.linalg.solve(sxx, stats["Sxy"][s]).T
            self.coef_[s] = coef
            weight = stats["Sw"][s]
            if weight > 0:
                cov = (stats["Syy"][s] - coef @ stats["Sxy"][s]) / weight
            else:
                cov = np.eye(d)
            cov = 0.5 * (cov + cov.T)
            if self.covariance_type == "diag":
                self.covars_[s] = np.clip(np.diag(cov), self.min_covar, None)
            else:
                self.covars_[s] = cov + self.min_covar * np.eye(d)

    def volatility_order(self) -> np.ndarray:
        """States sorted by ascending innovation volatility (calmest first)."""
        if self.covariance_type == "diag":
            magnitude = self.covars_.sum(axis=1)
        else:
            magnitude = np.trace(self.covars_, axis1=1, axis2=2)
        return np.argsort(magnitude)


class RegimeResult(BaseModel):
    """One asset's fitted regime model and its decoded regime path."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ticker: str
    # [Timestamp, Regime, Regime_Prob_0 .. Regime_Prob_{K-1}], warmup trimmed.
    states: pl.DataFrame = Field(repr=False)
    transition_matrix: np.ndarray = Field(repr=False)
    start_prob: np.ndarray = Field(repr=False)
    coef: np.ndarray = Field(repr=False)
    covars: np.ndarray = Field(repr=False)
    feature_names: list[str]
    n_states: int
    lag: int
    log_likelihood: float
    n_params: int

    def current_regime(self) -> int:
        """The most recent decoded regime (0 = calmest after canonicalization)."""
        return int(self.states[_REGIME_COLUMN][-1])

    def expected_durations(self) -> np.ndarray:
        """Expected dwell time per regime, ``1 / (1 - A_ii)`` (in bars)."""
        return 1.0 / (1.0 - np.clip(np.diag(self.transition_matrix), None, 1 - 1e-12))

    def bic(self) -> float:
        """Bayesian information criterion (lower is better) for model selection."""
        return -2.0 * self.log_likelihood + self.n_params * np.log(self.states.height)

    def regime_summary(self) -> pl.DataFrame:
        """Per-regime frequency, model dwell, and realized decoded run length.

        ``ModelDwell`` is ``1 / (1 - A_ii)`` from the transition matrix;
        ``RealizedDwell`` is the mean length of the regime's runs in the decoded
        path. A strong ``sticky`` prior inflates ``ModelDwell`` above the
        realized value, so ``RealizedDwell`` is the honest persistence measure.
        """
        labels = self.states[_REGIME_COLUMN].to_numpy()
        total = len(labels)
        breaks = np.where(np.diff(labels) != 0)[0]
        starts = np.array([0, *(breaks + 1).tolist()])
        lengths = np.diff([*starts.tolist(), total])
        realized = np.zeros(self.n_states)
        for s in range(self.n_states):
            runs = lengths[labels[starts] == s]
            realized[s] = float(runs.mean()) if runs.size else 0.0
        durations = self.expected_durations()
        return pl.DataFrame(
            {
                "Regime": list(range(self.n_states)),
                "Frequency": [
                    float((labels == s).mean()) for s in range(self.n_states)
                ],
                "ModelDwell": durations.tolist(),
                "RealizedDwell": realized.tolist(),
            }
        )


def run_regime_analysis(
    config: AnalysisConfig,
    ticker_class: TickerClass = "all",
) -> dict[str, RegimeResult]:
    """Fit one AR-HMM per asset; return ``ticker -> RegimeResult``.

    Assets without enough history for their parameter budget are skipped
    (logged), so the result may omit some tickers.
    """
    frame = _feature_frame(config, ticker_class)
    timestamps = frame[OHLCHeader.TIMESTAMP]
    tickers = get_tickers(config, ticker_class)

    def fit_one(ticker: str) -> tuple[str, RegimeResult | None]:
        columns = [f"{ticker}{_FEATURE_SEPARATOR}{f}" for f in config.regime_features]
        observations = _standardize(frame.select(columns).to_numpy())
        return ticker, _fit_asset(config, ticker, observations, timestamps)

    with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
        results = dict(pool.map(fit_one, tickers))
    return {ticker: r for ticker, r in results.items() if r is not None}


def predict_oos_regimes(
    config: AnalysisConfig,
    oos_dates: Sequence[str | dt.date | dt.datetime],
    ticker_class: TickerClass = "all",
) -> pl.DataFrame:
    """Causal out-of-sample regime nowcast with a fixed (no-refit) AR-HMM.

    Fits one AR-HMM per asset on the **in-sample** window (bars at or before
    ``config.end_date``), then -- holding every fitted model *fixed* -- decodes
    the filtered regime posterior at each date in *oos_dates*. Keeping the model
    fixed is what makes this genuinely out-of-sample: just extending the window
    and re-running :func:`run_regime_analysis` would let the new bars influence
    the EM fit (lookahead) and could permute the state labels between runs
    (label switching). Holding the fit fixed keeps the canonical labels (0 =
    calmest) comparable in and out of sample.

    The features are trailing/causal (rolling residualization and normalization,
    an EWM log-volatility), so the in-sample feature values are unchanged by the
    presence of later bars; the posterior at an OOS date is the forward-filtered
    state probability over the contiguous observation path up to that date
    (``predict_proba(prefix)[-1]``, whose smoothing weight is one at the last
    bar). Standardization uses **in-sample** statistics only, so no future
    information leaks in (this deliberately differs from
    :func:`run_regime_analysis`, whose full-sample z-score is a mild lookahead).

    Returns a wide frame with one row per requested date: ``Timestamp`` (the
    actual bar at or before the date) plus ``{ticker}_Regime`` (canonical label)
    and ``{ticker}_Regime_Prob`` (that regime's filtered posterior) for every
    asset whose model fit. A date before all available history -- or an asset
    with too little in-sample data to fit -- yields nulls.
    """
    if not oos_dates:
        raise ValueError("predict_oos_regimes needs at least one out-of-sample date")
    targets = [_as_date(d) for d in oos_dates]
    # Build features once over the full horizon so the OOS bars exist (the fetch
    # path is bounded by end_date); the trailing features keep the in-sample
    # values identical to a window that had stopped at end_date.
    horizon = max(_as_date(config.end_date), *targets)
    extended = replace(config, end_date=horizon)
    frame = _feature_frame(extended, ticker_class)
    timestamp = frame[OHLCHeader.TIMESTAMP]
    stamps = timestamp.to_numpy()
    in_sample = stamps <= np.datetime64(_as_date(config.end_date))
    indices = [_resolve_oos_index(stamps, target) for target in targets]
    tickers = get_tickers(extended, ticker_class)

    def predict_one(ticker: str) -> tuple[str, list[int | None], list[float | None]]:
        names = [_feature_column(ticker, f) for f in config.regime_features]
        observations = frame.select(names).to_numpy()
        in_obs = observations[in_sample]
        mean = in_obs.mean(axis=0)
        std = in_obs.std(axis=0)
        std = np.where(std == 0.0, 1.0, std)
        model = _select_model(config, ticker, (in_obs - mean) / std)
        if model is None:
            return ticker, [], []
        order = model.volatility_order()
        standardized = (observations - mean) / std
        labels: list[int | None] = []
        confidence: list[float | None] = []
        for index in indices:
            if index is None or index < model.lag:
                labels.append(None)
                confidence.append(None)
                continue
            posterior = model.predict_proba(standardized[: index + 1])[-1][order]
            labels.append(int(np.argmax(posterior)))
            confidence.append(float(posterior.max()))
        return ticker, labels, confidence

    with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
        fitted = list(pool.map(predict_one, tickers))

    columns: dict[str, list[Any]] = {
        OHLCHeader.TIMESTAMP: [
            timestamp[index] if index is not None else None for index in indices
        ]
    }
    for ticker, labels, confidence in fitted:
        if not labels:
            continue
        columns[f"{ticker}_{_REGIME_COLUMN}"] = labels
        columns[f"{ticker}_{_REGIME_PROB_COLUMN}"] = confidence
    return pl.DataFrame(columns)


def _as_date(value: str | dt.date | dt.datetime) -> dt.date:
    """Coerce an ISO string, date, or datetime bound to a plain date."""
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])


def _resolve_oos_index(timestamps: np.ndarray, oos_date: dt.date) -> int | None:
    """Index of the last bar at or before *oos_date* (``None`` if before all)."""
    count = int((timestamps <= np.datetime64(oos_date)).sum())
    return count - 1 if count > 0 else None


def _feature_frame(
    config: AnalysisConfig,
    ticker_class: TickerClass,
) -> pl.DataFrame:
    """``Timestamp`` + every ``{ticker}__{feature}`` column for the slice."""
    tickers = get_tickers(config, ticker_class)
    raw, normalized = _residual_frames(config, ticker_class)
    raw_columns = [c for c in raw.columns if c != OHLCHeader.TIMESTAMP]
    norm_columns = [c for c in normalized.columns if c != OHLCHeader.TIMESTAMP]
    # log-vol from the RAW residual (the normalized one is ~unit-variance).
    volatility = raw.select(
        OHLCHeader.TIMESTAMP,
        *[
            (
                pl.col(raw_columns[i])
                .pow(2)
                .ewm_mean(half_life=config.regime_vol_halflife)
            )
            .add(_VOL_EPS)
            .log()
            .mul(0.5)
            .alias(_feature_column(ticker, "log_vol"))
            for i, ticker in enumerate(tickers)
        ],
    )
    residual = normalized.select(
        OHLCHeader.TIMESTAMP,
        *[
            pl.col(norm_columns[i]).alias(_feature_column(ticker, "residual"))
            for i, ticker in enumerate(tickers)
        ],
        *[
            pl.col(norm_columns[i]).abs().alias(_feature_column(ticker, "abs_residual"))
            for i, ticker in enumerate(tickers)
        ],
    )
    return residual.join(volatility, on=OHLCHeader.TIMESTAMP, how="inner").drop_nulls()


def _residual_frames(
    config: AnalysisConfig,
    ticker_class: TickerClass,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Raw and normalized idiosyncratic residuals for the regime features.

    With a ``benchmark_ticker`` set, residualize each asset against it via the
    rolling market-model (OLS) regression -- the benchmark is the explicit
    market factor. Otherwise fall back to rolling-PCA residualization, which
    strips the basket's own leading factor(s) without needing a benchmark.
    """
    if config.benchmark_ticker is not None:
        return (
            get_ols_residuals(config, ticker_class),
            get_normalized_ols_residuals(config, ticker_class),
        )
    return (
        get_pca_residuals(config, ticker_class),
        get_normalized_pca_residuals(config, ticker_class),
    )


def _feature_column(ticker: str, feature: RegimeFeature) -> str:
    return f"{ticker}{_FEATURE_SEPARATOR}{feature}"


def _select_model(
    config: AnalysisConfig, ticker: str, observations: np.ndarray
) -> _GaussianARHMM | None:
    """Fit the AR-HMM, choosing the state count by BIC within the param budget."""
    dimensions = observations.shape[1]
    usable = observations.shape[0] - config.regime_hmm_lag
    candidates = (
        range(config.regime_state_range[0], config.regime_state_range[1] + 1)
        if config.regime_auto_select_states
        else [config.regime_hmm_hidden_states]
    )
    best: _GaussianARHMM | None = None
    best_bic = np.inf
    for k in candidates:
        n_params = _n_params(config, k, dimensions)
        if usable < config.regime_min_obs_per_param * n_params:
            logger.warning(
                "Skipping %s K=%d: needs >=%d usable bars per the budget, has %d",
                ticker,
                k,
                config.regime_min_obs_per_param * n_params,
                usable,
            )
            continue
        model = _fit_best(config, observations, k)
        if model is None:
            continue
        bic = -2.0 * model.score(observations) + n_params * np.log(usable)
        if bic < best_bic:
            best, best_bic = model, bic
    if best is None:
        logger.warning("Skipping %s: no AR-HMM converged within the budget", ticker)
    return best


def _fit_asset(
    config: AnalysisConfig,
    ticker: str,
    observations: np.ndarray,
    timestamps: pl.Series,
) -> RegimeResult | None:
    best = _select_model(config, ticker, observations)
    if best is None:
        return None
    return _build_result(config, ticker, best, observations, timestamps)


def _fit_best(
    config: AnalysisConfig, observations: np.ndarray, k: int
) -> _GaussianARHMM | None:
    """Best of ``n_init`` random restarts by log-likelihood (EM is non-convex)."""
    # Sticky self-transition prior scaled by the sample length (T-robust).
    transmat_prior = 1.0 + config.regime_sticky * observations.shape[0] * np.eye(k)
    best: _GaussianARHMM | None = None
    best_score = -np.inf
    for restart in range(config.regime_n_init):
        model = _GaussianARHMM(
            n_components=k,
            lag=config.regime_hmm_lag,
            covariance_type=config.regime_covariance_type,
            min_covar=config.regime_min_covar,
            n_iter=config.regime_n_iter,
            transmat_prior=transmat_prior,
            random_state=config.regime_random_state + restart,
        )
        try:
            model.fit(observations)
            score = float(model.score(observations))
        except (ValueError, np.linalg.LinAlgError) as error:
            logger.debug("AR-HMM restart %d failed: %s", restart, error)
            continue
        if np.isfinite(score) and score > best_score:
            best, best_score = model, score
    return best


def _build_result(
    config: AnalysisConfig,
    ticker: str,
    model: _GaussianARHMM,
    observations: np.ndarray,
    timestamps: pl.Series,
) -> RegimeResult:
    lag, k = model.lag, model.n_components
    states = model.predict(observations)[lag:]
    posteriors = model.predict_proba(observations)[lag:]
    # Canonicalize: relabel so regime 0 is the calmest (lowest innovation vol).
    order = model.volatility_order()
    new_of_old = np.empty(k, dtype=int)
    new_of_old[order] = np.arange(k)
    states = new_of_old[states]
    posteriors = posteriors[:, order]
    states_frame = pl.DataFrame(
        {OHLCHeader.TIMESTAMP: timestamps.slice(lag), _REGIME_COLUMN: states}
    ).with_columns(
        pl.Series(f"{_REGIME_PROB_PREFIX}_{j}", posteriors[:, j]) for j in range(k)
    )
    return RegimeResult(
        ticker=ticker,
        states=states_frame,
        transition_matrix=model.transmat_[np.ix_(order, order)],
        start_prob=model.startprob_[order],
        coef=model.coef_[order],
        covars=model.covars_[order],
        feature_names=list(config.regime_features),
        n_states=k,
        lag=lag,
        log_likelihood=float(model.score(observations)),
        n_params=_n_params(config, k, observations.shape[1]),
    )


def _n_params(config: AnalysisConfig, k: int, dimensions: int) -> int:
    regressors = 1 + config.regime_hmm_lag * dimensions
    coef = k * dimensions * regressors
    covariance = (
        k * dimensions
        if config.regime_covariance_type == "diag"
        else k * dimensions * (dimensions + 1) // 2
    )
    chain = (k - 1) + k * (k - 1)  # start probs + transition rows
    return chain + coef + covariance


def _standardize(observations: np.ndarray) -> np.ndarray:
    """Column z-score (full sample) so mixed-scale features are comparable."""
    mean = observations.mean(axis=0)
    std = observations.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return (observations - mean) / std


def regime_frame(results: Mapping[str, RegimeResult]) -> pl.DataFrame:
    """Join per-asset regime paths into one wide ``{ticker}_Regime[_Prob]`` frame."""
    frames = [
        result.states.select(
            OHLCHeader.TIMESTAMP,
            pl.col(_REGIME_COLUMN).alias(f"{ticker}_{_REGIME_COLUMN}"),
            pl.max_horizontal(
                f"{_REGIME_PROB_PREFIX}_{j}" for j in range(result.n_states)
            ).alias(f"{ticker}_{_REGIME_PROB_COLUMN}"),
        )
        for ticker, result in results.items()
    ]
    if not frames:
        return pl.DataFrame()
    return reduce(
        lambda left, right: left.join(
            right, on=OHLCHeader.TIMESTAMP, how="full", coalesce=True
        ),
        frames,
    ).sort(OHLCHeader.TIMESTAMP)
