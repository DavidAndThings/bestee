"""Tests for the per-asset autoregressive-HMM regime detection functions.

Two layers: (1) the ``_GaussianARHMM`` engine is checked for *recovery* -- given
data simulated from a known switching-AR process it must learn the regimes back;
(2) ``run_regime_analysis`` is exercised end-to-end through an injected
panel (no network), asserting per-asset results, canonical labels, the overfit
guard, and the wide ``regime_frame`` join.
"""

import datetime as dt
from typing import Any

import numpy as np
import polars as pl
import pytest

from bestee_compute.stocks.models import OHLCHeader
from bestee_compute.workflow import regimes
from bestee_compute.workflow.tools import AnalysisConfig

TS = OHLCHeader.TIMESTAMP
CLOSE = OHLCHeader.CLOSE


def _weekdays(n: int) -> list[dt.datetime]:
    out: list[dt.datetime] = []
    day = dt.datetime(2023, 1, 2, tzinfo=dt.UTC)
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day)
        day += dt.timedelta(days=1)
    return out


def _simulate_ar_hmm(
    transmat: np.ndarray,
    startprob: np.ndarray,
    coef: np.ndarray,
    covars: np.ndarray,
    lag: int,
    n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``(states, observations)`` from a switching VAR(``lag``)."""
    rng = np.random.default_rng(seed)
    k = len(startprob)
    d = coef.shape[1]
    states = np.empty(n, dtype=int)
    states[0] = rng.choice(k, p=startprob)
    for t in range(1, n):
        states[t] = rng.choice(k, p=transmat[states[t - 1]])
    obs = np.zeros((n, d))
    obs[:lag] = rng.normal(0.0, 0.1, size=(lag, d))
    for t in range(lag, n):
        regressor = np.concatenate([[1.0], *[obs[t - i] for i in range(1, lag + 1)]])
        mean = coef[states[t]] @ regressor
        cov = covars[states[t]]
        if cov.ndim == 1:
            obs[t] = mean + rng.normal(0.0, np.sqrt(cov))
        else:
            obs[t] = rng.multivariate_normal(mean, cov)
    return states, obs


def _regime_close_panel(
    tickers: list[str], *, n_days: int = 500, seed: int = 3
) -> pl.DataFrame:
    """Close panel whose idiosyncratic volatility switches calm -> turbulent."""
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0, 0.008, n_days)
    sigma = np.where(np.arange(n_days) < n_days // 2, 0.004, 0.020)
    data: dict[str, object] = {TS: _weekdays(n_days)}
    for ticker in tickers:
        returns = market + rng.normal(0.0, 1.0, n_days) * sigma
        data[f"{ticker}_{CLOSE}"] = 100.0 * np.exp(np.cumsum(returns))
    return pl.DataFrame(data)


def _config(panel: pl.DataFrame, tickers: list[str], **kwargs: Any) -> AnalysisConfig:
    params: dict[str, Any] = {
        "tickers": tickers,
        "start_date": "2023-01-01",
        "end_date": "2024-12-31",
        "injected_panel": panel,
        "ohlc_column": CLOSE,
        "residualization_window": 20,
        "normalization_window": 20,
        "regime_hmm_hidden_states": 2,
        "regime_hmm_lag": 1,
        "regime_vol_halflife": 10,
        "regime_n_init": 2,
        "regime_n_iter": 50,
        "regime_min_obs_per_param": 5,
        "regime_random_state": 0,
        "max_workers": 4,
    }
    params.update(kwargs)
    return AnalysisConfig(**params)


class TestGaussianARHMMEngine:
    def test_recovers_two_volatility_regimes(self) -> None:
        # Persistent calm/turbulent AR(1) regimes that differ only in variance.
        transmat = np.array([[0.97, 0.03], [0.04, 0.96]])
        startprob = np.array([1.0, 0.0])
        coef = np.array([[[0.0, 0.3]], [[0.0, 0.3]]])  # (K, d=1, 1+lag*d=2)
        covars = np.array([[0.05], [1.5]])  # diag (K, d): calm vs turbulent
        truth, obs = _simulate_ar_hmm(
            transmat, startprob, coef, covars, lag=1, n=3000, seed=0
        )

        model = regimes._GaussianARHMM(
            n_components=2, lag=1, covariance_type="diag", n_iter=200, random_state=0
        )
        model.fit(obs)
        predicted = model.predict(obs)

        truth_v, pred_v = truth[1:], predicted[1:]  # drop the AR warmup row
        accuracy = max((pred_v == truth_v).mean(), (pred_v == 1 - truth_v).mean())
        assert accuracy > 0.9
        # One regime's innovation variance is clearly larger than the other's.
        variances = np.sort(model.covars_.sum(axis=1))
        assert variances[1] / variances[0] > 3.0

    def test_recovers_multivariate_regimes(self) -> None:
        transmat = np.array([[0.98, 0.02], [0.03, 0.97]])
        startprob = np.array([1.0, 0.0])
        coef = np.zeros((2, 2, 3))  # (K, d=2, 1+lag*d=3), zero AR for a clean test
        covars = np.array([np.eye(2) * 0.1, np.eye(2) * 2.0])  # full cov
        truth, obs = _simulate_ar_hmm(
            transmat, startprob, coef, covars, lag=1, n=2500, seed=1
        )
        model = regimes._GaussianARHMM(
            n_components=2, lag=1, covariance_type="full", n_iter=200, random_state=0
        )
        model.fit(obs)
        predicted = model.predict(obs)
        truth_v, pred_v = truth[1:], predicted[1:]
        accuracy = max((pred_v == truth_v).mean(), (pred_v == 1 - truth_v).mean())
        assert accuracy > 0.9


class TestRegimeAnalysisRunner:
    def test_returns_per_asset_results(self) -> None:
        tickers = ["AAA", "BBB", "CCC"]
        config = _config(_regime_close_panel(tickers), tickers)
        results = regimes.run_regime_analysis(config)

        assert set(results) <= set(tickers) and results
        for ticker, result in results.items():
            assert isinstance(result, regimes.RegimeResult)
            assert result.states.columns == [
                TS,
                "Regime",
                "Regime_Prob_0",
                "Regime_Prob_1",
            ]
            regime_labels = result.states["Regime"].to_numpy()
            assert set(np.unique(regime_labels)) <= {0, 1}
            probs = result.states.select("Regime_Prob_0", "Regime_Prob_1").to_numpy()
            assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
            assert result.current_regime() in (0, 1)
            assert result.expected_durations().shape == (2,)
            assert np.isfinite(result.bic())

    def test_canonical_labels_put_calm_regime_first(self) -> None:
        tickers = ["AAA", "BBB"]
        config = _config(_regime_close_panel(tickers, seed=5), tickers)
        results = regimes.run_regime_analysis(config)
        assert results
        for result in results.values():
            # Regime 0 must be the lower-innovation-volatility regime.
            volatility = (
                result.covars.sum(axis=1)
                if result.covars.ndim == 2
                else np.trace(result.covars, axis1=1, axis2=2)
            )
            assert volatility[0] <= volatility[1]

    def test_detects_the_volatility_switch(self) -> None:
        # Calm first half, turbulent second half -> later regimes should be higher.
        tickers = ["AAA", "BBB", "CCC"]
        config = _config(_regime_close_panel(tickers, seed=7), tickers)
        results = regimes.run_regime_analysis(config)
        assert results
        elevated = 0
        for result in results.values():
            regime_labels = result.states["Regime"].to_numpy()
            third = len(regime_labels) // 3
            if regime_labels[-third:].mean() > regime_labels[:third].mean():
                elevated += 1
        assert elevated >= len(results) - 1  # at least all but one asset

    def test_regime_frame_joins_assets_wide(self) -> None:
        tickers = ["AAA", "BBB"]
        config = _config(_regime_close_panel(tickers), tickers)
        results = regimes.run_regime_analysis(config)
        frame = regimes.regime_frame(results)
        expected = {str(TS)}
        for ticker in results:
            expected |= {f"{ticker}_Regime", f"{ticker}_Regime_Prob"}
        assert set(frame.columns) == expected
        assert frame.height > 0

    def test_param_budget_skips_short_history(self) -> None:
        tickers = ["AAA", "BBB"]
        config = _config(
            _regime_close_panel(tickers, n_days=90),
            tickers,
            regime_min_obs_per_param=50,  # far more than the short history can support
        )
        assert regimes.run_regime_analysis(config) == {}

    def test_features_validator_rejects_empty(self) -> None:
        tickers = ["AAA"]
        with pytest.raises(ValueError, match="at least one feature"):
            _config(_regime_close_panel(tickers), tickers, regime_features=[])
