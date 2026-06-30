"""Tests for the result-serving ``aspect`` views on the tuning objects.

These exercise the pure table transformations each ``*Tuning`` exposes for the
results API (``cluster_label``, ``coordinates``, ``regime_label``,
``ff_residuals``) plus the shared :func:`frame_to_table` serializer. The tuning
objects are built directly from synthetic data -- no fitting, no network.
"""

import datetime as dt

import numpy as np
import polars as pl

from bestee_compute.stocks.models import OHLCHeader
from bestee_compute.workflow import regimes
from bestee_compute.workflow.tuning import (
    ClusteringTuning,
    FamaFrenchTuning,
    RegimeTuning,
    RRGTuning,
    frame_to_table,
)

TS = OHLCHeader.TIMESTAMP


def test_frame_to_table_stringifies_temporals() -> None:
    frame = pl.DataFrame({"date": [dt.date(2024, 1, 1)], "x": [1.5]})
    table = frame_to_table(frame)
    assert table["columns"] == ["date", "x"]
    assert table["rows"] == [{"date": "2024-01-01", "x": 1.5}]


def test_cluster_label_table() -> None:
    tuning = ClusteringTuning(
        labels={"BBB": 1, "AAA": 0, "CCC": 1},
        residualization_window=20,
        normalization_window=20,
        n_components=1,
        similarity_metric="corr",
        silhouette=0.5,
        n_clusters=2,
    )
    table = tuning.cluster_label_table()
    assert table.columns == ["ticker", "cluster_label"]
    assert table["ticker"].to_list() == ["AAA", "BBB", "CCC"]  # sorted
    assert table["cluster_label"].to_list() == [0, 1, 1]


def test_coordinates_table_melts_and_joins() -> None:
    timestamps = [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2)]
    strength = pl.DataFrame(
        {TS: timestamps, "AAA_rel_strength": [1.0, 1.1], "BBB_rel_strength": [0.9, 0.8]}
    )
    momentum = pl.DataFrame(
        {
            TS: timestamps,
            "AAA_rel_momentum": [0.5, 0.6],
            "BBB_rel_momentum": [-0.1, -0.2],
        }
    )
    tuning = RRGTuning(
        normalization_window=20,
        momentum_lookback=5,
        smoothing_span=None,
        signal_to_noise=1.0,
        relative_strength=strength,
        relative_momentum=momentum,
    )
    table = tuning.coordinates_table()
    assert table.columns == [
        "timestamp",
        "ticker",
        "relative_strength",
        "relative_momentum",
    ]
    assert table.height == 4  # 2 tickers x 2 timestamps
    aaa = table.filter(pl.col("ticker") == "AAA").sort("timestamp")
    assert aaa["relative_strength"].to_list() == [1.0, 1.1]
    assert aaa["relative_momentum"].to_list() == [0.5, 0.6]


def _regime_result(
    ticker: str, timestamps: list[dt.datetime], labels: list[int]
) -> regimes.RegimeResult:
    states = pl.DataFrame(
        {
            TS: timestamps,
            "Regime": labels,
            "Regime_Prob_0": [0.5] * len(labels),
            "Regime_Prob_1": [0.5] * len(labels),
        }
    )
    return regimes.RegimeResult(
        ticker=ticker,
        states=states,
        transition_matrix=np.eye(2),
        start_prob=np.array([0.5, 0.5]),
        coef=np.zeros((2, 1, 2)),
        covars=np.ones((2, 1)),
        feature_names=["residual", "log_vol"],
        n_states=2,
        lag=1,
        log_likelihood=-10.0,
        n_params=5,
    )


def _regime_tuning(oos: pl.DataFrame) -> RegimeTuning:
    timestamps = [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2)]
    return RegimeTuning(
        labels={"AAA": 1, "BBB": 1},
        results={
            "AAA": _regime_result("AAA", timestamps, [0, 1]),
            "BBB": _regime_result("BBB", timestamps, [1, 1]),
        },
        residualization_window=20,
        normalization_window=20,
        hmm_lag=1,
        mean_bic=100.0,
        oos_regimes=oos,
    )


def test_regime_label_table_includes_in_sample_and_oos() -> None:
    oos = pl.DataFrame(
        {
            TS: [dt.datetime(2024, 2, 1)],
            "AAA_Regime": [1],
            "AAA_Regime_Prob": [0.8],
            "BBB_Regime": [0],
            "BBB_Regime_Prob": [0.7],
        }
    )
    table = _regime_tuning(oos).regime_label_table()
    assert table.columns == ["timestamp", "ticker", "regime_label"]
    assert table.height == 6  # 2 tickers x (2 in-sample + 1 oos)
    aaa = table.filter(pl.col("ticker") == "AAA").sort("timestamp")
    # In-sample path (0, 1) then the fixed-model out-of-sample label (1).
    assert aaa["regime_label"].to_list() == [0, 1, 1]
    assert aaa["timestamp"].to_list()[-1] == dt.datetime(2024, 2, 1)


def test_regime_label_table_in_sample_only_without_oos() -> None:
    table = _regime_tuning(pl.DataFrame()).regime_label_table()
    assert table.height == 4  # only the in-sample path, no oos rows
    assert table["timestamp"].max() == dt.datetime(2024, 1, 2)


def test_ff_residuals_table_keeps_pvalue_per_specification() -> None:
    oos = pl.DataFrame(
        {
            "Date": [dt.date(2024, 1, 15), dt.date(2024, 1, 15)],
            "Ticker": ["AAA", "BBB"],
            "Specification": ["FF6_2020-01-01_2023-12-31"] * 2,
            "Residual": [0.01, -0.02],
            "ZScore": [1.0, -2.0],
            "PValue": [0.31, 0.045],
            "Estimated": [False, True],
        }
    )
    tuning = FamaFrenchTuning(
        factors_to_use=6,
        specifications=[],
        results={},
        mean_adjusted_r_squared=0.5,
        oos_residuals=oos,
    )
    table = tuning.ff_residuals_table()
    assert table.columns == [
        "date",
        "ticker",
        "specification",
        "residual",
        "p_value",
        "estimated",
    ]
    assert "ZScore" not in table.columns
    assert table.height == 2
    bbb = table.filter(pl.col("ticker") == "BBB")
    assert bbb["p_value"].to_list() == [0.045]
    assert bbb["estimated"].to_list() == [True]


def test_ff_residuals_table_defaults_estimated_for_legacy_results() -> None:
    # Results persisted before the estimated-factor feature have no Estimated
    # column; the aspect view must still render, defaulting it to False.
    oos = pl.DataFrame(
        {
            "Date": [dt.date(2024, 1, 15)],
            "Ticker": ["AAA"],
            "Specification": ["FF6_2020-01-01_2023-12-31"],
            "Residual": [0.01],
            "ZScore": [1.0],
            "PValue": [0.31],
        }
    )
    tuning = FamaFrenchTuning(
        factors_to_use=6,
        specifications=[],
        results={},
        mean_adjusted_r_squared=0.5,
        oos_residuals=oos,
    )
    table = tuning.ff_residuals_table()
    assert table["estimated"].to_list() == [False]
