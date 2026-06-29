"""Tests for the ``AnalysisConfig`` data-layer refactor.

These exercise the wiring rather than the network: an injected ``panel`` is
sliced (no fetch), and the fetch fallbacks are validated by patching
``tools.get_ohlc`` / ``tools.get_grouped_daily_column``. Downstream analytics
(log returns, PCA/OLS residuals, RRG, spectral clustering) are checked to read
their data through the shared ``AnalysisConfig``.
"""

import datetime as dt
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from bestee_compute.stocks.models import OHLCHeader
from bestee_compute.workflow import clustering, rrg, tools
from bestee_compute.workflow.tools import AnalysisConfig

TS = OHLCHeader.TIMESTAMP
CLOSE = OHLCHeader.CLOSE
VOLUME = OHLCHeader.VOLUME

_OHLC_PATCH = "bestee_compute.workflow.tools.get_ohlc"
_BULK_PATCH = "bestee_compute.workflow.tools.get_grouped_daily_column"


def _weekdays(n: int) -> list[dt.datetime]:
    out: list[dt.datetime] = []
    day = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day)
        day += dt.timedelta(days=1)
    return out


def _close_volume_panel(
    tickers: list[str],
    *,
    n_days: int = 160,
    seed: int = 1,
    dollar_volumes: dict[str, float] | None = None,
) -> pl.DataFrame:
    """Wide ``{ticker}_Close`` + ``{ticker}_Volume`` panel of geometric walks.

    Volume is set to ``dollar_volume / close`` so each name's median daily
    dollar volume equals its requested ``dollar_volumes`` entry (default 1e7).
    """
    rng = np.random.default_rng(seed)
    data: dict[str, object] = {TS: _weekdays(n_days)}
    for ticker in tickers:
        price = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n_days)))
        target = (dollar_volumes or {}).get(ticker, 1e7)
        data[f"{ticker}_{CLOSE}"] = price
        data[f"{ticker}_{VOLUME}"] = target / price
    return pl.DataFrame(data)


def _factor_panel(
    groups: dict[int, list[str]],
    *,
    n_days: int = 220,
    seed: int = 7,
    dollar_volumes: dict[str, float] | None = None,
) -> pl.DataFrame:
    """Panel where each group shares a latent factor on top of a market factor.

    After PCA residualization strips the (dominant) market factor, same-group
    names stay correlated and cross-group names decorrelate -- a clean block
    structure for clustering to recover.
    """
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0, 0.012, n_days)
    data: dict[str, object] = {TS: _weekdays(n_days)}
    for members in groups.values():
        factor = rng.normal(0.0, 0.011, n_days)
        for ticker in members:
            returns = market + factor + rng.normal(0.0, 0.0015, n_days)
            price = 100.0 * np.exp(np.cumsum(returns))
            target = (dollar_volumes or {}).get(ticker, 1e8)
            data[f"{ticker}_{CLOSE}"] = price
            data[f"{ticker}_{VOLUME}"] = target / price
    return pl.DataFrame(data)


def _benchmark_factor_panel(
    groups: dict[int, list[str]],
    benchmark: str,
    *,
    n_days: int = 220,
    seed: int = 7,
) -> pl.DataFrame:
    """Like :func:`_factor_panel`, but exposes the market factor as *benchmark*.

    The benchmark tracks the shared market factor, so a rolling market-model
    (OLS) regression against it strips the market and leaves each group's own
    factor -- the same block structure rolling PCA recovers, but the benchmark
    is the explicit market proxy rather than an estimated principal component.
    """
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0, 0.012, n_days)
    data: dict[str, object] = {TS: _weekdays(n_days)}
    benchmark_price = 100.0 * np.exp(np.cumsum(market))
    data[f"{benchmark}_{CLOSE}"] = benchmark_price
    data[f"{benchmark}_{VOLUME}"] = 1e8 / benchmark_price
    for members in groups.values():
        factor = rng.normal(0.0, 0.011, n_days)
        for ticker in members:
            returns = market + factor + rng.normal(0.0, 0.0015, n_days)
            price = 100.0 * np.exp(np.cumsum(returns))
            data[f"{ticker}_{CLOSE}"] = price
            data[f"{ticker}_{VOLUME}"] = 1e8 / price
    return pl.DataFrame(data)


def _source(tickers: list[str], panel: pl.DataFrame, **kwargs: Any) -> AnalysisConfig:
    params: dict[str, Any] = {
        "tickers": tickers,
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "injected_panel": panel,
    }
    params.update(kwargs)
    return AnalysisConfig(**params)


class TestDataSourceColumnAccess:
    def test_column_slices_injected_panel_without_fetching(self) -> None:
        tickers = ["AAA", "BBB", "CCC"]
        panel = _close_volume_panel(tickers)
        source = _source(tickers, panel)
        with patch(_OHLC_PATCH) as ohlc, patch(_BULK_PATCH) as bulk:
            close = tools.column(source, CLOSE)
        assert close.columns == [TS, "AAA_Close", "BBB_Close", "CCC_Close"]
        assert close.height == panel.height
        ohlc.assert_not_called()
        bulk.assert_not_called()

    def test_benchmark_column_slices_injected_panel(self) -> None:
        panel = _close_volume_panel(["AAA", "SPY"])
        source = _source(["AAA"], panel, benchmark_ticker="SPY")
        with patch(_OHLC_PATCH) as ohlc:
            bench = tools.benchmark_column(source, CLOSE)
        assert bench.columns == [TS, "SPY_Close"]
        ohlc.assert_not_called()

    def test_benchmark_column_requires_benchmark(self) -> None:
        source = _source(["AAA"], _close_volume_panel(["AAA"]))
        with pytest.raises(ValueError, match="benchmark_ticker is not set"):
            tools.benchmark_column(source, CLOSE)

    def test_column_falls_through_to_fetch_when_panel_lacks_column(self) -> None:
        # A Close-only panel cannot serve Volume -> it must fetch it.
        tickers = ["AAA", "BBB"]
        close_only = _close_volume_panel(tickers).select(TS, "AAA_Close", "BBB_Close")
        source = _source(tickers, close_only)
        volume = _close_volume_panel(tickers).select(TS, "AAA_Volume", "BBB_Volume")
        with patch(_BULK_PATCH) as bulk, patch(_OHLC_PATCH) as ohlc:
            ohlc.side_effect = lambda ticker, query: volume.rename(
                {f"{ticker}_Volume": VOLUME}
            ).select(TS, VOLUME)
            # 2 tickers < trading days in 2024 -> per-ticker path.
            tools.column(source, VOLUME)
        bulk.assert_not_called()
        assert ohlc.call_count == 2

    def test_column_uses_bulk_when_tickers_outnumber_trading_days(self) -> None:
        tickers = [f"T{i}" for i in range(6)]
        # A 2-weekday window so 6 tickers > 2 trading days -> bulk path.
        source = AnalysisConfig(
            tickers=tickers, start_date="2024-01-02", end_date="2024-01-03"
        )
        wide = pl.DataFrame(
            {TS: _weekdays(2), **{f"{t}_{CLOSE}": [1.0, 2.0] for t in tickers}}
        )
        with patch(_BULK_PATCH, return_value=wide) as bulk, patch(_OHLC_PATCH) as ohlc:
            tools.column(source, CLOSE)
        bulk.assert_called_once()
        ohlc.assert_not_called()

    def test_universe_auto_screens_without_an_injected_panel(self) -> None:
        # 6 tickers over a 2-weekday window -> bulk path; no panel injected.
        tickers = [f"T{i}" for i in range(6)]
        dates = _weekdays(2)
        # T5 only trades 1 of 2 days (50% coverage) -> dropped at the 95% screen.
        close = pl.DataFrame(
            {
                TS: dates,
                **{f"T{i}_{CLOSE}": [10.0 + i, 11.0 + i] for i in range(5)},
                f"T5_{CLOSE}": [None, 7.0],
            }
        )

        def fake_bulk(query, requested, column, *, drop_missing=True, max_workers=16):
            return close.select(TS, *[f"{t}_{column}" for t in requested])

        source = AnalysisConfig(
            tickers=tickers, start_date="2024-01-02", end_date="2024-01-03"
        )
        with patch(_BULK_PATCH, side_effect=fake_bulk), patch(_OHLC_PATCH) as ohlc:
            # No injection: the clean panel is built and screened automatically.
            covered = tools.get_tickers(source, "all")
            column = tools.column(source, CLOSE)
        assert covered == [f"T{i}" for i in range(5)]  # T5 screened out
        assert column.columns == [TS, *[f"T{i}_{CLOSE}" for i in range(5)]]
        assert column[f"T0_{CLOSE}"].null_count() == 0  # gap-filled, rectangular
        ohlc.assert_not_called()  # resolved from the clean panel, not per-ticker

    def test_small_basket_skips_screening_and_keeps_all_tickers(self) -> None:
        tickers = ["AAA", "BBB"]
        dates = _weekdays(10)
        frames = {
            t: pl.DataFrame({TS: dates, CLOSE: [100.0 + i for i in range(10)]})
            for t in tickers
        }
        # 2 tickers < the window's trading days -> per-ticker path, no screening.
        source = AnalysisConfig(
            tickers=tickers, start_date="2024-01-01", end_date="2024-01-31"
        )
        with (
            patch(_OHLC_PATCH, side_effect=lambda t, q: frames[t]) as ohlc,
            patch(_BULK_PATCH) as bulk,
        ):
            assert tools.get_tickers(source, "all") == tickers
            column = tools.column(source, CLOSE)
        assert column.columns == [TS, "AAA_Close", "BBB_Close"]
        bulk.assert_not_called()  # never built a clean panel
        assert ohlc.call_count == 2


class TestLiquiditySplit:
    def test_liquidity_split_reads_from_panel(self) -> None:
        tickers = ["LQ1", "LQ2", "IL1"]
        panel = _close_volume_panel(
            tickers, dollar_volumes={"LQ1": 2e7, "LQ2": 3e7, "IL1": 5e5}
        )
        source = _source(tickers, panel, min_dollar_volume=1e6)
        with patch(_OHLC_PATCH) as ohlc, patch(_BULK_PATCH) as bulk:
            liquidity = tools.get_median_dollar_volume(source)
            assert tools.get_tickers(source, "liquid") == ["LQ1", "LQ2"]
            assert tools.get_tickers(source, "illiquid") == ["IL1"]
            assert tools.get_tickers(source, "all") == tickers
        ohlc.assert_not_called()
        bulk.assert_not_called()
        assert liquidity["LQ1"] == pytest.approx(2e7, rel=1e-6)
        assert liquidity["IL1"] == pytest.approx(5e5, rel=1e-6)

    def test_get_tickers_all_does_not_compute_liquidity(self) -> None:
        # "all" must never touch the (potentially fetching) liquidity median.
        source = _source(["AAA"], _close_volume_panel(["AAA"]).drop("AAA_Volume"))
        with patch(_OHLC_PATCH) as ohlc, patch(_BULK_PATCH) as bulk:
            assert tools.get_tickers(source, "all") == ["AAA"]
        ohlc.assert_not_called()
        bulk.assert_not_called()


class TestGetCleanPanel:
    def test_screens_short_history_and_fills_prices_only(self) -> None:
        tickers = ["FULL", "SHORT"]
        dates = _weekdays(20)
        full_close: list[float | None] = [100.0 + i for i in range(20)]
        full_close[5] = None  # a gap
        full_close[6] = -1.0  # a non-positive print (nulled, then filled)
        close = pl.DataFrame(
            {
                TS: dates,
                "FULL_Close": full_close,
                "SHORT_Close": [None] * 17 + [50.0, 51.0, 52.0],
            }
        )
        full_volume: list[float | None] = [1e6 + i for i in range(20)]
        full_volume[8] = None  # a volume gap that must be PRESERVED
        volume = pl.DataFrame(
            {TS: dates, "FULL_Volume": full_volume, "SHORT_Volume": [2e5] * 20}
        )

        def fake_bulk(query, requested, column, *, drop_missing=True, max_workers=16):
            base = close if column == CLOSE else volume
            return base.select(TS, *[f"{t}_{column}" for t in requested])

        source = AnalysisConfig(
            tickers=tickers,
            start_date="2024-01-01",
            end_date="2024-01-31",
            min_coverage=0.9,
        )
        with patch(_BULK_PATCH, side_effect=fake_bulk):
            panel = tools.get_clean_panel(source, CLOSE, VOLUME)

        assert "FULL_Close" in panel.columns
        assert "SHORT_Close" not in panel.columns  # 15% coverage -> dropped
        assert panel["FULL_Close"].null_count() == 0  # gap + non-positive filled
        # Index 5 (gap) and 6 (non-positive) both forward-fill from index 4 (104.0).
        assert panel["FULL_Close"][5] == pytest.approx(104.0)
        assert panel["FULL_Close"][6] == pytest.approx(104.0)
        assert panel["FULL_Volume"].null_count() == 1  # volume gap preserved

    def test_raises_when_nothing_clears_coverage(self) -> None:
        dates = _weekdays(20)
        close = pl.DataFrame({TS: dates, "X_Close": [None] * 18 + [1.0, 2.0]})

        def fake_bulk(query, requested, column, *, drop_missing=True, max_workers=16):
            return close.select(TS, *[f"{t}_{column}" for t in requested])

        source = AnalysisConfig(
            tickers=["X"],
            start_date="2024-01-01",
            end_date="2024-01-31",
            min_coverage=0.95,
        )
        with patch(_BULK_PATCH, side_effect=fake_bulk):
            with pytest.raises(ValueError, match="coverage screen"):
                tools.get_clean_panel(source, CLOSE)


class TestDelegatedAnalytics:
    def test_log_returns_delegate_to_panel(self) -> None:
        tickers = ["AAA", "BBB"]
        panel = _close_volume_panel(tickers, n_days=30)
        config = _source(tickers, panel, ohlc_column=CLOSE)
        returns = tools.get_log_return_for_tickers(config)
        assert returns.columns == [TS, "AAA_LogReturn", "BBB_LogReturn"]
        assert returns.height == 29  # first diff dropped

    def test_pca_and_ols_residuals_from_panel(self) -> None:
        tickers = ["AAA", "BBB", "CCC"]
        panel = _close_volume_panel([*tickers, "SPY"], n_days=120)
        config = _source(
            tickers,
            panel,
            benchmark_ticker="SPY",
            ohlc_column=CLOSE,
            residualization_window=20,
            normalization_window=20,
        )
        pca = tools.get_normalized_pca_residuals(config)
        assert [c for c in pca.columns if c != TS] == [
            "AAA_PCA_Residual",
            "BBB_PCA_Residual",
            "CCC_PCA_Residual",
        ]
        assert pca.height > 0
        ols = tools.get_normalized_ols_residuals(config)
        assert [c for c in ols.columns if c != TS] == [
            "AAA_OLS_Residual",
            "BBB_OLS_Residual",
            "CCC_OLS_Residual",
        ]
        assert ols.height > 0

    def test_n_components_validator_uses_data_source_tickers(self) -> None:
        panel = _close_volume_panel(["AAA", "BBB"], n_days=40)
        with pytest.raises(ValueError, match="n_components"):
            _source(
                ["AAA", "BBB"],
                panel,
                ohlc_column=CLOSE,
                residualization_window=20,
                normalization_window=20,
                n_components=3,  # > n_tickers (2)
            )


class TestRRGFromPanel:
    def test_mean_reference_runs_from_panel(self) -> None:
        tickers = ["AAA", "BBB", "CCC"]
        panel = _close_volume_panel(tickers, n_days=120)
        config = _source(
            tickers,
            panel,
            ohlc_column=CLOSE,
            rrg_reference_type="mean",
            normalization_window=14,
            rrg_momentum_lookback=5,
            rrg_smoothing_span=5,
        )
        with patch(_OHLC_PATCH) as ohlc:
            strength = rrg.normalized_relative_strength(config)
            momentum = rrg.normalized_relative_momentum(config)
        ohlc.assert_not_called()
        assert "AAA_rel_strength" in strength.columns
        assert "AAA_rel_momentum" in momentum.columns
        assert strength.height > 0 and momentum.height > 0

    def test_ticker_reference_runs_from_panel(self) -> None:
        tickers = ["AAA", "BBB"]
        panel = _close_volume_panel([*tickers, "SPY"], n_days=120)
        config = _source(
            tickers,
            panel,
            benchmark_ticker="SPY",
            ohlc_column=CLOSE,
            rrg_reference_type="ticker",
            normalization_window=14,
            rrg_momentum_lookback=5,
        )
        with patch(_OHLC_PATCH) as ohlc:
            strength = rrg.normalized_relative_strength(config)
        ohlc.assert_not_called()
        assert strength.height > 0

    def test_ticker_reference_requires_benchmark(self) -> None:
        panel = _close_volume_panel(["AAA"], n_days=30)
        config = _source(
            ["AAA"],
            panel,
            ohlc_column=CLOSE,
            rrg_reference_type="ticker",
            normalization_window=14,
            rrg_momentum_lookback=5,
        )
        with pytest.raises(ValueError, match="reference_ticker is required"):
            rrg.normalized_relative_strength(config)


class TestSpectralClustering:
    def test_recovers_factor_blocks_from_panel(self) -> None:
        groups = {0: ["A1", "A2", "A3"], 1: ["B1", "B2", "B3"]}
        tickers = [t for members in groups.values() for t in members]
        panel = _factor_panel(groups, n_days=220)
        config = _source(
            tickers,
            panel,
            min_dollar_volume=0.0,
            ohlc_column=CLOSE,
            residualization_window=40,
            normalization_window=40,
            clustering_min_num_clusters=2,
            clustering_max_num_clusters=2,
        )
        with patch(_OHLC_PATCH) as ohlc, patch(_BULK_PATCH) as bulk:
            labels = clustering.run_spectral_clustering(config)
        ohlc.assert_not_called()
        bulk.assert_not_called()
        assert set(labels) == set(tickers)
        assert labels["A1"] == labels["A2"] == labels["A3"]
        assert labels["B1"] == labels["B2"] == labels["B3"]
        assert labels["A1"] != labels["B1"]

    def test_benchmark_selects_ols_residuals(self) -> None:
        # A benchmark switches the residualization to the market-model (OLS) fit;
        # without one it stays on rolling PCA over the basket.
        groups = {0: ["A1", "A2"], 1: ["B1", "B2"]}
        tickers = [t for members in groups.values() for t in members]
        panel = _benchmark_factor_panel(groups, "MKT", n_days=120)
        shared = dict(
            min_dollar_volume=0.0,
            ohlc_column=CLOSE,
            residualization_window=20,
            normalization_window=20,
        )
        with_benchmark = _source(tickers, panel, benchmark_ticker="MKT", **shared)
        without_benchmark = _source(tickers, panel, **shared)
        with patch(_OHLC_PATCH), patch(_BULK_PATCH):
            ols = clustering._get_residuals(with_benchmark, "all")
            pca = clustering._get_residuals(without_benchmark, "all")
        assert ols.equals(tools.get_normalized_ols_residuals(with_benchmark, "all"))
        assert pca.equals(tools.get_normalized_pca_residuals(without_benchmark, "all"))

    def test_recovers_factor_blocks_with_benchmark(self) -> None:
        groups = {0: ["A1", "A2", "A3"], 1: ["B1", "B2", "B3"]}
        tickers = [t for members in groups.values() for t in members]
        panel = _benchmark_factor_panel(groups, "MKT", n_days=220)
        config = _source(
            tickers,
            panel,
            benchmark_ticker="MKT",
            min_dollar_volume=0.0,
            ohlc_column=CLOSE,
            residualization_window=40,
            normalization_window=40,
            clustering_min_num_clusters=2,
            clustering_max_num_clusters=2,
        )
        with patch(_OHLC_PATCH) as ohlc, patch(_BULK_PATCH) as bulk:
            labels = clustering.run_spectral_clustering(config)
        ohlc.assert_not_called()
        bulk.assert_not_called()
        assert set(labels) == set(tickers)
        assert labels["A1"] == labels["A2"] == labels["A3"]
        assert labels["B1"] == labels["B2"] == labels["B3"]
        assert labels["A1"] != labels["B1"]

    def test_assigns_illiquid_to_nearest_cluster(self) -> None:
        groups = {0: ["A1", "A2", "A3"], 1: ["B1", "B2", "B3"]}
        tickers = [t for members in groups.values() for t in members]
        # A3 / B3 are micro-caps assigned (not clustered) by the liquidity floor.
        dollar_volumes = {t: 1e8 for t in tickers}
        dollar_volumes["A3"] = dollar_volumes["B3"] = 1e5
        panel = _factor_panel(groups, n_days=220, dollar_volumes=dollar_volumes)
        config = _source(
            tickers,
            panel,
            min_dollar_volume=1e6,
            ohlc_column=CLOSE,
            residualization_window=40,
            normalization_window=40,
            clustering_assign_illiquid=True,
            clustering_min_num_clusters=2,
            clustering_max_num_clusters=2,
        )
        with patch(_OHLC_PATCH) as ohlc, patch(_BULK_PATCH) as bulk:
            labels = clustering.run_spectral_clustering(config)
        ohlc.assert_not_called()
        bulk.assert_not_called()
        assert set(labels) == set(tickers)
        assert len(set(labels.values())) == 2
        assert labels["A3"] == labels["A1"]  # illiquid folded into its factor block
        assert labels["B3"] == labels["B1"]


class TestTimestampAlignment:
    def test_grouped_daily_matches_get_ohlc_precision(self) -> None:
        # Regression: pl.from_epoch yields microsecond precision regardless of
        # the input unit, so the bulk path must normalize back to the declared
        # millisecond dtype -- otherwise it can't inner-join the per-ticker
        # get_ohlc fetch (e.g. a benchmark close merged in _prepare_panel).
        from massive.rest.models import GroupedDailyAgg

        bar = MagicMock(spec=GroupedDailyAgg)
        bar.ticker = "AAA"
        bar.timestamp = 1_731_628_800_000  # 2024-11-15 00:00 UTC, epoch ms
        bar.close = 100.0
        config = AnalysisConfig(
            tickers=["AAA"], start_date="2024-11-15", end_date="2024-11-15"
        )
        with patch("bestee_compute.workflow.tools.get_client") as client:
            client.return_value.get_grouped_daily_aggs.return_value = [bar]
            frame = tools.get_grouped_daily_column(config, ["AAA"], CLOSE)
        # Must match get_ohlc's declared Timestamp dtype so the two join.
        assert frame.schema[TS] == pl.Datetime("ms", time_zone="UTC")
        assert frame.height == 1
