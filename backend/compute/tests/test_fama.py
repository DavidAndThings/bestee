"""Tests for the Fama-French 6-factor variable builder.

The Ken French CSV parser is checked as a pure function (no network), and
``get_variables`` is exercised through an injected price panel with the factor
download patched out, so the excess-return arithmetic and column layout are
verified deterministically.
"""

import datetime as dt
import math
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from bestee_compute.stocks.models import OHLCHeader
from bestee_compute.workflow import fama
from bestee_compute.workflow.fama import (
    _FACTOR_MODELS,
    FAMA_FRENCH_6_FACTORS,
    FamaFrenchResult,
    FamaFrenchSpecification,
    _ordinary_least_squares,
    _parse_daily_factor_csv,
    _residual_significance,
)
from bestee_compute.workflow.tools import AnalysisConfig

TS = OHLCHeader.TIMESTAMP
CLOSE = OHLCHeader.CLOSE
_FF5 = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
_BETAS = {
    "Mkt-RF": 1.1,
    "SMB": 0.3,
    "HML": -0.4,
    "RMW": 0.2,
    "CMA": -0.1,
    "Mom": 0.15,
}


def _weekdays(n: int) -> list[dt.datetime]:
    out: list[dt.datetime] = []
    day = dt.datetime(2018, 1, 1, tzinfo=dt.UTC)
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day)
        day += dt.timedelta(days=1)
    return out


def _factor_frame(*, n: int = 600, seed: int = 0, alpha: float = 3e-4) -> pl.DataFrame:
    """A per-ticker frame whose ExcessReturn follows a known FF6 process."""
    rng = np.random.default_rng(seed)
    factor_columns = {f: rng.normal(0.0, 0.01, n) for f in FAMA_FRENCH_6_FACTORS}
    excess = (
        alpha
        + sum(_BETAS[f] * factor_columns[f] for f in FAMA_FRENCH_6_FACTORS)
        + rng.normal(0.0, 0.002, n)
    )
    return pl.DataFrame({TS: _weekdays(n), "ExcessReturn": excess, **factor_columns})


def _config(tickers: list[str]) -> AnalysisConfig:
    return AnalysisConfig(
        tickers=tickers,
        start_date="2018-01-01",
        end_date="2030-01-01",
        ohlc_column=CLOSE,
    )


def _spec(
    start: str = "2018-01-01", end: str = "2030-01-01", factors: int = 6
) -> FamaFrenchSpecification:
    return FamaFrenchSpecification(
        start_date=start, end_date=end, factors_to_use=factors
    )


class TestParseDailyFactorCsv:
    def test_parses_data_rows_scales_and_nulls_missing(self) -> None:
        text = "\n".join(
            [
                "This file was created by using the CRSP database.",
                "Some prose preamble.",
                "",
                ",Mkt-RF,SMB,HML,RMW,CMA,RF",
                "20200102,    1.00,   -0.50,    0.25,    0.10,   -0.20,    0.010",
                "20200103,  -99.99,    0.30,    0.15,    0.05,    0.00,    0.010",
                "",
                "Copyright 2024 Eugene F. Fama and Kenneth R. French",
            ]
        )
        frame = _parse_daily_factor_csv(text, _FF5)

        assert frame.columns == [_c for _c in ["date", *_FF5]]
        assert frame.height == 2  # only the two YYYYMMDD rows
        first = frame.row(0, named=True)
        assert first["date"] == dt.date(2020, 1, 2)
        assert first["Mkt-RF"] == pytest.approx(0.01)  # 1.00% / 100
        assert first["RF"] == pytest.approx(0.0001)  # 0.010% / 100
        assert first["SMB"] == pytest.approx(-0.005)
        # -99.99 sentinel becomes null (not -0.9999).
        assert frame["Mkt-RF"][1] is None

    def test_parses_single_momentum_column(self) -> None:
        text = "\n".join(
            [
                "momentum preamble",
                "Missing data are indicated by -99.99 or -999.",
                ",Mom",
                "20200102,   0.56",
                "20200103,  -1.20",
                "Copyright",
            ]
        )
        frame = _parse_daily_factor_csv(text, ["Mom"])
        assert frame.columns == ["date", "Mom"]
        assert frame.height == 2
        assert frame["Mom"].to_list() == pytest.approx([0.0056, -0.012])

    def test_ignores_rows_with_wrong_column_count(self) -> None:
        text = "\n".join(
            [
                ",Mkt-RF,SMB,HML,RMW,CMA,RF",
                "20200102,1.0,2.0,3.0",  # too few values -> skipped
                "20200103,1.0,2.0,3.0,4.0,5.0,6.0",
            ]
        )
        frame = _parse_daily_factor_csv(text, _FF5)
        assert frame.height == 1
        assert frame["date"][0] == dt.date(2020, 1, 3)


class TestGetVariables:
    def _weekdays(self, n: int) -> list[dt.datetime]:
        out: list[dt.datetime] = []
        day = dt.datetime(2020, 1, 1, tzinfo=dt.UTC)
        while len(out) < n:
            if day.weekday() < 5:
                out.append(day)
            day += dt.timedelta(days=1)
        return out

    def test_builds_excess_returns_and_factor_columns(self) -> None:
        dates = self._weekdays(4)
        panel = pl.DataFrame(
            {
                TS: dates,
                "AAA_Close": [100.0, 110.0, 121.0, 121.0],  # +10%, +10%, 0%
                "BBB_Close": [50.0, 55.0, 55.0, 60.5],  # +10%, 0%, +10%
            }
        )
        config = AnalysisConfig(
            tickers=["AAA", "BBB"],
            start_date="2020-01-01",
            end_date="2020-12-31",
            injected_panel=panel,
            ohlc_column=CLOSE,
        )
        factors = pl.DataFrame(
            {
                "date": [d.date() for d in dates],
                "Mkt-RF": [0.01, 0.02, 0.03, 0.04],
                "SMB": [0.001, 0.002, 0.003, 0.004],
                "HML": [0.0, 0.0, 0.0, 0.0],
                "RMW": [0.0, 0.0, 0.0, 0.0],
                "CMA": [0.0, 0.0, 0.0, 0.0],
                "RF": [0.001, 0.001, 0.001, 0.001],
                "Mom": [0.005, 0.006, 0.007, 0.008],
            }
        )
        with patch(
            "bestee_compute.workflow.fama.fetch_fama_french_factors",
            return_value=factors,
        ):
            variables = fama.get_variables(config)

        # One frame per ticker.
        assert set(variables) == {"AAA", "BBB"}
        for frame in variables.values():
            assert frame.columns == [
                TS,
                "ExcessReturn",
                "Mkt-RF",
                "SMB",
                "HML",
                "RMW",
                "CMA",
                "Mom",
            ]
            # First bar (null pct_change) dropped -> 3 rows remain.
            assert frame.height == 3
            # RF is netted out, not exposed as a column.
            assert "RF" not in frame.columns
        # Excess return = simple return - RF; AAA went +10% on days 2 and 3.
        assert variables["AAA"]["ExcessReturn"][0] == pytest.approx(0.10 - 0.001)
        # AAA was flat (0%) on day 4: excess = -RF.
        assert variables["AAA"]["ExcessReturn"][2] == pytest.approx(-0.001)
        # BBB went +10% on day 4 (index 2 after the warm-up drop).
        assert variables["BBB"]["ExcessReturn"][2] == pytest.approx(0.10 - 0.001)
        # BBB was flat on day 3.
        assert variables["BBB"]["ExcessReturn"][1] == pytest.approx(-0.001)

    def test_inner_join_drops_dates_without_factors(self) -> None:
        dates = self._weekdays(3)
        panel = pl.DataFrame({TS: dates, "AAA_Close": [100.0, 101.0, 102.0]})
        config = AnalysisConfig(
            tickers=["AAA"],
            start_date="2020-01-01",
            end_date="2020-12-31",
            injected_panel=panel,
            ohlc_column=CLOSE,
        )
        # Factors cover only the final date -> only that row can survive.
        factors = pl.DataFrame(
            {
                "date": [dates[-1].date()],
                "Mkt-RF": [0.01],
                "SMB": [0.0],
                "HML": [0.0],
                "RMW": [0.0],
                "CMA": [0.0],
                "RF": [0.001],
                "Mom": [0.0],
            }
        )
        with patch(
            "bestee_compute.workflow.fama.fetch_fama_french_factors",
            return_value=factors,
        ):
            variables = fama.get_variables(config)
        assert set(variables) == {"AAA"}
        frame = variables["AAA"]
        assert frame.height == 1
        assert frame[TS][0] == dates[-1]


class TestOrdinaryLeastSquares:
    def test_recovers_a_known_line(self) -> None:
        rng = np.random.default_rng(0)
        n = 1000
        factors = rng.normal(0.0, 1.0, (n, 2))
        truth = np.array([0.5, 2.0, -1.0])  # alpha, b1, b2
        response = truth[0] + factors @ truth[1:] + rng.normal(0.0, 0.1, n)
        fit = _ordinary_least_squares(response, factors)
        assert fit.coef == pytest.approx(truth, abs=0.02)
        assert fit.r_squared > 0.99
        assert fit.t_stats[1] > 20  # strong, true beta


class TestResidualSignificance:
    def test_two_sigma_residual(self) -> None:
        z, p = _residual_significance(0.02, 0.01)  # 2 sigma move
        assert z == pytest.approx(2.0)
        assert p == pytest.approx(math.erfc(2.0 / math.sqrt(2)))  # ~0.0455

    def test_zero_residual_is_not_significant(self) -> None:
        z, p = _residual_significance(0.0, 0.01)
        assert z == 0.0
        assert p == pytest.approx(1.0)

    def test_sign_does_not_change_two_sided_pvalue(self) -> None:
        _, p_up = _residual_significance(0.015, 0.01)
        _, p_down = _residual_significance(-0.015, 0.01)
        assert p_up == pytest.approx(p_down)

    def test_non_positive_sigma_yields_nan(self) -> None:
        z, p = _residual_significance(0.01, 0.0)
        assert math.isnan(z) and math.isnan(p)


class TestFitModel:
    def test_recovers_six_factor_coefficients(self) -> None:
        frame = _factor_frame(n=600, seed=1, alpha=3e-4)
        config = _config(["AAA"])
        with patch(
            "bestee_compute.workflow.fama.get_variables", return_value={"AAA": frame}
        ):
            results = fama.fit_model(config, _spec(factors=6))

        assert set(results) == {"AAA"}
        result = results["AAA"]
        assert isinstance(result, FamaFrenchResult)
        assert result.factors == FAMA_FRENCH_6_FACTORS
        assert result.n_observations == 600
        assert result.alpha == pytest.approx(3e-4, abs=5e-4)
        for factor, beta in _BETAS.items():
            assert result.betas[factor] == pytest.approx(beta, abs=0.05)
        assert result.r_squared > 0.9
        assert set(result.t_stats) == {"alpha", *FAMA_FRENCH_6_FACTORS}
        # True betas are large relative to noise -> highly significant.
        assert abs(result.t_stats["Mom"]) > 3

    @pytest.mark.parametrize("factors_to_use", [3, 4, 5, 6])
    def test_factor_selection(self, factors_to_use: int) -> None:
        frame = _factor_frame(n=300, seed=2)
        config = _config(["AAA"])
        with patch(
            "bestee_compute.workflow.fama.get_variables", return_value={"AAA": frame}
        ):
            result = fama.fit_model(config, _spec(factors=factors_to_use))["AAA"]
        expected = _FACTOR_MODELS[factors_to_use]
        assert result.factors == expected
        assert set(result.betas) == set(expected)
        assert set(result.t_stats) == {"alpha", *expected}
        assert set(result.std_errors) == {"alpha", *expected}

    def test_rejects_invalid_factor_count(self) -> None:
        config = _config(["AAA"])
        with patch("bestee_compute.workflow.fama.get_variables", return_value={}):
            with pytest.raises(ValueError, match="factors_to_use"):
                fama.fit_model(config, _spec(factors=7))

    def test_filters_to_the_requested_window(self) -> None:
        frame = _factor_frame(n=400, seed=3)
        dates = frame[TS].to_list()
        config = _config(["AAA"])
        with patch(
            "bestee_compute.workflow.fama.get_variables", return_value={"AAA": frame}
        ):
            result = fama.fit_model(
                config,
                _spec(
                    start=dates[0].date().isoformat(),
                    end=dates[99].date().isoformat(),
                    factors=3,
                ),
            )["AAA"]
        assert result.n_observations == 100  # inclusive window
        assert result.start_date == dates[0].date()
        assert result.end_date == dates[99].date()

    def test_skips_tickers_with_too_few_observations(self) -> None:
        frame = _factor_frame(n=400, seed=4)
        dates = frame[TS].to_list()
        config = _config(["AAA"])
        with patch(
            "bestee_compute.workflow.fama.get_variables", return_value={"AAA": frame}
        ):
            # A 4-day window can't support a 6-factor (7-parameter) fit.
            results = fama.fit_model(
                config,
                _spec(
                    start=dates[0].date().isoformat(),
                    end=dates[3].date().isoformat(),
                    factors=6,
                ),
            )
        assert results == {}

    def test_summary_frame(self) -> None:
        frame = _factor_frame(n=300, seed=5)
        config = _config(["AAA"])
        with patch(
            "bestee_compute.workflow.fama.get_variables", return_value={"AAA": frame}
        ):
            result = fama.fit_model(config, _spec(factors=4))["AAA"]
        summary = result.summary()
        assert summary.columns == ["Parameter", "Coefficient", "StdError", "tStat"]
        assert summary["Parameter"].to_list() == ["alpha", *_FACTOR_MODELS[4]]
        assert summary.height == 5


class TestGetResiduals:
    def test_out_of_sample_residual_matches_actual_minus_prediction(self) -> None:
        frame = _factor_frame(n=400, seed=1)
        dates = frame[TS].to_list()
        config = _config(["AAA"])
        spec = _spec(
            start=dates[0].date().isoformat(),
            end=dates[299].date().isoformat(),
            factors=6,
        )
        target = dates[350]  # after the [0, 299] window
        with patch(
            "bestee_compute.workflow.fama.get_variables", return_value={"AAA": frame}
        ):
            model = fama.fit_model(config, spec)["AAA"]
            residuals = fama.get_residuals(config, target.date().isoformat(), [spec])

        assert residuals.columns == [
            "Ticker",
            "Specification",
            "Residual",
            "ZScore",
            "PValue",
        ]
        assert residuals["Ticker"].to_list() == ["AAA"]
        assert residuals["Specification"].to_list() == [spec.name]
        # residual = actual excess - model prediction on the target date.
        row = frame.filter(pl.col(TS).dt.date() == target.date())
        predicted = model.alpha + sum(model.betas[f] * row[f][0] for f in model.factors)
        expected = row["ExcessReturn"][0] - predicted
        assert residuals["Residual"][0] == pytest.approx(expected)
        # z standardizes by this spec's own residual std; p is its Normal tail.
        z = residuals["ZScore"][0]
        assert z == pytest.approx(residuals["Residual"][0] / model.residual_std)
        assert residuals["PValue"][0] == pytest.approx(math.erfc(abs(z) / math.sqrt(2)))

    def test_requires_a_date_after_the_model_window(self) -> None:
        frame = _factor_frame(n=400, seed=2)
        dates = frame[TS].to_list()
        config = _config(["AAA"])
        spec = _spec(
            start=dates[0].date().isoformat(),
            end=dates[299].date().isoformat(),
            factors=3,
        )
        in_sample = dates[100].date().isoformat()  # inside the window
        with patch(
            "bestee_compute.workflow.fama.get_variables", return_value={"AAA": frame}
        ):
            with pytest.raises(ValueError, match="out-of-sample"):
                fama.get_residuals(config, in_sample, [spec])

    def test_raises_when_no_observation_on_date(self) -> None:
        frame = _factor_frame(n=300, seed=3)
        dates = frame[TS].to_list()
        config = _config(["AAA"])
        spec = _spec(
            start=dates[0].date().isoformat(),
            end=dates[200].date().isoformat(),
            factors=3,
        )
        # After the window AND past the data -> no row for any ticker.
        future = (dates[-1] + dt.timedelta(days=30)).date().isoformat()
        with patch(
            "bestee_compute.workflow.fama.get_variables", return_value={"AAA": frame}
        ):
            with pytest.raises(ValueError, match="No ticker had an observation"):
                fama.get_residuals(config, future, [spec])

    def test_one_row_per_ticker(self) -> None:
        frame_a = _factor_frame(n=400, seed=4)
        frame_b = _factor_frame(n=400, seed=5)
        dates = frame_a[TS].to_list()
        config = _config(["AAA", "BBB"])
        spec = _spec(
            start=dates[0].date().isoformat(),
            end=dates[299].date().isoformat(),
            factors=6,
        )
        target = dates[350].date().isoformat()
        with patch(
            "bestee_compute.workflow.fama.get_variables",
            return_value={"AAA": frame_a, "BBB": frame_b},
        ):
            residuals = fama.get_residuals(config, target, [spec])
        assert residuals.height == 2
        assert set(residuals["Ticker"].to_list()) == {"AAA", "BBB"}

    def test_multiple_specifications_one_row_each(self) -> None:
        frame = _factor_frame(n=400, seed=6)
        dates = frame[TS].to_list()
        config = _config(["AAA"])
        ff3 = _spec(
            start=dates[0].date().isoformat(),
            end=dates[299].date().isoformat(),
            factors=3,
        )
        ff6 = _spec(
            start=dates[0].date().isoformat(),
            end=dates[299].date().isoformat(),
            factors=6,
        )
        target = dates[350].date().isoformat()
        with patch(
            "bestee_compute.workflow.fama.get_variables", return_value={"AAA": frame}
        ):
            residuals = fama.get_residuals(config, target, [ff3, ff6])
        # One (ticker, spec) row per specification, labelled by spec name.
        assert residuals.height == 2
        assert residuals["Specification"].to_list() == [ff3.name, ff6.name]
        assert residuals["Ticker"].to_list() == ["AAA", "AAA"]
        # Different models -> different residuals on the same date.
        assert residuals["Residual"][0] != residuals["Residual"][1]

    def test_pvalue_is_local_to_each_specification(self) -> None:
        frame = _factor_frame(n=400, seed=8)
        dates = frame[TS].to_list()
        config = _config(["AAA"])
        ff3 = _spec(
            start=dates[0].date().isoformat(),
            end=dates[299].date().isoformat(),
            factors=3,
        )
        ff6 = _spec(
            start=dates[0].date().isoformat(),
            end=dates[299].date().isoformat(),
            factors=6,
        )
        target = dates[350].date().isoformat()
        with patch(
            "bestee_compute.workflow.fama.get_variables", return_value={"AAA": frame}
        ):
            m3 = fama.fit_model(config, ff3)["AAA"]
            m6 = fama.fit_model(config, ff6)["AAA"]
            residuals = fama.get_residuals(config, target, [ff3, ff6])
        # Each row's z-score is standardized by *its own* spec's residual std.
        for spec, model in ((ff3, m3), (ff6, m6)):
            row = residuals.filter(pl.col("Specification") == spec.name)
            z = row["ZScore"][0]
            assert z == pytest.approx(row["Residual"][0] / model.residual_std)
            assert row["PValue"][0] == pytest.approx(math.erfc(abs(z) / math.sqrt(2)))
        # The two specs use different sigmas -> different standardizations.
        z3 = residuals.filter(pl.col("Specification") == ff3.name)["ZScore"][0]
        z6 = residuals.filter(pl.col("Specification") == ff6.name)["ZScore"][0]
        assert z3 != z6

    def test_rejects_date_before_any_spec_window(self) -> None:
        frame = _factor_frame(n=400, seed=7)
        dates = frame[TS].to_list()
        config = _config(["AAA"])
        early = _spec(
            start=dates[0].date().isoformat(),
            end=dates[299].date().isoformat(),
            factors=3,
        )
        # This window ends *after* the target -> the whole call must reject.
        late = _spec(
            start=dates[0].date().isoformat(),
            end=dates[360].date().isoformat(),
            factors=6,
        )
        target = dates[350].date().isoformat()
        with patch(
            "bestee_compute.workflow.fama.get_variables", return_value={"AAA": frame}
        ):
            with pytest.raises(ValueError, match="out-of-sample"):
                fama.get_residuals(config, target, [early, late])


class TestSpecificationVerifyOutOfSample:
    def test_allows_a_later_date(self) -> None:
        spec = _spec(end="2023-12-29")
        spec.verify_out_of_sample(dt.date(2024, 1, 2))  # must not raise

    def test_rejects_an_in_window_or_boundary_date(self) -> None:
        spec = _spec(end="2023-12-29")
        with pytest.raises(ValueError, match="out-of-sample"):
            spec.verify_out_of_sample(dt.date(2023, 12, 29))  # boundary (==end)
        with pytest.raises(ValueError, match="out-of-sample"):
            spec.verify_out_of_sample(dt.date(2023, 6, 1))  # inside window
