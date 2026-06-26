"""Tests for bestee_compute.workflow.liquidity — Fed Net Liquidity pipeline.

The HTTP layer is mocked, so these run offline and assert the analytical
core: unit scaling, the net-liquidity equation, publication-lag shifting
(no look-ahead), weekend retention, and the rate-of-change feature.
"""

import math
from unittest.mock import MagicMock, patch

import httpx
import polars as pl
import pytest

from bestee_compute.workflow.liquidity import (
    FederalNetLiquidityPipeline,
    LiquidityComponent,
)

_HTTPX_PATCH = "bestee_compute.workflow.liquidity.httpx.Client"

# Synthetic FRED observations. WALCL/WTREGEN are Wednesday levels in
# millions; RRPONTSYD is daily in billions (so values are ×1000 in the
# equation). A "." encodes a FRED holiday gap that must be dropped.
_OBSERVATIONS: dict[str, list[dict[str, str]]] = {
    "WALCL": [
        {"date": "2024-01-03", "value": "7000000"},  # avail Fri 01-05
        {"date": "2024-01-10", "value": "7100000"},  # avail Fri 01-12
    ],
    "WTREGEN": [
        {"date": "2024-01-03", "value": "700000"},  # avail Fri 01-05
        {"date": "2024-01-10", "value": "750000"},  # avail Fri 01-12
    ],
    "RRPONTSYD": [
        {"date": "2024-01-03", "value": "600"},  # avail 01-04
        {"date": "2024-01-04", "value": "595"},  # avail 01-05
        {"date": "2024-01-05", "value": "590"},  # avail Sat 01-06
        {"date": "2024-01-08", "value": "580"},  # avail 01-09
        {"date": "2024-01-09", "value": "585"},  # avail 01-10
        {"date": "2024-01-10", "value": "575"},  # avail 01-11
        {"date": "2024-01-11", "value": "."},  # holiday gap -> dropped
        {"date": "2024-01-12", "value": "570"},  # avail Sat 01-13
    ],
}


def _response(observations: list[dict[str, str]]) -> MagicMock:
    response = MagicMock(spec=httpx.Response)
    response.raise_for_status = MagicMock()
    response.json = MagicMock(return_value={"observations": observations})
    return response


def _mock_client() -> MagicMock:
    """A client whose .get() returns the payload for the requested series."""
    client = MagicMock()

    def _get(url: str, params: dict[str, str] | None = None) -> MagicMock:
        assert params is not None
        return _response(_OBSERVATIONS[params["series_id"]])

    client.get.side_effect = _get
    return client


def _run(*, roc_window: int = 1, log_returns: bool = True) -> pl.DataFrame:
    pipe = FederalNetLiquidityPipeline(
        api_key="test-key", roc_window=roc_window, log_returns=log_returns
    )
    with patch(_HTTPX_PATCH) as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value = _mock_client()
        return pipe.generate_liquidity_anchor()


# ── Construction / validation ────────────────────────────────────────


class TestConstruction:
    def test_missing_required_component_raises(self) -> None:
        with pytest.raises(ValueError, match="must define"):
            FederalNetLiquidityPipeline(
                api_key="k",
                components={"WALCL": LiquidityComponent("WALCL", 2)},
            )

    def test_non_positive_roc_raises(self) -> None:
        with pytest.raises(ValueError, match="roc_window"):
            FederalNetLiquidityPipeline(api_key="k", roc_window=0)

    @patch("bestee_compute.workflow.liquidity.find_dotenv", return_value="")
    @patch("bestee_compute.workflow.liquidity.load_dotenv")
    def test_missing_api_key_raises(
        self, _load: MagicMock, _find: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="FRED API key"):
            FederalNetLiquidityPipeline()


# ── Output shape & invariants ────────────────────────────────────────


class TestLiquidityAnchor:
    def test_schema_and_no_nulls(self) -> None:
        df = _run()
        assert df.columns == ["date", "net_liquidity", "liquidity_roc"]
        assert df.schema["net_liquidity"] == pl.Float64
        assert df.null_count().to_numpy().sum() == 0

    def test_only_business_days(self) -> None:
        df = _run()
        weekdays = df["date"].dt.weekday().to_list()
        assert all(1 <= wd <= 5 for wd in weekdays)  # Mon..Fri

    def test_dates_sorted_and_unique(self) -> None:
        df = _run()
        assert df["date"].is_sorted()
        assert df["date"].n_unique() == df.height

    def test_net_liquidity_equation_with_unit_scaling(self) -> None:
        """WALCL − TGA − RRP×1000, using the most recent published values."""
        df = _run()
        row = df.filter(pl.col("date") == pl.date(2024, 1, 9)).row(0, named=True)
        # WALCL/TGA from Wed 01-03 (avail Fri 01-05); RRP from Mon 01-08=580B.
        assert row["net_liquidity"] == pytest.approx(7_000_000 - 700_000 - 580_000)

    def test_weekend_value_is_retained(self) -> None:
        """A Friday RRP reading shifts to Saturday; Monday must still see it.

        The original fixed-shift + business-day reindex silently dropped
        such weekend-stamped observations.
        """
        df = _run()
        row = df.filter(pl.col("date") == pl.date(2024, 1, 8)).row(0, named=True)
        # RRP from Fri 01-05 (=590B, avail Sat 01-06) carried into Mon 01-08.
        assert row["net_liquidity"] == pytest.approx(7_000_000 - 700_000 - 590_000)

    def test_no_lookahead_before_publication(self) -> None:
        """Thursday 01-11 cannot yet see the Wed 01-10 balance sheet."""
        df = _run()
        row = df.filter(pl.col("date") == pl.date(2024, 1, 11)).row(0, named=True)
        # WALCL still the 01-03 vintage (7.0M), not 01-10 (7.1M, avail 01-12).
        assert row["net_liquidity"] == pytest.approx(7_000_000 - 700_000 - 575_000)

    def test_holiday_gap_dropped(self) -> None:
        """The '.' RRP value must not crash or leak a null into the output."""
        df = _run()
        assert df.height > 0
        assert df["net_liquidity"].null_count() == 0


# ── Rate-of-change feature ───────────────────────────────────────────


class TestRateOfChange:
    def test_log_return_is_default(self) -> None:
        df = _run(roc_window=1).sort("date")
        nl = df["net_liquidity"].to_list()
        roc = df["liquidity_roc"].to_list()
        assert roc[1] == pytest.approx(math.log(nl[1] / nl[0]))

    def test_simple_return_when_disabled(self) -> None:
        df = _run(roc_window=1, log_returns=False).sort("date")
        nl = df["net_liquidity"].to_list()
        roc = df["liquidity_roc"].to_list()
        assert roc[1] == pytest.approx(nl[1] / nl[0] - 1)

    def test_window_controls_lookback(self) -> None:
        """A 2-session window compares t to t−2, not t−1."""
        df = _run(roc_window=2).sort("date")
        nl = df["net_liquidity"].to_list()
        roc = df["liquidity_roc"].to_list()
        assert roc[2] == pytest.approx(math.log(nl[2] / nl[0]))
