"""Tests for bestee_compute.securities.options — vol-skew computation.

The Massive client (and the FRED rate) are mocked, so these run offline
and assert: snapshot selection (nearest + interpolated), smile and term
interpolation, liquidity filtering, the historical Black–Scholes path,
input validation, and the batch frame / Great Tables output.
"""

import datetime
from contextlib import contextmanager
from typing import cast
from unittest.mock import patch

import pytest
from great_tables import GT
from massive import RESTClient
from massive.rest.models.aggs import Agg
from massive.rest.models.contracts import OptionsContract
from massive.rest.models.snapshot import OptionContractSnapshot

from bestee_compute.securities.black_scholes import bs_price
from bestee_compute.securities.options import (
    _interp_iv_in_delta,
    _nearest_active_expiration,
    _Quote,
    _term_interp_iv,
    compute_normalized_volatility_skew_result,
    compute_volatility_skew_frame,
    volatility_skew_table,
)

_PATCH_CLIENT = "bestee_compute.securities.options.get_client"
_PATCH_RATE = "bestee_compute.securities.options.get_risk_free_rate"

TODAY = datetime.date.today()
SNAP_EXP = TODAY + datetime.timedelta(days=45)


# ── Builders for SDK model instances (via from_dict — modelclass) ────


def _snap(
    contract_type: str,
    delta: float | None,
    iv: float | None,
    *,
    ticker: str,
    expiration: datetime.date = SNAP_EXP,
    strike: float = 100.0,
    open_interest: float | None = None,
    bid: float | None = None,
    ask: float | None = None,
    underlying_price: float | None = None,
) -> OptionContractSnapshot:
    payload: dict[str, object] = {
        "details": {
            "contract_type": contract_type,
            "ticker": ticker,
            "strike_price": strike,
            "expiration_date": expiration.isoformat(),
        },
        "implied_volatility": iv,
    }
    if delta is not None:
        payload["greeks"] = {"delta": delta}
    if open_interest is not None:
        payload["open_interest"] = open_interest
    if bid is not None or ask is not None:
        payload["last_quote"] = {"bid": bid, "ask": ask}
    if underlying_price is not None:
        payload["underlying_asset"] = {"price": underlying_price}
    return OptionContractSnapshot.from_dict(payload)


def _contract(
    expiration: datetime.date,
    *,
    ticker: str | None = None,
    contract_type: str | None = None,
    strike: float | None = None,
) -> OptionsContract:
    return OptionsContract.from_dict(
        {
            "expiration_date": expiration.isoformat(),
            "ticker": ticker,
            "contract_type": contract_type,
            "strike_price": strike,
        }
    )


def _agg(close: float, volume: float) -> Agg:
    return Agg.from_dict({"c": close, "v": volume})


# Standard one-expiration snapshot smile.
_CHAIN: list[OptionContractSnapshot] = [
    _snap("call", 0.20, 0.30, ticker="C20", underlying_price=100.0),
    _snap("call", 0.50, 0.25, ticker="C50"),
    _snap("call", 0.80, 0.28, ticker="C80"),
    _snap("put", -0.20, 0.34, ticker="P20"),
    _snap("put", -0.50, 0.26, ticker="P50"),
    _snap("put", -0.80, 0.40, ticker="P80"),
]


class _FakeClient:
    """Stand-in for massive.RESTClient covering the calls options.py makes."""

    def __init__(
        self,
        *,
        available_expirations: list[datetime.date],
        chains: dict[datetime.date, list[OptionContractSnapshot]] | None = None,
        contracts_by_exp: dict[datetime.date, list[OptionsContract]] | None = None,
        bars: dict[tuple[str, datetime.date], tuple[float, float]] | None = None,
        empty_for: set[str] | None = None,
    ) -> None:
        self.available_expirations = sorted(available_expirations)
        self.chains = chains or {}
        self.contracts_by_exp = contracts_by_exp or {}
        self.bars = bars or {}
        self.empty_for = empty_for or set()

    def list_options_contracts(self, **kwargs: object) -> list[OptionsContract]:
        exp_exact = kwargs.get("expiration_date")
        if exp_exact is not None:  # historical surface listing
            exp = cast(datetime.date, exp_exact)
            lo = cast("float | None", kwargs.get("strike_price_gte"))
            hi = cast("float | None", kwargs.get("strike_price_lte"))
            out: list[OptionsContract] = []
            for contract in self.contracts_by_exp.get(exp, []):
                strike = contract.strike_price
                if strike is not None and lo is not None and strike < lo:
                    continue
                if strike is not None and hi is not None and strike > hi:
                    continue
                out.append(contract)
            return out
        gte = cast("datetime.date | None", kwargs.get("expiration_date_gte"))
        lte = cast("datetime.date | None", kwargs.get("expiration_date_lte"))
        chosen: datetime.date | None = None
        if gte is not None:
            forward = [e for e in self.available_expirations if e >= gte]
            chosen = forward[0] if forward else None
        elif lte is not None:
            backward = [e for e in self.available_expirations if e <= lte]
            chosen = backward[-1] if backward else None
        return [] if chosen is None else [_contract(chosen)]

    def list_snapshot_options_chain(
        self, underlying: str, params: dict[str, object] | None = None
    ) -> list[OptionContractSnapshot]:
        if underlying in self.empty_for or params is None:
            return []
        exp = datetime.date.fromisoformat(cast(str, params["expiration_date"]))
        return list(self.chains.get(exp, []))

    def get_aggs(
        self,
        ticker: str,
        multiplier: int,
        timespan: str,
        from_: datetime.date,
        to: datetime.date,
        **_: object,
    ) -> list[Agg]:
        bar = self.bars.get((ticker, from_))
        return [] if bar is None else [_agg(bar[0], bar[1])]


@contextmanager
def _patched(client: _FakeClient):
    with patch(_PATCH_CLIENT, return_value=client):
        yield


def _snapshot_fake(chain: list[OptionContractSnapshot]) -> _FakeClient:
    return _FakeClient(available_expirations=[SNAP_EXP], chains={SNAP_EXP: chain})


# ── Input validation ─────────────────────────────────────────────────


class TestValidation:
    def test_call_delta_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="call_delta must be positive"):
            compute_normalized_volatility_skew_result(
                "AAPL", TODAY, 45, -0.25, -0.25, 0.50
            )

    def test_put_delta_must_be_negative(self) -> None:
        with pytest.raises(ValueError, match="put_delta must be negative"):
            compute_normalized_volatility_skew_result(
                "AAPL", TODAY, 45, 0.25, 0.25, 0.50
            )

    def test_lookahead_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="lookahead_days"):
            compute_normalized_volatility_skew_result(
                "AAPL", TODAY, 0, -0.25, 0.25, 0.50
            )


# ── Pure interpolation helpers ───────────────────────────────────────


class TestInterpolationHelpers:
    def test_interp_iv_in_delta(self) -> None:
        pool = [
            _Quote("call", "a", 0.20, 0.30),
            _Quote("call", "b", 0.50, 0.25),
        ]
        # 0.25 sits 1/6 of the way from 0.20 to 0.50.
        assert _interp_iv_in_delta(pool, 0.25) == pytest.approx(
            0.30 - (0.05 / 0.30) * 0.05
        )

    def test_interp_iv_clamps_outside_range(self) -> None:
        pool = [_Quote("call", "a", 0.20, 0.30), _Quote("call", "b", 0.50, 0.25)]
        assert _interp_iv_in_delta(pool, 0.05) == 0.30  # below range -> first
        assert _interp_iv_in_delta(pool, 0.95) == 0.25  # above range -> last

    def test_term_interp_in_total_variance(self) -> None:
        # w_lo=0.04*0.1, w_hi=0.16*0.2, t*=0.15 halfway -> w=0.018 -> sqrt(0.12).
        iv = _term_interp_iv(0.20, 0.1, 0.40, 0.2, 0.15)
        assert iv == pytest.approx((0.12) ** 0.5)


# ── Snapshot mode ────────────────────────────────────────────────────


class TestSnapshotNearest:
    def test_selects_nearest_delta_contracts(self) -> None:
        with _patched(_snapshot_fake(_CHAIN)):
            r = compute_normalized_volatility_skew_result(
                "AAPL",
                TODAY,
                45,
                -0.25,
                0.25,
                0.50,
                mode="snapshot",
                interpolate=False,
            )
        # call->C20(.30), put->P20(.34), denom call->C50(.25).
        assert r.skew == pytest.approx((0.30 - 0.34) / 0.25)
        assert r.interpolated is False
        assert r.call.ticker == "C20"  # concrete contract
        assert r.mode == "snapshot"
        assert r.underlying_price == 100.0

    def test_zero_denominator_iv_raises(self) -> None:
        chain = [
            _snap("call", 0.25, 0.30, ticker="C25"),
            _snap("call", 0.50, 0.0, ticker="C50Z"),
            _snap("put", -0.25, 0.34, ticker="P25"),
        ]
        with _patched(_snapshot_fake(chain)), pytest.raises(ValueError, match="zero"):
            compute_normalized_volatility_skew_result(
                "AAPL",
                TODAY,
                45,
                -0.25,
                0.25,
                0.50,
                mode="snapshot",
                interpolate=False,
            )

    def test_no_usable_data_raises(self) -> None:
        with _patched(_snapshot_fake([])), pytest.raises(ValueError, match="No usable"):
            compute_normalized_volatility_skew_result(
                "AAPL", TODAY, 45, -0.25, 0.25, 0.50, mode="snapshot"
            )


class TestSnapshotInterpolated:
    def test_smile_interpolation_single_expiration(self) -> None:
        with _patched(_snapshot_fake(_CHAIN)):
            r = compute_normalized_volatility_skew_result(
                "AAPL",
                TODAY,
                45,
                -0.25,
                0.25,
                0.50,
                mode="snapshot",
                interpolate=True,
            )
        call_iv = 0.30 - (0.05 / 0.30) * 0.05  # 0.291667
        put_iv = 0.26 + (0.25 / 0.30) * 0.08  # 0.326667
        denom_iv = 0.25  # interp at delta 0.50 endpoint
        assert r.skew == pytest.approx((call_iv - put_iv) / denom_iv)
        assert r.interpolated is True
        assert r.call.ticker is None  # synthetic
        assert r.call.delta == pytest.approx(0.25)

    def test_term_structure_interpolation_two_expirations(self) -> None:
        exp_lo = TODAY + datetime.timedelta(days=30)
        exp_hi = TODAY + datetime.timedelta(days=60)

        def flat(exp: datetime.date, call_iv: float, put_iv: float):
            return [
                _snap("call", 0.20, call_iv, ticker="cL", expiration=exp),
                _snap("call", 0.80, call_iv, ticker="cH", expiration=exp),
                _snap("put", -0.80, put_iv, ticker="pL", expiration=exp),
                _snap("put", -0.20, put_iv, ticker="pH", expiration=exp),
            ]

        fake = _FakeClient(
            available_expirations=[exp_lo, exp_hi],
            chains={exp_lo: flat(exp_lo, 0.20, 0.24), exp_hi: flat(exp_hi, 0.30, 0.34)},
        )
        with _patched(fake):
            r = compute_normalized_volatility_skew_result(
                "AAPL",
                TODAY,
                45,
                -0.25,
                0.25,
                0.50,
                mode="snapshot",
                interpolate=True,
            )
        assert r.interpolated is True
        assert r.expiration == TODAY + datetime.timedelta(days=45)  # synthetic target
        assert 0.20 < r.call.implied_volatility < 0.30  # between the two expiries
        assert 0.24 < r.put.implied_volatility < 0.34
        assert r.skew < 0  # puts richer than calls at both ends


# ── Liquidity filtering ──────────────────────────────────────────────


class TestLiquidityFilter:
    def test_min_open_interest_drops_thin_contracts(self) -> None:
        chain = [
            _snap("call", 0.25, 0.50, ticker="C25_thin", open_interest=10),
            _snap("call", 0.20, 0.30, ticker="C20", open_interest=500),
            _snap("call", 0.50, 0.25, ticker="C50", open_interest=500),
            _snap("put", -0.25, 0.34, ticker="P25", open_interest=500),
        ]
        with _patched(_snapshot_fake(chain)):
            r = compute_normalized_volatility_skew_result(
                "AAPL",
                TODAY,
                45,
                -0.25,
                0.25,
                0.50,
                mode="snapshot",
                interpolate=False,
                min_open_interest=100,
            )
        # Thin 0.25 call dropped -> nearest call is C20 (IV .30), not .50.
        assert r.call.ticker == "C20"
        assert r.skew == pytest.approx((0.30 - 0.34) / 0.25)

    def test_max_relative_spread_drops_wide_contracts(self) -> None:
        chain = [
            _snap("call", 0.25, 0.50, ticker="C25_wide", bid=1.0, ask=2.0),
            _snap("call", 0.20, 0.30, ticker="C20", bid=1.00, ask=1.02),
            _snap("call", 0.50, 0.25, ticker="C50", bid=1.00, ask=1.02),
            _snap("put", -0.25, 0.34, ticker="P25", bid=1.00, ask=1.02),
        ]
        with _patched(_snapshot_fake(chain)):
            r = compute_normalized_volatility_skew_result(
                "AAPL",
                TODAY,
                45,
                -0.25,
                0.25,
                0.50,
                mode="snapshot",
                interpolate=False,
                max_relative_spread=0.1,
            )
        assert r.call.ticker == "C20"  # wide-spread 0.25 call excluded


# ── Historical (point-in-time Black–Scholes) mode ────────────────────


class TestHistorical:
    def _fake(self, quote_date: datetime.date, exp: datetime.date) -> _FakeClient:
        t = (exp - quote_date).days / 365.0
        spot, rate = 100.0, 0.05
        call_price = bs_price(spot, 100.0, t, rate, 0.0, 0.25, "call")
        put_price = bs_price(spot, 100.0, t, rate, 0.0, 0.30, "put")
        return _FakeClient(
            available_expirations=[exp],
            contracts_by_exp={
                exp: [
                    _contract(exp, ticker="O:C", contract_type="call", strike=100.0),
                    _contract(exp, ticker="O:P", contract_type="put", strike=100.0),
                ]
            },
            bars={
                ("AAPL", quote_date): (spot, 1_000_000.0),
                ("O:C", quote_date): (call_price, 50.0),
                ("O:P", quote_date): (put_price, 50.0),
            },
        )

    def test_reconstructs_iv_via_black_scholes(self) -> None:
        quote_date = datetime.date(2024, 1, 2)
        exp = datetime.date(2024, 2, 16)  # 45 days
        with _patched(self._fake(quote_date, exp)):
            r = compute_normalized_volatility_skew_result(
                "AAPL",
                quote_date,
                45,
                -0.25,
                0.25,
                0.50,
                mode="historical",
                interpolate=False,
                risk_free_rate=0.05,
                dividend_yield=0.0,
            )
        assert r.mode == "historical"
        assert r.underlying_price == 100.0
        assert r.call.implied_volatility == pytest.approx(0.25, abs=1e-3)
        assert r.put.implied_volatility == pytest.approx(0.30, abs=1e-3)
        # denom is the (only) call -> same IV as the call leg.
        assert r.skew == pytest.approx((0.25 - 0.30) / 0.25, abs=1e-2)

    def test_fetches_rate_from_fred_when_unset(self) -> None:
        quote_date = datetime.date(2024, 1, 2)
        exp = datetime.date(2024, 2, 16)
        with (
            _patched(self._fake(quote_date, exp)),
            patch(_PATCH_RATE, return_value=0.05) as mock_rate,
        ):
            compute_normalized_volatility_skew_result(
                "AAPL",
                quote_date,
                45,
                -0.25,
                0.25,
                0.50,
                mode="historical",
                interpolate=False,
            )
        mock_rate.assert_called_once()


# ── Nearest-expiration resolution ────────────────────────────────────


class TestNearestExpiration:
    def test_picks_closer_of_forward_and_backward(self) -> None:
        client = _FakeClient(
            available_expirations=[
                datetime.date(2024, 3, 22),  # +7
                datetime.date(2024, 3, 14),  # -1
            ]
        )
        nearest = _nearest_active_expiration(
            cast(RESTClient, client), "AAPL", datetime.date(2024, 3, 15)
        )
        assert nearest == datetime.date(2024, 3, 14)

    def test_no_contracts_raises(self) -> None:
        client = _FakeClient(available_expirations=[])
        with pytest.raises(ValueError, match="No active option contracts"):
            _nearest_active_expiration(
                cast(RESTClient, client), "AAPL", datetime.date(2024, 3, 15)
            )


# ── Batch frame + Great Tables ───────────────────────────────────────


class TestBatchAndTable:
    def test_frame_has_row_per_success_and_skips_failures(self) -> None:
        fake = _FakeClient(
            available_expirations=[SNAP_EXP],
            chains={SNAP_EXP: _CHAIN},
            empty_for={"BAD"},  # no chain -> computation fails -> skipped
        )
        with _patched(fake):
            frame = compute_volatility_skew_frame(
                ["AAPL", "BAD", "SPY"],
                TODAY,
                45,
                -0.25,
                0.25,
                0.50,
                mode="snapshot",
                interpolate=False,
            )
        assert frame.height == 2  # BAD skipped
        assert set(frame["Ticker"].to_list()) == {"AAPL", "SPY"}
        assert frame["Skew"].to_list() == pytest.approx([(0.30 - 0.34) / 0.25] * 2)

    def test_table_renders(self) -> None:
        with _patched(_snapshot_fake(_CHAIN)):
            frame = compute_volatility_skew_frame(
                ["AAPL"],
                TODAY,
                45,
                -0.25,
                0.25,
                0.50,
                mode="snapshot",
                interpolate=False,
            )
        gt = volatility_skew_table(frame)
        assert isinstance(gt, GT)
        assert "Volatility Skew" in gt.as_raw_html()
