"""Tests for bestee_compute.securities.black_scholes.

Pure-math checks with no network or I/O: round-trip implied-volatility
recovery, put–call parity, delta bounds / signs / limits, no-arbitrage and
bad-input handling for implied volatility, and input validation for the
pricing helpers.
"""

import math

import pytest

from bestee_compute.securities.black_scholes import (
    OptionType,
    bs_delta,
    bs_price,
    implied_volatility,
)

# (spot, strike, t_years, rate, dividend_yield, sigma)
_CASES = [
    (100.0, 100.0, 1.0, 0.05, 0.00, 0.20),
    (100.0, 90.0, 0.5, 0.03, 0.01, 0.35),
    (100.0, 110.0, 2.0, 0.02, 0.04, 0.15),
    (50.0, 55.0, 0.25, 0.045, 0.02, 0.50),
    (200.0, 180.0, 1.5, 0.01, 0.03, 0.25),
]


# ── Known-value sanity checks ────────────────────────────────────────


class TestKnownValues:
    def test_call_textbook_value(self) -> None:
        # S=K=100, T=1, r=5%, q=0, sigma=20% -> ~10.4506 (textbook).
        price = bs_price(100.0, 100.0, 1.0, 0.05, 0.0, 0.20, "call")
        assert price == pytest.approx(10.4506, abs=1e-4)

    def test_put_textbook_value(self) -> None:
        price = bs_price(100.0, 100.0, 1.0, 0.05, 0.0, 0.20, "put")
        assert price == pytest.approx(5.5735, abs=1e-4)


# ── Round-trip: price -> implied volatility ──────────────────────────


class TestImpliedVolRoundTrip:
    @pytest.mark.parametrize("option_type", ["call", "put"])
    @pytest.mark.parametrize(("spot", "strike", "t", "r", "q", "sigma"), _CASES)
    def test_recovers_sigma(
        self,
        spot: float,
        strike: float,
        t: float,
        r: float,
        q: float,
        sigma: float,
        option_type: OptionType,
    ) -> None:
        price = bs_price(spot, strike, t, r, q, sigma, option_type)
        recovered = implied_volatility(price, spot, strike, t, r, q, option_type)
        assert recovered is not None
        assert recovered == pytest.approx(sigma, abs=1e-4)


# ── Put–call parity ──────────────────────────────────────────────────


class TestPutCallParity:
    @pytest.mark.parametrize(("spot", "strike", "t", "r", "q", "sigma"), _CASES)
    def test_parity_holds(
        self,
        spot: float,
        strike: float,
        t: float,
        r: float,
        q: float,
        sigma: float,
    ) -> None:
        call = bs_price(spot, strike, t, r, q, sigma, "call")
        put = bs_price(spot, strike, t, r, q, sigma, "put")
        expected = spot * math.exp(-q * t) - strike * math.exp(-r * t)
        assert call - put == pytest.approx(expected, abs=1e-9)


# ── Delta bounds, signs, and limits ──────────────────────────────────


class TestDelta:
    @pytest.mark.parametrize(("spot", "strike", "t", "r", "q", "sigma"), _CASES)
    def test_call_delta_in_unit_interval(
        self,
        spot: float,
        strike: float,
        t: float,
        r: float,
        q: float,
        sigma: float,
    ) -> None:
        delta = bs_delta(spot, strike, t, r, q, sigma, "call")
        assert 0.0 < delta < 1.0

    @pytest.mark.parametrize(("spot", "strike", "t", "r", "q", "sigma"), _CASES)
    def test_put_delta_in_negative_unit_interval(
        self,
        spot: float,
        strike: float,
        t: float,
        r: float,
        q: float,
        sigma: float,
    ) -> None:
        delta = bs_delta(spot, strike, t, r, q, sigma, "put")
        assert -1.0 < delta < 0.0

    def test_atm_delta_magnitude_about_half(self) -> None:
        call = bs_delta(100.0, 100.0, 1.0, 0.0, 0.0, 0.20, "call")
        put = bs_delta(100.0, 100.0, 1.0, 0.0, 0.0, 0.20, "put")
        assert call == pytest.approx(0.5, abs=0.05)
        assert put == pytest.approx(-0.5, abs=0.05)

    def test_deep_itm_call_delta_approaches_dividend_discount(self) -> None:
        q, t = 0.03, 1.0
        delta = bs_delta(1000.0, 10.0, t, 0.02, q, 0.20, "call")
        assert delta == pytest.approx(math.exp(-q * t), abs=1e-6)

    def test_deep_otm_call_delta_approaches_zero(self) -> None:
        delta = bs_delta(10.0, 1000.0, 1.0, 0.02, 0.03, 0.20, "call")
        assert delta == pytest.approx(0.0, abs=1e-6)


# ── Implied volatility: None on bad / impossible inputs ──────────────


class TestImpliedVolReturnsNone:
    def test_price_above_spot_call(self) -> None:
        # A call can never be worth more than the (discounted) spot.
        assert implied_volatility(150.0, 100.0, 100.0, 1.0, 0.05, 0.0, "call") is None

    def test_zero_t_years(self) -> None:
        assert implied_volatility(5.0, 100.0, 100.0, 0.0, 0.05, 0.0, "call") is None

    def test_nonpositive_price(self) -> None:
        assert implied_volatility(0.0, 100.0, 100.0, 1.0, 0.05, 0.0, "call") is None

    def test_nonpositive_spot(self) -> None:
        assert implied_volatility(5.0, 0.0, 100.0, 1.0, 0.05, 0.0, "call") is None

    def test_nonpositive_strike(self) -> None:
        assert implied_volatility(5.0, 100.0, 0.0, 1.0, 0.05, 0.0, "call") is None

    def test_price_below_intrinsic_call(self) -> None:
        # Discounted intrinsic is ~100; a near-zero price is not invertible.
        assert implied_volatility(1e-9, 200.0, 100.0, 1.0, 0.0, 0.0, "call") is None


# ── Pricing-helper input validation (raises) ─────────────────────────


class TestInputValidation:
    def test_bs_price_zero_sigma_raises(self) -> None:
        with pytest.raises(ValueError, match="sigma must be positive"):
            bs_price(100.0, 100.0, 1.0, 0.05, 0.0, 0.0, "call")

    def test_bs_price_negative_sigma_raises(self) -> None:
        with pytest.raises(ValueError, match="sigma must be positive"):
            bs_price(100.0, 100.0, 1.0, 0.05, 0.0, -0.20, "call")

    def test_bs_price_zero_t_years_raises(self) -> None:
        with pytest.raises(ValueError, match="t_years must be positive"):
            bs_price(100.0, 100.0, 0.0, 0.05, 0.0, 0.20, "call")

    def test_bs_price_nonpositive_spot_raises(self) -> None:
        with pytest.raises(ValueError, match="spot must be positive"):
            bs_price(0.0, 100.0, 1.0, 0.05, 0.0, 0.20, "call")

    def test_bs_price_nonpositive_strike_raises(self) -> None:
        with pytest.raises(ValueError, match="strike must be positive"):
            bs_price(100.0, -1.0, 1.0, 0.05, 0.0, 0.20, "put")

    def test_bs_delta_zero_sigma_raises(self) -> None:
        with pytest.raises(ValueError, match="sigma must be positive"):
            bs_delta(100.0, 100.0, 1.0, 0.05, 0.0, 0.0, "call")

    def test_bs_delta_zero_t_years_raises(self) -> None:
        with pytest.raises(ValueError, match="t_years must be positive"):
            bs_delta(100.0, 100.0, 0.0, 0.05, 0.0, 0.20, "put")
