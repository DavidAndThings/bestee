"""Tests for bestee_compute.securities.rates — point-in-time risk-free rate.

The HTTP layer is mocked, so these run offline and assert the analytical
core: percent->decimal conversion, point-in-time selection (the latest
valid observation on or before the as-of date), skipping of FRED ``"."``
missing values, and the error paths.
"""

import datetime
from unittest.mock import MagicMock, patch

import httpx
import pytest

from bestee_compute.securities.rates import get_risk_free_rate

_HTTPX_PATCH = "bestee_compute.securities.rates.httpx.Client"
_AS_OF = datetime.date(2024, 2, 1)


def _response(observations: list[dict[str, str]]) -> MagicMock:
    response = MagicMock(spec=httpx.Response)
    response.raise_for_status = MagicMock()
    response.json = MagicMock(return_value={"observations": observations})
    return response


def _mock_client(observations: list[dict[str, str]]) -> MagicMock:
    """A client whose .get() returns a fixed FRED observations payload."""
    client = MagicMock()
    client.get.return_value = _response(observations)
    return client


def _run(observations: list[dict[str, str]]) -> float:
    with patch(_HTTPX_PATCH) as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value = _mock_client(observations)
        return get_risk_free_rate(_AS_OF, api_key="test-key")


# ── Conversion & selection ───────────────────────────────────────────


class TestRiskFreeRate:
    def test_percent_to_decimal_conversion(self) -> None:
        """A FRED percent value is returned as a decimal fraction."""
        rate = _run([{"date": "2024-02-01", "value": "5.25"}])
        assert rate == pytest.approx(0.0525)

    def test_point_in_time_selection(self) -> None:
        """The most recent value <= as_of is used; an older one is not.

        FRED is queried newest-first with observation_end pinned to
        as_of, so the first row is the latest valid yield on/before it.
        """
        rate = _run(
            [
                {"date": "2024-02-01", "value": "5.30"},  # latest <= as_of
                {"date": "2024-01-31", "value": "5.25"},
                {"date": "2024-01-30", "value": "5.20"},
            ]
        )
        assert rate == pytest.approx(0.0530)
        assert rate != pytest.approx(0.0520)

    def test_request_is_point_in_time(self) -> None:
        """The query pins observation_end to as_of and sorts descending."""
        client = _mock_client([{"date": "2024-02-01", "value": "5.25"}])
        with patch(_HTTPX_PATCH) as mock_client_cls:
            mock_client_cls.return_value.__enter__.return_value = client
            get_risk_free_rate(_AS_OF, api_key="test-key")
        _, kwargs = client.get.call_args
        params = kwargs["params"]
        assert params["observation_end"] == _AS_OF.isoformat()
        assert params["sort_order"] == "desc"
        assert params["series_id"] == "DGS3MO"
        assert params["file_type"] == "json"

    def test_missing_values_are_skipped(self) -> None:
        """A leading ``"."`` is ignored; the next numeric value is used."""
        rate = _run(
            [
                {"date": "2024-02-01", "value": "."},  # missing -> skipped
                {"date": "2024-01-31", "value": "5.10"},  # used
            ]
        )
        assert rate == pytest.approx(0.0510)


# ── Error paths ──────────────────────────────────────────────────────


class TestErrors:
    @pytest.mark.parametrize(
        "observations",
        [
            [],
            [{"date": "2024-02-01", "value": "."}],
        ],
    )
    def test_no_usable_observation_raises(
        self, observations: list[dict[str, str]]
    ) -> None:
        with pytest.raises(ValueError, match="no usable observation"):
            _run(observations)

    @patch("bestee_compute.securities.rates.find_dotenv", return_value="")
    @patch("bestee_compute.securities.rates.load_dotenv")
    def test_missing_api_key_raises(
        self, _load: MagicMock, _find: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="FRED API key"):
            get_risk_free_rate(_AS_OF)
