"""Offline tests for the tuning API.

The Celery producer is mocked, so these run without a broker: submissions assert
the right task name and payload are dispatched, and polling reads a mocked
:class:`AsyncResult`.
"""

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import main
from auth import require_auth

# Bypass Clerk authentication in all tests.
main.app.dependency_overrides[require_auth] = lambda: {"sub": "test_user"}

client = TestClient(main.app)


@pytest.fixture(autouse=True)
def _stub_resolvers():
    """Keep every test offline.

    Resolve ticker terms to themselves (the real resolvers hit the Massive API),
    no-op the Redis-backed job record write on submit, and default the job-record
    read / start-time lookups so polling never touches the real result backend.
    Individual tests override these where they assert on the values.
    """
    with (
        patch(
            "resolve.resolve_terms_to_tickers",
            side_effect=lambda terms, **_: list(terms),
        ),
        patch(
            "resolve.resolve_term_to_ticker",
            side_effect=lambda term, **_: term,
        ),
        patch("submit.record_job"),
        patch("routers.jobs.get_job_record", return_value=None),
        patch("routers.jobs.get_task_start_times", return_value={}),
    ):
        yield


SUBMIT_CASES = [
    (
        "/tasks/clustering",
        "tuning.optimize_clustering",
        {
            "tickers": ["AAA", "BBB"],
            "start_date": "2020-01-01",
            "end_date": "2020-12-31",
        },
    ),
    (
        "/tasks/regime",
        "tuning.optimize_regime",
        {
            "tickers": ["AAA"],
            "start_date": "2020-01-01",
            "end_date": "2020-12-31",
        },
    ),
    (
        "/tasks/fama-french",
        "tuning.optimize_fama_french",
        {
            "tickers": ["AAA"],
            "intervals": [{"start_date": "2020-01-01", "end_date": "2020-12-31"}],
            "oos_dates": ["2021-01-15"],
        },
    ),
    (
        "/tasks/rrg",
        "tuning.optimize_rrg",
        {
            "tickers": ["AAA"],
            "start_date": "2020-01-01",
            "end_date": "2020-12-31",
        },
    ),
]


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_cors_allows_configured_origin() -> None:
    """An allow-listed browser origin is echoed back in the CORS header."""
    response = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert response.status_code == 200
    assert (
        response.headers.get("access-control-allow-origin") == "http://localhost:5173"
    )


@pytest.mark.parametrize(("path", "task_name", "body"), SUBMIT_CASES)
def test_submit_enqueues_task(path: str, task_name: str, body: dict) -> None:
    sent = MagicMock()
    sent.id = "task-123"
    with patch("celery_client.celery_app.send_task", return_value=sent) as send:
        response = client.post(path, json=body)
    assert response.status_code == 202
    body_json = response.json()
    assert body_json["task_id"] == "task-123"
    # The result id is a deterministic hash of the resolved request (identity
    # resolution here), known at submit time before the job runs.
    assert re.fullmatch(r"[a-z_]+_[0-9a-f]{16}", body_json["result_id"])
    assert send.call_args.args[0] == task_name
    assert send.call_args.kwargs["args"][0]["tickers"] == body["tickers"]


def test_submit_passes_user_email_to_task() -> None:
    from auth import get_user_email

    main.app.dependency_overrides[get_user_email] = lambda: "user@example.com"
    sent = MagicMock()
    sent.id = "task-xyz"
    try:
        with patch("celery_client.celery_app.send_task", return_value=sent) as send:
            response = client.post(
                "/tasks/clustering",
                json={
                    "tickers": ["AAA", "BBB"],
                    "start_date": "2020-01-01",
                    "end_date": "2020-12-31",
                },
            )
        assert response.status_code == 202
        # args = [payload, user_email]
        assert send.call_args.kwargs["args"][1] == "user@example.com"
    finally:
        main.app.dependency_overrides.pop(get_user_email, None)


def test_clustering_requires_two_tickers() -> None:
    body = {
        "tickers": ["AAA"],
        "start_date": "2020-01-01",
        "end_date": "2020-12-31",
    }
    response = client.post("/tasks/clustering", json=body)
    assert response.status_code == 422


def test_rrg_ticker_reference_requires_benchmark() -> None:
    body = {
        "tickers": ["AAA"],
        "start_date": "2020-01-01",
        "end_date": "2020-12-31",
        "reference_type": "ticker",
    }
    response = client.post("/tasks/rrg", json=body)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Term -> ticker resolution at submission
# ---------------------------------------------------------------------------


def test_submit_resolves_terms_to_tickers() -> None:
    """Industry-title / company-name terms are expanded before dispatch."""
    sent = MagicMock()
    sent.id = "task-resolve"
    with (
        patch(
            "resolve.resolve_terms_to_tickers",
            return_value=["AAPL", "MSFT", "ADBE"],
        ),
        patch("celery_client.celery_app.send_task", return_value=sent) as send,
    ):
        response = client.post(
            "/tasks/clustering",
            json={
                "tickers": ["SERVICES-PREPACKAGED SOFTWARE", "Apple Inc."],
                "start_date": "2020-01-01",
                "end_date": "2020-12-31",
            },
        )
    assert response.status_code == 202
    assert send.call_args.kwargs["args"][0]["tickers"] == ["AAPL", "MSFT", "ADBE"]


def test_submit_resolves_rrg_benchmark() -> None:
    """A benchmark term is resolved to a single symbol before dispatch."""
    sent = MagicMock()
    sent.id = "task-bench"
    with (
        patch("resolve.resolve_terms_to_tickers", return_value=["AAPL"]),
        patch("resolve.resolve_term_to_ticker", return_value="SPY"),
        patch("celery_client.celery_app.send_task", return_value=sent) as send,
    ):
        response = client.post(
            "/tasks/rrg",
            json={
                "tickers": ["Apple Inc."],
                "start_date": "2020-01-01",
                "end_date": "2020-12-31",
                "reference_type": "ticker",
                "benchmark_ticker": "SPDR S&P 500 ETF Trust",
            },
        )
    assert response.status_code == 202
    payload = send.call_args.kwargs["args"][0]
    assert payload["tickers"] == ["AAPL"]
    assert payload["benchmark_ticker"] == "SPY"


def test_submit_422_when_no_securities_match() -> None:
    """Terms that resolve to nothing are rejected synchronously."""
    with patch("resolve.resolve_terms_to_tickers", return_value=[]):
        response = client.post(
            "/tasks/regime",
            json={
                "tickers": ["No Such Industry"],
                "start_date": "2020-01-01",
                "end_date": "2020-12-31",
            },
        )
    assert response.status_code == 422
    assert "matched" in response.json()["detail"].lower()


def test_submit_422_when_resolution_below_cluster_minimum() -> None:
    """Two terms that resolve to one security fail clustering's >=2 rule."""
    with patch("resolve.resolve_terms_to_tickers", return_value=["AAPL"]):
        response = client.post(
            "/tasks/clustering",
            json={
                "tickers": ["Apple Inc.", "Apple Computer"],
                "start_date": "2020-01-01",
                "end_date": "2020-12-31",
            },
        )
    assert response.status_code == 422


def _mock_async_result(
    state: str, result: Any = None, date_done: datetime | None = None
) -> MagicMock:
    ar = MagicMock()
    ar.state = state
    ar.result = result
    ar.date_done = date_done
    return ar


def test_poll_returns_result_when_successful() -> None:
    ar = _mock_async_result("SUCCESS", result={"labels": {"AAA": 0}})
    with (
        patch("celery_client.AsyncResult", return_value=ar),
        patch("routers.jobs.get_task_start_times", return_value={}),
    ):
        response = client.get("/jobs/task-123")
    assert response.status_code == 200
    payload = response.json()
    assert payload["task_id"] == "task-123"
    assert payload["state"] == "SUCCESS"
    assert payload["error"] is None
    assert payload["result_id"] is None
    assert payload["started_at"] is None
    assert payload["finished_at"] is None
    assert payload["elapsed_seconds"] is None


def test_poll_surfaces_result_id_on_success() -> None:
    ar = _mock_async_result("SUCCESS", result={"result_id": "clustering_abc123"})
    with (
        patch("celery_client.AsyncResult", return_value=ar),
        patch("redis_client.get_task_start_times", return_value={}),
    ):
        response = client.get("/jobs/task-123")
    assert response.status_code == 200
    assert response.json()["result_id"] == "clustering_abc123"


def test_poll_returns_error_when_failed() -> None:
    ar = _mock_async_result("FAILURE", result=ValueError("boom"))
    with (
        patch("celery_client.AsyncResult", return_value=ar),
        patch("routers.jobs.get_task_start_times", return_value={}),
    ):
        response = client.get("/jobs/task-123")
    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] == "FAILURE"
    assert payload["error"] == "boom"


def test_poll_returns_state_only_when_pending() -> None:
    ar = _mock_async_result("PENDING")
    with (
        patch("celery_client.AsyncResult", return_value=ar),
        patch("routers.jobs.get_task_start_times", return_value={}),
    ):
        response = client.get("/jobs/task-456")
    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] == "PENDING"
    assert payload["error"] is None


def test_poll_includes_timing_when_start_time_recorded() -> None:
    ar = _mock_async_result(
        "SUCCESS",
        result={"ok": True},
        date_done=datetime(2024, 1, 1, 12, 0, 10, tzinfo=UTC),
    )
    start_times = {"task-999": "2024-01-01T12:00:00+00:00"}
    with (
        patch("celery_client.AsyncResult", return_value=ar),
        patch("routers.jobs.get_task_start_times", return_value=start_times),
    ):
        response = client.get("/jobs/task-999")
    payload = response.json()
    assert payload["started_at"] == "2024-01-01T12:00:00+00:00"
    assert payload["finished_at"] == "2024-01-01T12:00:10+00:00"
    assert payload["elapsed_seconds"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# GET /jobs  (list all jobs)
# ---------------------------------------------------------------------------


def _record(task_id: str, analysis: str = "clustering") -> dict[str, Any]:
    """A stored job record like :func:`redis_client.record_job` writes."""
    return {
        "task_id": task_id,
        "analysis": analysis,
        "payload": {"tickers": ["AAA"]},
        "result_id": f"{analysis}_{task_id}",
        "created_at": "2024-01-01T00:00:00+00:00",
    }


def test_list_jobs_empty() -> None:
    with patch("routers.jobs.list_user_jobs", return_value=(0, [])):
        response = client.get("/jobs")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
    assert data["items"] == []
    assert data["offset"] == 0
    assert data["limit"] == 50


def test_list_jobs_returns_recorded_jobs() -> None:
    records = [_record("aaa"), _record("bbb", analysis="regime")]
    metas = [
        {
            "task_id": "aaa",
            "status": "SUCCESS",
            "result": {"result_id": "clustering_aaa"},
        },
        {"task_id": "bbb", "status": "PENDING"},
    ]
    with (
        patch("routers.jobs.list_user_jobs", return_value=(2, records)),
        patch("routers.jobs.get_task_metas", return_value=metas),
    ):
        response = client.get("/jobs")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["items"]) == 2
    # Order is the per-user list order (newest first), not a re-sort.
    assert data["items"][0]["task_id"] == "aaa"
    assert data["items"][0]["state"] == "SUCCESS"
    assert data["items"][0]["analysis"] == "clustering"
    assert data["items"][0]["payload"] == {"tickers": ["AAA"]}
    assert data["items"][0]["result_id"] == "clustering_aaa"
    assert data["items"][1]["state"] == "PENDING"
    # A queued job still surfaces its recorded inputs + deterministic result id.
    assert data["items"][1]["result_id"] == "regime_bbb"


def test_list_jobs_failure_entry_formats_error() -> None:
    records = [_record("xyz", analysis="rrg")]
    metas = [
        {
            "task_id": "xyz",
            "status": "FAILURE",
            "result": {"exc_type": "ValueError", "exc_message": ["bad input"]},
        }
    ]
    with (
        patch("routers.jobs.list_user_jobs", return_value=(1, records)),
        patch("routers.jobs.get_task_metas", return_value=metas),
    ):
        response = client.get("/jobs")
    item = response.json()["items"][0]
    assert item["state"] == "FAILURE"
    assert "ValueError" in item["error"]
    assert "bad input" in item["error"]


def test_list_jobs_pagination() -> None:
    records = [_record("task-2"), _record("task-3")]
    metas = [
        {"task_id": "task-2", "status": "SUCCESS", "result": {}},
        {"task_id": "task-3", "status": "SUCCESS", "result": {}},
    ]
    with (
        patch("routers.jobs.list_user_jobs", return_value=(5, records)),
        patch("routers.jobs.get_task_metas", return_value=metas),
    ):
        response = client.get("/jobs?offset=2&limit=2")
    data = response.json()
    assert data["total"] == 5
    assert data["offset"] == 2
    assert data["limit"] == 2
    assert [i["task_id"] for i in data["items"]] == ["task-2", "task-3"]


def test_list_jobs_rejects_invalid_limit() -> None:
    response = client.get("/jobs?limit=0")
    assert response.status_code == 422


def test_list_jobs_rejects_limit_above_max() -> None:
    response = client.get("/jobs?limit=101")
    assert response.status_code == 422


def test_list_jobs_includes_timing_when_start_time_recorded() -> None:
    records = [_record("task-abc")]
    metas = [
        {
            "task_id": "task-abc",
            "status": "SUCCESS",
            "result": {},
            "date_done": "2024-01-01T12:00:05+00:00",
        }
    ]
    start_times = {"task-abc": "2024-01-01T12:00:00+00:00"}
    with (
        patch("routers.jobs.list_user_jobs", return_value=(1, records)),
        patch("routers.jobs.get_task_metas", return_value=metas),
        patch("routers.jobs.get_task_start_times", return_value=start_times),
    ):
        response = client.get("/jobs")
    item = response.json()["items"][0]
    assert item["started_at"] == "2024-01-01T12:00:00+00:00"
    assert item["finished_at"] == "2024-01-01T12:00:05+00:00"
    assert item["elapsed_seconds"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# GET /results  and  GET /results/{result_id}
# ---------------------------------------------------------------------------


def test_list_results_returns_empty_when_dir_absent() -> None:
    with patch("routers.results._get_results_dir", return_value=Path("/no/such/dir")):
        response = client.get("/results")
    assert response.status_code == 200
    assert response.json() == []


# ---------------------------------------------------------------------------
# GET /search
# ---------------------------------------------------------------------------


def test_search_ranks_prefix_matches_first() -> None:
    terms = [
        "SERVICES-PREPACKAGED SOFTWARE",
        "Software AG",
        "Microsoft Corp",
        "Apple Inc",
    ]
    with patch("routers.ticker.get_all_search_terms", return_value=terms):
        response = client.get("/search", params={"q": "software"})
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "software"
    # "Software AG" (prefix) ranks ahead of the substring match.
    assert data["results"] == ["Software AG", "SERVICES-PREPACKAGED SOFTWARE"]
    assert data["count"] == 2


def test_search_respects_limit() -> None:
    terms = [f"Bank of {i:02d}" for i in range(50)]
    with patch("routers.ticker.get_all_search_terms", return_value=terms):
        response = client.get("/search", params={"q": "bank", "limit": 5})
    assert response.status_code == 200
    assert len(response.json()["results"]) == 5


def test_search_requires_query() -> None:
    response = client.get("/search")  # q is required
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /sic  and  GET /sic/{sic_code}/tickers
# ---------------------------------------------------------------------------


def test_list_sic_codes() -> None:
    import polars as pl

    df = pl.DataFrame(
        {
            "SIC Code": ["7372", "6021"],
            "Industry Title": [
                "SERVICES-PREPACKAGED SOFTWARE",
                "NATIONAL COMMERCIAL BANKS",
            ],
        }
    )
    with patch("routers.sic.get_sic_codes_df", return_value=df):
        response = client.get("/sic")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert data["codes"][0] == {
        "sic_code": "7372",
        "industry_title": "SERVICES-PREPACKAGED SOFTWARE",
    }


def test_tickers_for_sic_code() -> None:
    with (
        patch("routers.sic.get_tickers_by_sic_code", return_value=["AAA", "BBB"]),
        patch("routers.sic.get_ticker_name_map", return_value={"AAA": "Alpha Inc."}),
    ):
        response = client.get("/sic/7372/tickers")
    assert response.status_code == 200
    data = response.json()
    assert data["sic_code"] == "7372"
    assert data["count"] == 2
    # Names are looked up per ticker; an unknown ticker gets a null name.
    assert data["tickers"] == [
        {"ticker": "AAA", "name": "Alpha Inc."},
        {"ticker": "BBB", "name": None},
    ]


def test_get_result_returns_404_for_missing_result() -> None:
    with patch("routers.results._get_results_dir", return_value=Path("/no/such/dir")):
        response = client.get("/results/clustering_abc123def4")
    assert response.status_code == 404


def test_get_result_returns_422_for_unknown_analysis_type() -> None:
    # Make the path appear to exist so we reach the loader dispatch.
    mock_path = MagicMock(spec=Path)
    mock_path.__truediv__ = lambda self, other: mock_path
    mock_path.exists.return_value = True
    (mock_path / "metadata.json").exists.return_value = True
    with patch("routers.results._get_results_dir", return_value=mock_path):
        response = client.get("/results/unknown_abc123def4")
    assert response.status_code == 422


def test_get_result_loads_and_returns_clustering_result() -> None:
    from bestee_compute.workflow.tuning import ClusteringTuning

    fake = ClusteringTuning(
        labels={"AAA": 0, "BBB": 1},
        residualization_window=20,
        normalization_window=20,
        n_components=1,
        similarity_metric="corr",
        silhouette=0.72,
        n_clusters=2,
    )
    mock_path = MagicMock(spec=Path)
    mock_path.__truediv__ = lambda self, other: mock_path
    mock_path.exists.return_value = True
    (mock_path / "metadata.json").exists.return_value = True
    with (
        patch("routers.results._get_results_dir", return_value=mock_path),
        patch("routers.results._LOADERS", {"clustering": lambda _: fake}),
    ):
        response = client.get("/results/clustering_abc123def4")
    assert response.status_code == 200
    data = response.json()
    assert data["result_id"] == "clustering_abc123def4"
    assert data["labels"] == {"AAA": 0, "BBB": 1}
    assert data["silhouette"] == pytest.approx(0.72)


# ---------------------------------------------------------------------------
# GET /results/{result_id}/{aspect_name}
# ---------------------------------------------------------------------------


def _mock_result_path() -> MagicMock:
    """A Path whose result directory and metadata.json appear to exist."""
    mock_path = MagicMock(spec=Path)
    mock_path.__truediv__ = lambda self, other: mock_path
    mock_path.exists.return_value = True
    (mock_path / "metadata.json").exists.return_value = True
    return mock_path


def _get_aspect(result_id: str, aspect: str, loaders: dict) -> Any:
    with (
        patch("routers.results._get_results_dir", return_value=_mock_result_path()),
        patch("routers.results._LOADERS", loaders),
    ):
        return client.get(f"/results/{result_id}/{aspect}")


def test_aspect_cluster_label() -> None:
    from bestee_compute.workflow.tuning import ClusteringTuning

    fake = ClusteringTuning(
        labels={"AAA": 0, "BBB": 1},
        residualization_window=20,
        normalization_window=20,
        n_components=1,
        similarity_metric="corr",
        silhouette=0.7,
        n_clusters=2,
    )
    response = _get_aspect(
        "clustering_abc123def4", "cluster_label", {"clustering": lambda _: fake}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["result_id"] == "clustering_abc123def4"
    assert data["aspect"] == "cluster_label"
    assert data["columns"] == ["ticker", "cluster_label"]
    assert data["rows"] == [
        {"ticker": "AAA", "cluster_label": 0},
        {"ticker": "BBB", "cluster_label": 1},
    ]


def test_aspect_coordinates() -> None:
    import datetime as dt

    import polars as pl
    from bestee_compute.workflow.tuning import RRGTuning

    fake = RRGTuning(
        normalization_window=20,
        momentum_lookback=5,
        smoothing_span=None,
        signal_to_noise=1.0,
        relative_strength=pl.DataFrame(
            {"Timestamp": [dt.datetime(2024, 1, 2)], "AAA_rel_strength": [1.1]}
        ),
        relative_momentum=pl.DataFrame(
            {"Timestamp": [dt.datetime(2024, 1, 2)], "AAA_rel_momentum": [0.5]}
        ),
    )
    response = _get_aspect("rrg_abc123def4", "coordinates", {"rrg": lambda _: fake})
    assert response.status_code == 200
    data = response.json()
    assert data["columns"] == [
        "timestamp",
        "ticker",
        "relative_strength",
        "relative_momentum",
    ]
    assert len(data["rows"]) == 1
    row = data["rows"][0]
    assert row["ticker"] == "AAA"
    assert row["relative_strength"] == pytest.approx(1.1)
    assert row["relative_momentum"] == pytest.approx(0.5)
    assert row["timestamp"].startswith("2024-01-02")


def test_aspect_regime_label() -> None:
    import datetime as dt

    import numpy as np
    import polars as pl
    from bestee_compute.workflow import regimes
    from bestee_compute.workflow.tuning import RegimeTuning

    states = pl.DataFrame(
        {
            "Timestamp": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2)],
            "Regime": [0, 1],
            "Regime_Prob_0": [0.5, 0.5],
            "Regime_Prob_1": [0.5, 0.5],
        }
    )
    result = regimes.RegimeResult(
        ticker="AAA",
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
    fake = RegimeTuning(
        labels={"AAA": 1},
        results={"AAA": result},
        residualization_window=20,
        normalization_window=20,
        hmm_lag=1,
        mean_bic=100.0,
        oos_regimes=pl.DataFrame(
            {
                "Timestamp": [dt.datetime(2024, 2, 1)],
                "AAA_Regime": [1],
                "AAA_Regime_Prob": [0.8],
            }
        ),
    )
    response = _get_aspect(
        "regime_abc123def4", "regime_label", {"regime": lambda _: fake}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["columns"] == ["timestamp", "ticker", "regime_label"]
    # Two in-sample labels plus the out-of-sample nowcast.
    assert [row["regime_label"] for row in data["rows"]] == [0, 1, 1]


def test_aspect_ff_residuals() -> None:
    import datetime as dt

    import polars as pl
    from bestee_compute.workflow.tuning import FamaFrenchTuning

    fake = FamaFrenchTuning(
        factors_to_use=6,
        specifications=[],
        results={},
        mean_adjusted_r_squared=0.5,
        oos_residuals=pl.DataFrame(
            {
                "Date": [dt.date(2024, 1, 15)],
                "Ticker": ["AAA"],
                "Specification": ["FF6_2020-01-01_2023-12-31"],
                "Residual": [0.01],
                "ZScore": [1.0],
                "PValue": [0.31],
            }
        ),
    )
    response = _get_aspect(
        "fama_french_abc123def4", "ff_residuals", {"fama_french": lambda _: fake}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["columns"] == [
        "date",
        "ticker",
        "specification",
        "residual",
        "p_value",
    ]
    assert data["rows"][0]["p_value"] == pytest.approx(0.31)


def test_aspect_unknown_returns_404() -> None:
    from bestee_compute.workflow.tuning import ClusteringTuning

    fake = ClusteringTuning(
        labels={"AAA": 0},
        residualization_window=20,
        normalization_window=20,
        n_components=1,
        similarity_metric="corr",
        silhouette=0.7,
        n_clusters=1,
    )
    response = _get_aspect(
        "clustering_abc123def4", "not_an_aspect", {"clustering": lambda _: fake}
    )
    assert response.status_code == 404
