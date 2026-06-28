"""Offline tests for the tuning API.

The Celery producer is mocked, so these run without a broker: submissions assert
the right task name and payload are dispatched, and polling reads a mocked
:class:`AsyncResult`.
"""

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import main

client = TestClient(main.app)

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


@pytest.mark.parametrize(("path", "task_name", "body"), SUBMIT_CASES)
def test_submit_enqueues_task(path: str, task_name: str, body: dict) -> None:
    sent = MagicMock()
    sent.id = "task-123"
    with patch("celery_client.celery_app.send_task", return_value=sent) as send:
        response = client.post(path, json=body)
    assert response.status_code == 202
    assert response.json() == {"task_id": "task-123"}
    assert send.call_args.args[0] == task_name
    assert send.call_args.kwargs["args"][0]["tickers"] == body["tickers"]


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
    assert payload["result"] == {"labels": {"AAA": 0}}
    assert payload["error"] is None
    assert payload["started_at"] is None
    assert payload["finished_at"] is None
    assert payload["elapsed_seconds"] is None


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
    assert payload["result"] is None


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
    assert payload["result"] is None
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


def _mock_redis(
    keys: list[str],
    mget_values: list[Any] | None = None,
) -> MagicMock:
    """Build a mock Redis client for list-jobs tests."""
    mock = MagicMock()
    mock.scan_iter.return_value = keys
    if mget_values is not None:
        mock.mget.return_value = mget_values
    return mock


def test_list_jobs_empty() -> None:
    with (
        patch("redis_client._redis_client", _mock_redis([])),
        patch("routers.jobs.get_task_start_times", return_value={}),
    ):
        response = client.get("/jobs")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
    assert data["items"] == []
    assert data["offset"] == 0
    assert data["limit"] == 20


def test_list_jobs_returns_all_statuses() -> None:
    keys = ["celery-task-meta-aaa", "celery-task-meta-bbb"]
    mget_vals = [
        json.dumps({"task_id": "aaa", "status": "SUCCESS", "result": {"x": 1}}),
        json.dumps({"task_id": "bbb", "status": "PENDING"}),
    ]
    with (
        patch("redis_client._redis_client", _mock_redis(keys, mget_vals)),
        patch("routers.jobs.get_task_start_times", return_value={}),
    ):
        response = client.get("/jobs")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["items"]) == 2
    # keys are sorted, so aaa comes first
    assert data["items"][0]["task_id"] == "aaa"
    assert data["items"][0]["state"] == "SUCCESS"
    assert data["items"][0]["result"] == {"x": 1}
    assert data["items"][1]["state"] == "PENDING"


def test_list_jobs_failure_entry_formats_error() -> None:
    keys = ["celery-task-meta-xyz"]
    mget_vals = [
        json.dumps(
            {
                "task_id": "xyz",
                "status": "FAILURE",
                "result": {"exc_type": "ValueError", "exc_message": ["bad input"]},
            }
        )
    ]
    with (
        patch("redis_client._redis_client", _mock_redis(keys, mget_vals)),
        patch("routers.jobs.get_task_start_times", return_value={}),
    ):
        response = client.get("/jobs")
    item = response.json()["items"][0]
    assert item["state"] == "FAILURE"
    assert "ValueError" in item["error"]
    assert "bad input" in item["error"]
    assert item["result"] is None


def test_list_jobs_pagination() -> None:
    # 5 tasks; request page starting at offset=2, size=2
    keys = [f"celery-task-meta-task-{i}" for i in range(5)]
    # sorted task ids: task-0 .. task-4; offset=2,limit=2 → task-2, task-3
    mget_vals = [
        json.dumps({"task_id": f"task-{i}", "status": "SUCCESS", "result": {}})
        for i in range(2, 4)
    ]
    with (
        patch("redis_client._redis_client", _mock_redis(keys, mget_vals)),
        patch("routers.jobs.get_task_start_times", return_value={}),
    ):
        response = client.get("/jobs?offset=2&limit=2")
    data = response.json()
    assert data["total"] == 5
    assert data["offset"] == 2
    assert data["limit"] == 2
    assert len(data["items"]) == 2
    assert data["items"][0]["task_id"] == "task-2"
    assert data["items"][1]["task_id"] == "task-3"


def test_list_jobs_rejects_invalid_limit() -> None:
    response = client.get("/jobs?limit=0")
    assert response.status_code == 422


def test_list_jobs_rejects_limit_above_max() -> None:
    response = client.get("/jobs?limit=101")
    assert response.status_code == 422


def test_list_jobs_includes_timing_when_start_time_recorded() -> None:
    keys = ["celery-task-meta-task-abc"]
    mget_vals = [
        json.dumps(
            {
                "task_id": "task-abc",
                "status": "SUCCESS",
                "result": {},
                "date_done": "2024-01-01T12:00:05+00:00",
            }
        )
    ]
    start_times = {"task-abc": "2024-01-01T12:00:00+00:00"}
    with (
        patch("redis_client._redis_client", _mock_redis(keys, mget_vals)),
        patch("routers.jobs.get_task_start_times", return_value=start_times),
    ):
        response = client.get("/jobs")
    item = response.json()["items"][0]
    assert item["started_at"] == "2024-01-01T12:00:00+00:00"
    assert item["finished_at"] == "2024-01-01T12:00:05+00:00"
    assert item["elapsed_seconds"] == pytest.approx(5.0)
