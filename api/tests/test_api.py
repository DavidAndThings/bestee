"""Offline tests for the tuning API.

The Celery producer is mocked, so these run without a broker: submissions assert
the right task name and payload are dispatched, and polling reads a mocked
:class:`AsyncResult`.
"""

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
            "start_date": "2020-01-01",
            "end_date": "2020-12-31",
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


def test_poll_returns_result_when_successful() -> None:
    async_result = MagicMock()
    async_result.state = "SUCCESS"
    async_result.result = {"labels": {"AAA": 0}}
    with patch("celery_client.AsyncResult", return_value=async_result):
        response = client.get("/jobs/task-123")
    assert response.status_code == 200
    payload = response.json()
    assert payload["task_id"] == "task-123"
    assert payload["state"] == "SUCCESS"
    assert payload["result"] == {"labels": {"AAA": 0}}
    assert payload["error"] is None


def test_poll_returns_error_when_failed() -> None:
    async_result = MagicMock()
    async_result.state = "FAILURE"
    async_result.result = ValueError("boom")
    with patch("celery_client.AsyncResult", return_value=async_result):
        response = client.get("/jobs/task-123")
    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] == "FAILURE"
    assert payload["error"] == "boom"
    assert payload["result"] is None


def test_poll_returns_state_only_when_pending() -> None:
    async_result = MagicMock()
    async_result.state = "PENDING"
    async_result.result = None
    with patch("celery_client.AsyncResult", return_value=async_result):
        response = client.get("/jobs/task-456")
    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] == "PENDING"
    assert payload["result"] is None
    assert payload["error"] is None
