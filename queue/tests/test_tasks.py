"""Offline tests for the Celery task wiring (no broker, no network).

The optimizers fetch live market data, so these only exercise what runs before
any work: task registration / naming, the app configuration, and request
validation (an invalid payload is rejected by the pydantic model first).
"""

from typing import Any

import pytest
from pydantic import ValidationError

import tasks
from main import get_celery_app

_TASKS: dict[str, Any] = {
    "tuning.optimize_clustering": tasks.optimize_clustering,
    "tuning.optimize_regime": tasks.optimize_regime,
    "tuning.optimize_fama_french": tasks.optimize_fama_french,
    "tuning.optimize_rrg": tasks.optimize_rrg,
}


@pytest.mark.parametrize(("name", "task"), _TASKS.items())
def test_task_is_registered_under_its_name(name: str, task: Any) -> None:
    assert task.name == name


def test_app_is_configured_for_json_results() -> None:
    app = get_celery_app()
    assert app.conf.task_serializer == "json"
    assert app.conf.result_serializer == "json"
    assert app.conf.accept_content == ["json"]


@pytest.mark.parametrize("task", _TASKS.values())
def test_invalid_request_is_rejected_before_any_work(task: Any) -> None:
    # An empty payload is missing the required fields, so the pydantic request
    # model raises before the optimizer (and any network call) runs.
    with pytest.raises(ValidationError):
        task({})


def test_completion_email_noop_without_recipient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[Any] = []
    monkeypatch.setattr(tasks.resend.Emails, "send", sent.append)
    tasks.send_completion_email("task-1", "clustering_abc", None)
    assert sent == []


def test_completion_email_noop_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    sent: list[Any] = []
    monkeypatch.setattr(tasks.resend.Emails, "send", sent.append)
    tasks.send_completion_email("task-1", "clustering_abc", "user@example.com")
    assert sent == []


def test_completion_email_includes_task_and_result_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    sent: list[dict[str, Any]] = []
    monkeypatch.setattr(tasks.resend.Emails, "send", sent.append)
    tasks.send_completion_email("task-abc", "clustering_xyz", "user@example.com")
    assert len(sent) == 1
    payload = sent[0]
    assert payload["to"] == "user@example.com"
    assert "task-abc" in payload["html"]
    assert "clustering_xyz" in payload["html"]


def test_completion_email_swallows_send_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")

    def _boom(_payload: Any) -> None:
        raise RuntimeError("mailer down")

    monkeypatch.setattr(tasks.resend.Emails, "send", _boom)
    # Must not raise -- a flaky mailer should never fail the task.
    tasks.send_completion_email("task-1", "rrg_abc", "user@example.com")
