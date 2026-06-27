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
