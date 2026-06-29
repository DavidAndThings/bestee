"""Celery *producer* client.

The API does not run the analyses -- it only enqueues them: this client talks to
the same Redis broker / backend as the ``queue`` worker (built from the shared
:func:`bestee_tasking.build_celery_app` factory) and dispatches tasks by their
registered name, then reads results back by task id.
"""

from typing import Any

from bestee_tasking import build_celery_app
from celery.result import AsyncResult

celery_app = build_celery_app("api")


def dispatch(
    task_name: str, payload: dict[str, Any], user_email: str | None = None
) -> str:
    """Enqueue *task_name* with *payload* (+ requester email) and return its id."""
    return celery_app.send_task(task_name, args=[payload, user_email]).id


def get_result(task_id: str) -> AsyncResult:
    """The :class:`AsyncResult` for *task_id*, read from the result backend."""
    return AsyncResult(task_id, app=celery_app)
