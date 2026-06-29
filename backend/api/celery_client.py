"""Celery *producer* client.

The API does not run the analyses -- it only enqueues them: this client talks to
the same Redis broker / backend as the ``queue`` worker and dispatches tasks by
their registered name, then reads results back by task id.
"""

import os
from typing import Any

from celery import Celery
from celery.result import AsyncResult
from dotenv import load_dotenv


def build_celery_app() -> Celery:
    """Build the producer Celery app from ``DO_REDIS_CONNECTION``."""
    load_dotenv()
    redis_connection = os.environ.get("DO_REDIS_CONNECTION")
    app = Celery("api", broker=redis_connection, backend=redis_connection)
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
    )
    return app


celery_app = build_celery_app()


def dispatch(
    task_name: str, payload: dict[str, Any], user_email: str | None = None
) -> str:
    """Enqueue *task_name* with *payload* (+ requester email) and return its id."""
    return celery_app.send_task(task_name, args=[payload, user_email]).id


def get_result(task_id: str) -> AsyncResult:
    """The :class:`AsyncResult` for *task_id*, read from the result backend."""
    return AsyncResult(task_id, app=celery_app)
