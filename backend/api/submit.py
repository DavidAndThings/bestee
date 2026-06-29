"""Resolve ticker terms, enqueue a job, and persist it for the listing."""

import datetime as dt

from bestee_compute.workflow import tuning
from pydantic import BaseModel

from celery_client import dispatch
from redis_client import record_job
from resolve import resolve_request
from schemas import TaskHandle


def enqueue(
    analysis: str,
    task_name: str,
    request: BaseModel,
    email: str | None,
    user_id: str,
) -> TaskHandle:
    """Resolve, enqueue *task_name*, record the job, and return its ids.

    ``result_id`` is a hash of the *resolved* request, identical to what the
    worker recomputes and saves under, so it is known at submit time -- before
    the job runs -- and lets a client build the ``/results/{result_id}/...``
    URL.  The job is also persisted (per ``user_id``) so it appears in the
    listing immediately and survives across browsers / devices.
    """
    resolved = resolve_request(request)
    task_id = dispatch(task_name, resolved, email)
    result_id = tuning.result_id(analysis, type(request).model_validate(resolved))
    record_job(
        user_id,
        task_id,
        analysis,
        resolved,
        result_id,
        dt.datetime.now(dt.UTC).isoformat(),
    )
    return TaskHandle(task_id=task_id, result_id=result_id)
