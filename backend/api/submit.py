"""Resolve ticker terms, enqueue a job, and report its deterministic result id."""

from bestee_compute.workflow import tuning
from pydantic import BaseModel

from celery_client import dispatch
from resolve import resolve_request
from schemas import TaskHandle


def enqueue(
    analysis: str, task_name: str, request: BaseModel, email: str | None
) -> TaskHandle:
    """Resolve, enqueue *task_name*, and return the task id + deterministic result id.

    ``result_id`` is a hash of the *resolved* request, identical to what the
    worker recomputes and saves under, so it is known at submit time -- before
    the job runs -- and lets a client build the ``/results/{result_id}/...`` URL.
    """
    resolved = resolve_request(request)
    task_id = dispatch(task_name, resolved, email)
    result_id = tuning.result_id(analysis, type(request).model_validate(resolved))
    return TaskHandle(task_id=task_id, result_id=result_id)
