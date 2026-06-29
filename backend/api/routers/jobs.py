"""List a user's jobs, or poll a single job by task id.

A job's inputs (analysis, payload, submit time) come from the per-user record
written at dispatch time (see :func:`redis_client.record_job`); its live state,
timing, and result/error come from the Celery result backend. Merging the two
lets the listing show queued jobs immediately and survive across devices.
"""

import datetime
from typing import Annotated, Any

from celery import states
from fastapi import APIRouter, Depends, Query

from auth import get_user_id
from celery_client import get_result
from redis_client import (
    get_job_record,
    get_task_metas,
    get_task_start_times,
    list_user_jobs,
)
from schemas import JobsPage, TaskStatus

router = APIRouter(prefix="/jobs", tags=["jobs"])


def _compute_elapsed(
    started_at: str | None, finished_at: str | None, state: str
) -> float | None:
    """Wall-clock duration in seconds, or ``None`` when start time is unknown.

    For completed tasks the duration is ``finished_at - started_at``.
    For tasks still running the duration is ``now - started_at``.
    """
    if not started_at:
        return None
    try:
        start = datetime.datetime.fromisoformat(started_at)
        if finished_at:
            end = datetime.datetime.fromisoformat(finished_at)
        elif state not in states.READY_STATES:
            end = datetime.datetime.now(datetime.UTC)
        else:
            return None
        # Normalise both to offset-aware before subtracting.
        if start.tzinfo is None:
            start = start.replace(tzinfo=datetime.UTC)
        if end.tzinfo is None:
            end = end.replace(tzinfo=datetime.UTC)
        return max(0.0, (end - start).total_seconds())
    except ValueError:
        return None


def _result_id(task_result: Any) -> str | None:
    """Extract ``result_id`` from a task return dict, or ``None`` if absent."""
    return task_result.get("result_id") if isinstance(task_result, dict) else None


def _format_error(raw: Any) -> str:
    """A readable error string from a Celery FAILURE payload."""
    if isinstance(raw, dict):
        exc_type = raw.get("exc_type", "Error")
        exc_message = raw.get("exc_message", [])
        return f"{exc_type}: {', '.join(str(m) for m in exc_message)}"
    return str(raw)


def _status_from(
    record: dict[str, Any],
    meta: dict[str, Any] | None,
    started_at: str | None,
) -> TaskStatus:
    """Merge a stored job *record* with its live Celery *meta* into a status."""
    task_id = record.get("task_id", "")
    state = (meta or {}).get("status", "PENDING")
    finished_at = (meta or {}).get("date_done")
    elapsed = _compute_elapsed(started_at, finished_at, state)
    # The recorded id is the deterministic one, known for every state; on success
    # prefer the worker's returned id (identical, but authoritative).
    result_id = record.get("result_id")
    error: str | None = None
    if meta and state == "SUCCESS":
        result_id = _result_id(meta.get("result")) or result_id
    elif meta and state == "FAILURE":
        error = _format_error(meta.get("result"))
    return TaskStatus(
        task_id=task_id,
        state=state,
        result_id=result_id,
        analysis=record.get("analysis"),
        payload=record.get("payload"),
        created_at=record.get("created_at"),
        error=error,
        started_at=started_at,
        finished_at=finished_at,
        elapsed_seconds=elapsed,
    )


@router.get("", response_model=JobsPage)
def list_jobs(
    user_id: Annotated[str, Depends(get_user_id)],
    offset: int = Query(default=0, ge=0, description="Number of jobs to skip."),
    limit: int = Query(default=50, ge=1, le=100, description="Max jobs to return."),
) -> JobsPage:
    """List the authenticated user's jobs, newest first, with pagination."""
    total, records = list_user_jobs(user_id, offset, limit)
    task_ids = [r["task_id"] for r in records]
    metas = {m.get("task_id"): m for m in get_task_metas(task_ids)}
    starts = get_task_start_times(task_ids)
    items = [
        _status_from(r, metas.get(r["task_id"]), starts.get(r["task_id"]))
        for r in records
    ]
    return JobsPage(total=total, offset=offset, limit=limit, items=items)


@router.get("/{task_id}", response_model=TaskStatus)
def get_job(task_id: str) -> TaskStatus:
    """Return a job's state, timing, recorded inputs, and result or error."""
    record = get_job_record(task_id) or {"task_id": task_id}
    result = get_result(task_id)
    state = result.state
    started_at = get_task_start_times([task_id]).get(task_id)
    # Celery's date_done property calls isoparse() and returns datetime | None.
    date_done = result.date_done
    if isinstance(date_done, str):
        finished_at: str | None = date_done
    elif date_done is not None:
        finished_at = date_done.isoformat()
    else:
        finished_at = None
    elapsed = _compute_elapsed(started_at, finished_at, state)
    result_id = record.get("result_id")
    error: str | None = None
    if state == "SUCCESS":
        result_id = _result_id(result.result) or result_id
    elif state == "FAILURE":
        error = str(result.result)
    return TaskStatus(
        task_id=task_id,
        state=state,
        result_id=result_id,
        analysis=record.get("analysis"),
        payload=record.get("payload"),
        created_at=record.get("created_at"),
        error=error,
        started_at=started_at,
        finished_at=finished_at,
        elapsed_seconds=elapsed,
    )
