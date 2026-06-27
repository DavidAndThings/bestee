"""Poll a single job by task id, or list all jobs in the result backend."""

import datetime
from typing import Any

from celery import states
from fastapi import APIRouter, Query

from celery_client import get_result
from redis_client import get_task_metas, get_task_start_times, list_all_task_ids
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


def _meta_to_status(meta: dict[str, Any], started_at: str | None = None) -> TaskStatus:
    """Convert a raw Celery result-backend entry to a :class:`TaskStatus`."""
    task_id: str = meta.get("task_id", "")
    state: str = meta.get("status", "UNKNOWN")
    finished_at: str | None = meta.get("date_done")
    elapsed = _compute_elapsed(started_at, finished_at, state)

    if state == "SUCCESS":
        return TaskStatus(
            task_id=task_id,
            state=state,
            result=meta.get("result"),
            started_at=started_at,
            finished_at=finished_at,
            elapsed_seconds=elapsed,
        )
    if state == "FAILURE":
        raw = meta.get("result", {})
        if isinstance(raw, dict):
            exc_type = raw.get("exc_type", "Error")
            exc_message = raw.get("exc_message", [])
            error = f"{exc_type}: {', '.join(str(m) for m in exc_message)}"
        else:
            error = str(raw)
        return TaskStatus(
            task_id=task_id,
            state=state,
            error=error,
            started_at=started_at,
            finished_at=finished_at,
            elapsed_seconds=elapsed,
        )
    return TaskStatus(
        task_id=task_id,
        state=state,
        started_at=started_at,
        finished_at=finished_at,
        elapsed_seconds=elapsed,
    )


@router.get("", response_model=JobsPage)
def list_jobs(
    offset: int = Query(default=0, ge=0, description="Number of jobs to skip."),
    limit: int = Query(default=20, ge=1, le=100, description="Max jobs to return."),
) -> JobsPage:
    """List every job in the result backend, sorted by task id, with pagination."""
    all_ids = list_all_task_ids()
    total = len(all_ids)
    page_ids = all_ids[offset : offset + limit]
    metas = get_task_metas(page_ids)
    start_times = get_task_start_times(page_ids)
    items = [
        _meta_to_status(m, started_at=start_times.get(m.get("task_id", "")))
        for m in metas
    ]
    return JobsPage(total=total, offset=offset, limit=limit, items=items)


@router.get("/{task_id}", response_model=TaskStatus)
def get_job(task_id: str) -> TaskStatus:
    """Return a job's state, timing, and result or error when finished."""
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

    if state == "SUCCESS":
        return TaskStatus(
            task_id=task_id,
            state=state,
            result=result.result,
            started_at=started_at,
            finished_at=finished_at,
            elapsed_seconds=elapsed,
        )
    if state == "FAILURE":
        return TaskStatus(
            task_id=task_id,
            state=state,
            error=str(result.result),
            started_at=started_at,
            finished_at=finished_at,
            elapsed_seconds=elapsed,
        )
    return TaskStatus(
        task_id=task_id,
        state=state,
        started_at=started_at,
        finished_at=finished_at,
        elapsed_seconds=elapsed,
    )
