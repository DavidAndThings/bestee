"""Poll a single job by task id, or list all jobs in the result backend."""

from typing import Any

from fastapi import APIRouter, Query

from celery_client import get_result
from redis_client import get_task_metas, list_all_task_ids
from schemas import JobsPage, TaskStatus

router = APIRouter(prefix="/jobs", tags=["jobs"])


def _meta_to_status(meta: dict[str, Any]) -> TaskStatus:
    """Convert a raw Celery result-backend entry to a :class:`TaskStatus`."""
    task_id: str = meta.get("task_id", "")
    state: str = meta.get("status", "UNKNOWN")
    if state == "SUCCESS":
        return TaskStatus(task_id=task_id, state=state, result=meta.get("result"))
    if state == "FAILURE":
        raw = meta.get("result", {})
        if isinstance(raw, dict):
            exc_type = raw.get("exc_type", "Error")
            exc_message = raw.get("exc_message", [])
            error = f"{exc_type}: {', '.join(str(m) for m in exc_message)}"
        else:
            error = str(raw)
        return TaskStatus(task_id=task_id, state=state, error=error)
    return TaskStatus(task_id=task_id, state=state)


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
    items = [_meta_to_status(m) for m in metas]
    return JobsPage(total=total, offset=offset, limit=limit, items=items)


@router.get("/{task_id}", response_model=TaskStatus)
def get_job(task_id: str) -> TaskStatus:
    """Return a job's state, plus its result on success or error on failure."""
    result = get_result(task_id)
    state = result.state
    if state == "SUCCESS":
        return TaskStatus(task_id=task_id, state=state, result=result.result)
    if state == "FAILURE":
        return TaskStatus(task_id=task_id, state=state, error=str(result.result))
    return TaskStatus(task_id=task_id, state=state)
