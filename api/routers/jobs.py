"""Poll a submitted job by its Celery task id."""

from fastapi import APIRouter

from celery_client import get_result
from schemas import TaskStatus

router = APIRouter(prefix="/jobs", tags=["jobs"])


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
