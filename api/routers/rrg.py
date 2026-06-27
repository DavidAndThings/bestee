"""Submit an auto-tuned relative-rotation-graph job."""

from fastapi import APIRouter

from celery_client import dispatch
from schemas import RRGRequest, TaskHandle

router = APIRouter(prefix="/tasks", tags=["rrg"])


@router.post("/rrg", response_model=TaskHandle, status_code=202)
def submit_rrg(request: RRGRequest) -> TaskHandle:
    """Enqueue an RRG job and return its task id."""
    task_id = dispatch("tuning.optimize_rrg", request.model_dump())
    return TaskHandle(task_id=task_id)
