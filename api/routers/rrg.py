"""Submit an auto-tuned relative-rotation-graph job."""

from typing import Annotated

from bestee_compute.workflow.tuning import RRGRequest
from fastapi import APIRouter, Depends

from auth import get_user_email
from celery_client import dispatch
from resolve import resolve_request
from schemas import TaskHandle

router = APIRouter(prefix="/tasks", tags=["rrg"])


@router.post("/rrg", response_model=TaskHandle, status_code=202)
def submit_rrg(
    request: RRGRequest,
    email: Annotated[str | None, Depends(get_user_email)],
) -> TaskHandle:
    """Enqueue an RRG job and return its task id."""
    task_id = dispatch("tuning.optimize_rrg", resolve_request(request), email)
    return TaskHandle(task_id=task_id)
