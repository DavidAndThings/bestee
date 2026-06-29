"""Submit an auto-tuned Fama-French factor-model job."""

from typing import Annotated

from bestee_compute.workflow.tuning import FamaFrenchRequest
from fastapi import APIRouter, Depends

from auth import get_user_email
from celery_client import dispatch
from resolve import resolve_request
from schemas import TaskHandle

router = APIRouter(prefix="/tasks", tags=["fama-french"])


@router.post("/fama-french", response_model=TaskHandle, status_code=202)
def submit_fama_french(
    request: FamaFrenchRequest,
    email: Annotated[str | None, Depends(get_user_email)],
) -> TaskHandle:
    """Enqueue a Fama-French job and return its task id."""
    task_id = dispatch("tuning.optimize_fama_french", resolve_request(request), email)
    return TaskHandle(task_id=task_id)
