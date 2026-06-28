"""Submit an auto-tuned per-asset regime-detection job."""

from typing import Annotated

from bestee_compute.workflow.tuning import RegimeRequest
from fastapi import APIRouter, Depends

from auth import get_user_email
from celery_client import dispatch
from schemas import TaskHandle

router = APIRouter(prefix="/tasks", tags=["regime"])


@router.post("/regime", response_model=TaskHandle, status_code=202)
def submit_regime(
    request: RegimeRequest,
    email: Annotated[str | None, Depends(get_user_email)],
) -> TaskHandle:
    """Enqueue a regime-detection job and return its task id."""
    task_id = dispatch("tuning.optimize_regime", request.model_dump(), email)
    return TaskHandle(task_id=task_id)
