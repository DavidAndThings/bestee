"""Submit an auto-tuned per-asset regime-detection job."""

from fastapi import APIRouter

from celery_client import dispatch
from schemas import RegimeRequest, TaskHandle

router = APIRouter(prefix="/tasks", tags=["regime"])


@router.post("/regime", response_model=TaskHandle, status_code=202)
def submit_regime(request: RegimeRequest) -> TaskHandle:
    """Enqueue a regime-detection job and return its task id."""
    task_id = dispatch("tuning.optimize_regime", request.model_dump())
    return TaskHandle(task_id=task_id)
