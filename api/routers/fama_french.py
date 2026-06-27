"""Submit an auto-tuned Fama-French factor-model job."""

from fastapi import APIRouter

from celery_client import dispatch
from schemas import FamaFrenchRequest, TaskHandle

router = APIRouter(prefix="/tasks", tags=["fama-french"])


@router.post("/fama-french", response_model=TaskHandle, status_code=202)
def submit_fama_french(request: FamaFrenchRequest) -> TaskHandle:
    """Enqueue a Fama-French job and return its task id."""
    task_id = dispatch("tuning.optimize_fama_french", request.model_dump())
    return TaskHandle(task_id=task_id)
