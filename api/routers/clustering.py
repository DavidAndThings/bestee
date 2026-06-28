"""Submit an auto-tuned spectral clustering job."""

from typing import Annotated

from bestee_compute.workflow.tuning import ClusteringRequest
from fastapi import APIRouter, Depends

from auth import get_user_email
from celery_client import dispatch
from schemas import TaskHandle

router = APIRouter(prefix="/tasks", tags=["clustering"])


@router.post("/clustering", response_model=TaskHandle, status_code=202)
def submit_clustering(
    request: ClusteringRequest,
    email: Annotated[str | None, Depends(get_user_email)],
) -> TaskHandle:
    """Enqueue a clustering job and return its task id."""
    task_id = dispatch("tuning.optimize_clustering", request.model_dump(), email)
    return TaskHandle(task_id=task_id)
