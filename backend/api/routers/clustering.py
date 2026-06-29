"""Submit an auto-tuned spectral clustering job."""

from typing import Annotated

from bestee_compute.workflow.tuning import ClusteringRequest
from fastapi import APIRouter, Depends

from auth import get_user_email, get_user_id
from schemas import TaskHandle
from submit import enqueue

router = APIRouter(prefix="/tasks", tags=["clustering"])


@router.post("/clustering", response_model=TaskHandle, status_code=202)
def submit_clustering(
    request: ClusteringRequest,
    email: Annotated[str | None, Depends(get_user_email)],
    user_id: Annotated[str, Depends(get_user_id)],
) -> TaskHandle:
    """Enqueue a clustering job and return its task id."""
    return enqueue("clustering", "tuning.optimize_clustering", request, email, user_id)
