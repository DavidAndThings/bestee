"""Submit an auto-tuned relative-rotation-graph job."""

from typing import Annotated

from bestee_compute.workflow.tuning import RRGRequest
from fastapi import APIRouter, Depends

from auth import get_user_email, get_user_id
from schemas import TaskHandle
from submit import enqueue

router = APIRouter(prefix="/tasks", tags=["rrg"])


@router.post("/rrg", response_model=TaskHandle, status_code=202)
def submit_rrg(
    request: RRGRequest,
    email: Annotated[str | None, Depends(get_user_email)],
    user_id: Annotated[str, Depends(get_user_id)],
) -> TaskHandle:
    """Enqueue an RRG job and return its task id."""
    return enqueue("rrg", "tuning.optimize_rrg", request, email, user_id)
