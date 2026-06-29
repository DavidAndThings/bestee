"""Submit an auto-tuned Fama-French factor-model job."""

from typing import Annotated

from bestee_compute.workflow.tuning import FamaFrenchRequest
from fastapi import APIRouter, Depends

from auth import get_user_email, get_user_id
from schemas import TaskHandle
from submit import enqueue

router = APIRouter(prefix="/tasks", tags=["fama-french"])


@router.post("/fama-french", response_model=TaskHandle, status_code=202)
def submit_fama_french(
    request: FamaFrenchRequest,
    email: Annotated[str | None, Depends(get_user_email)],
    user_id: Annotated[str, Depends(get_user_id)],
) -> TaskHandle:
    """Enqueue a Fama-French job and return its task id."""
    return enqueue(
        "fama_french", "tuning.optimize_fama_french", request, email, user_id
    )
