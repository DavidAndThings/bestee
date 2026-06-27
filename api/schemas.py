"""Response models for the tuning API.

The *request* models are reused directly from ``bestee_compute.workflow.tuning``
(imported in each router) so the API and the Celery worker validate against a
single source of truth. Only the API-specific response models live here.
"""

from typing import Any

from pydantic import BaseModel


class TaskHandle(BaseModel):
    """Returned when a job is accepted onto the queue."""

    task_id: str


class TaskStatus(BaseModel):
    """The state of a submitted job, polled by task id."""

    task_id: str
    state: str
    result: dict[str, Any] | None = None
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_seconds: float | None = None


class JobsPage(BaseModel):
    """A paginated slice of all jobs in the result backend."""

    total: int
    offset: int
    limit: int
    items: list[TaskStatus]
