"""Response models for the tuning API.

The *request* models are reused directly from ``bestee_compute.workflow.tuning``
(imported in each router) so the API and the Celery worker validate against a
single source of truth. Only the API-specific response models live here.
"""

from pydantic import BaseModel


class TaskHandle(BaseModel):
    """Returned when a job is accepted onto the queue."""

    task_id: str


class TaskStatus(BaseModel):
    """The state of a submitted job, polled by task id."""

    task_id: str
    state: str
    result_id: str | None = None
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


class ResultMeta(BaseModel):
    """Summary of a persisted tuning result, returned by the listing endpoint."""

    result_id: str
    analysis_type: str
    created_at: str  # ISO UTC datetime string


class SearchResults(BaseModel):
    """Search terms matching a query, ranked with prefix matches first."""

    query: str
    count: int
    results: list[str]


class SicCode(BaseModel):
    """A single SIC industry code and its title."""

    sic_code: str
    industry_title: str


class SicCodeList(BaseModel):
    """The full list of SIC industry codes."""

    count: int
    codes: list[SicCode]


class SicTickers(BaseModel):
    """The ticker symbols classified under a SIC code."""

    sic_code: str
    count: int
    tickers: list[str]
