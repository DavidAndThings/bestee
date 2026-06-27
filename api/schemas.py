"""Request and response models for the tuning API.

The request models mirror the ``*Request`` models in
``bestee_compute.workflow.tuning`` so a validated body can be handed straight to
the matching Celery task. They are duplicated here (rather than imported) to
keep the API a lightweight producer with no dependency on the analytics stack.
"""

from typing import Any, Literal, Self

from pydantic import BaseModel, Field, model_validator

ReferenceType = Literal["mean", "ticker"]


class ClusteringRequest(BaseModel):
    """Inputs for an auto-tuned spectral clustering job."""

    tickers: list[str] = Field(min_length=2)
    start_date: str
    end_date: str
    min_num_clusters: int = Field(default=2, ge=2)
    max_num_clusters: int = Field(default=50, ge=2)

    @model_validator(mode="after")
    def _check_cluster_bounds(self) -> Self:
        if self.min_num_clusters > self.max_num_clusters:
            raise ValueError("min_num_clusters cannot exceed max_num_clusters")
        return self


class RegimeRequest(BaseModel):
    """Inputs for an auto-tuned per-asset regime-detection job."""

    tickers: list[str] = Field(min_length=1)
    start_date: str
    end_date: str


class FamaFrenchRequest(BaseModel):
    """Inputs for an auto-tuned Fama-French factor-model job."""

    tickers: list[str] = Field(min_length=1)
    start_date: str
    end_date: str


class RRGRequest(BaseModel):
    """Inputs for an auto-tuned relative-rotation-graph job."""

    tickers: list[str] = Field(min_length=1)
    start_date: str
    end_date: str
    reference_type: ReferenceType = "mean"
    benchmark_ticker: str | None = None

    @model_validator(mode="after")
    def _require_benchmark_for_ticker_reference(self) -> Self:
        if self.reference_type == "ticker" and self.benchmark_ticker is None:
            raise ValueError(
                "benchmark_ticker is required when reference_type is 'ticker'"
            )
        return self


class TaskHandle(BaseModel):
    """Returned when a job is accepted onto the queue."""

    task_id: str


class TaskStatus(BaseModel):
    """The state of a submitted job, polled by task id."""

    task_id: str
    state: str
    result: dict[str, Any] | None = None
    error: str | None = None
