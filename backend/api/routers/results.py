"""Query persisted tuning results by their stable result_id."""

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bestee_compute.workflow.tuning import (
    ClusteringTuning,
    FamaFrenchTuning,
    RegimeTuning,
    RRGTuning,
    frame_to_table,
)
from fastapi import APIRouter, HTTPException

from schemas import ResultMeta, ResultTable

router = APIRouter(prefix="/results", tags=["results"])

_LOADERS = {
    "clustering": ClusteringTuning.load,
    "fama_french": FamaFrenchTuning.load,
    "regime": RegimeTuning.load,
    "rrg": RRGTuning.load,
}

# Per-analysis aspects: the method on each tuning object that builds the table.
_ASPECTS: dict[str, dict[str, str]] = {
    "clustering": {"cluster_label": "cluster_label_table"},
    "fama_french": {"ff_residuals": "ff_residuals_table"},
    "regime": {"regime_label": "regime_label_table"},
    "rrg": {"coordinates": "coordinates_table"},
}


def _get_results_dir() -> Path:
    """Read RESULTS_DIR from the environment on each call so tests can patch it."""
    return Path(os.environ.get("RESULTS_DIR", "./results"))


@router.get("", response_model=list[ResultMeta])
def list_results() -> list[ResultMeta]:
    """List all stored tuning results, newest first."""
    results_dir = _get_results_dir()
    if not results_dir.exists():
        return []
    items: list[ResultMeta] = []
    for p in results_dir.iterdir():
        if not p.is_dir() or not (p / "metadata.json").exists():
            continue
        analysis_type = p.name.rsplit("_", 1)[0]
        if analysis_type not in _LOADERS:
            continue
        created_at = datetime.fromtimestamp(
            (p / "metadata.json").stat().st_mtime, tz=UTC
        ).isoformat()
        items.append(
            ResultMeta(
                result_id=p.name,
                analysis_type=analysis_type,
                created_at=created_at,
            )
        )
    return sorted(items, key=lambda r: r.created_at, reverse=True)


@router.get("/{result_id}")
def get_result(result_id: str) -> dict[str, Any]:
    """Load and return the full tuning result for *result_id*."""
    results_dir = _get_results_dir()
    path = results_dir / result_id
    if not path.exists() or not (path / "metadata.json").exists():
        raise HTTPException(status_code=404, detail=f"Result {result_id!r} not found.")
    analysis_type = result_id.rsplit("_", 1)[0]
    loader = _LOADERS.get(analysis_type)
    if loader is None:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown analysis type {analysis_type!r}.",
        )
    try:
        tuning_obj = loader(path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"result_id": result_id, **tuning_obj.to_dict()}


@router.get("/{result_id}/{aspect_name}", response_model=ResultTable)
def get_result_aspect(result_id: str, aspect_name: str) -> ResultTable:
    """Return one analysis-specific *aspect* of a stored result, as a table.

    The available aspects depend on the analysis that produced the result --
    e.g. ``cluster_label`` (clustering), ``coordinates`` (RRG), ``regime_label``
    (regime), ``ff_residuals`` (Fama-French).
    """
    results_dir = _get_results_dir()
    path = results_dir / result_id
    if not path.exists() or not (path / "metadata.json").exists():
        raise HTTPException(status_code=404, detail=f"Result {result_id!r} not found.")
    analysis_type = result_id.rsplit("_", 1)[0]
    loader = _LOADERS.get(analysis_type)
    if loader is None:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown analysis type {analysis_type!r}.",
        )
    method_name = _ASPECTS.get(analysis_type, {}).get(aspect_name)
    if method_name is None:
        available = sorted(_ASPECTS.get(analysis_type, {}))
        raise HTTPException(
            status_code=404,
            detail=(
                f"Aspect {aspect_name!r} is not available for {analysis_type!r} "
                f"results; available: {available}."
            ),
        )
    try:
        tuning_obj = loader(path)
        frame = getattr(tuning_obj, method_name)()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    table = frame_to_table(frame)
    return ResultTable(
        result_id=result_id,
        aspect=aspect_name,
        columns=table["columns"],
        rows=table["rows"],
    )
