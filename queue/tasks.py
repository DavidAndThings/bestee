"""Celery tasks -- one per ``*Request`` model in ``bestee_compute.workflow.tuning``.

Each task validates the incoming payload, runs the matching ``optimize_*`` grid
search, persists the full result to disk, then returns only the stable
``result_id``.  Callers retrieve the full result through the
``GET /results/{result_id}`` API endpoint.

Workers must provide the relevant API keys (``MASSIVE_API_KEY``, and
``FRED_API_KEY`` for Fama-French factors) and ``RESULTS_DIR`` in their
environment.
"""

import os
from pathlib import Path
from typing import Any

from bestee_compute.workflow import tuning
from celery import shared_task
from dotenv import load_dotenv

load_dotenv()
_RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "./results"))


@shared_task(name="tuning.optimize_clustering")
def optimize_clustering(request: dict[str, Any]) -> dict[str, str]:
    """Auto-tuned spectral clustering for a :class:`ClusteringRequest` payload."""
    req = tuning.ClusteringRequest.model_validate(request)
    result = tuning.optimize_clustering(req)
    rid = tuning.result_id("clustering", req)
    result.save(_RESULTS_DIR / rid)
    return {"result_id": rid}


@shared_task(name="tuning.optimize_regime")
def optimize_regime(request: dict[str, Any]) -> dict[str, str]:
    """Auto-tuned per-asset regime detection for a :class:`RegimeRequest` payload."""
    req = tuning.RegimeRequest.model_validate(request)
    result = tuning.optimize_regime(req)
    rid = tuning.result_id("regime", req)
    result.save(_RESULTS_DIR / rid)
    return {"result_id": rid}


@shared_task(name="tuning.optimize_fama_french")
def optimize_fama_french(request: dict[str, Any]) -> dict[str, str]:
    """Auto-tuned Fama-French fit for a :class:`FamaFrenchRequest` payload."""
    req = tuning.FamaFrenchRequest.model_validate(request)
    result = tuning.optimize_fama_french(req)
    rid = tuning.result_id("fama_french", req)
    result.save(_RESULTS_DIR / rid)
    return {"result_id": rid}


@shared_task(name="tuning.optimize_rrg")
def optimize_rrg(request: dict[str, Any]) -> dict[str, str]:
    """Auto-tuned relative-rotation graph for an :class:`RRGRequest` payload."""
    req = tuning.RRGRequest.model_validate(request)
    result = tuning.optimize_rrg(req)
    rid = tuning.result_id("rrg", req)
    result.save(_RESULTS_DIR / rid)
    return {"result_id": rid}
