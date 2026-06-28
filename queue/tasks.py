"""Celery tasks -- one per ``*Request`` model in ``bestee_compute.workflow.tuning``.

Each task validates the incoming request payload into its pydantic ``*Request``
model, runs the matching ``optimize_*`` grid search in the compute package, and
returns a JSON-serializable summary (the chosen parameters, the score, and the
analysis output).

The compute import is deferred into each task body so this module imports
without the heavy analytics stack present (Celery only pays for it when a task
runs). Tasks fetch live market data through ``optimize_*``, so the worker
environment must provide the relevant API keys (``MASSIVE_API_KEY``, and
``FRED_API_KEY`` for the Fama-French factors).
"""

import os
from pathlib import Path
from typing import Any

import resend
from bestee_compute.workflow import tuning
from celery import shared_task
from dotenv import load_dotenv

load_dotenv()
resend.api_key = os.environ.get("RESEND_API_KEY")
_RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "./results"))


@shared_task(name="tuning.optimize_clustering", bind=True)
def optimize_clustering(self, request: dict[str, Any]) -> dict[str, Any]:
    """Auto-tuned spectral clustering for a :class:`ClusteringRequest` payload."""
    req = tuning.ClusteringRequest.model_validate(request)
    result = tuning.optimize_clustering(req)
    rid = tuning.result_id("clustering", req)
    result.save(_RESULTS_DIR / rid)
    send_completion_confirmation(self.request.id)
    return {"result_id": rid, **result.to_dict()}


@shared_task(name="tuning.optimize_regime", bind=True)
def optimize_regime(self, request: dict[str, Any]) -> dict[str, Any]:
    """Auto-tuned per-asset regime detection for a :class:`RegimeRequest` payload."""
    req = tuning.RegimeRequest.model_validate(request)
    result = tuning.optimize_regime(req)
    rid = tuning.result_id("regime", req)
    result.save(_RESULTS_DIR / rid)
    # The fitted ``RegimeResult`` objects carry DataFrames / arrays, so only the
    # serializable summary (current regime per asset + chosen params) is returned.
    return {
        "result_id": rid,
        "labels": dict(result.labels),
        "residualization_window": result.residualization_window,
        "normalization_window": result.normalization_window,
        "hmm_lag": result.hmm_lag,
        "mean_bic": result.mean_bic,
    }


@shared_task(name="tuning.optimize_fama_french", bind=True)
def optimize_fama_french(self, request: dict[str, Any]) -> dict[str, Any]:
    """Auto-tuned Fama-French fit for a :class:`FamaFrenchRequest` payload."""
    req = tuning.FamaFrenchRequest.model_validate(request)
    result = tuning.optimize_fama_french(req)
    rid = tuning.result_id("fama_french", req)
    result.save(_RESULTS_DIR / rid)
    return {
        "result_id": rid,
        "factors_to_use": result.factors_to_use,
        "specifications": [spec.model_dump() for spec in result.specifications],
        "mean_adjusted_r_squared": result.mean_adjusted_r_squared,
        "results": {
            spec_name: {
                ticker: model.model_dump(mode="json")
                for ticker, model in spec_results.items()
            }
            for spec_name, spec_results in result.results.items()
        },
        "oos_residuals": result.oos_residuals.to_dicts(),
    }


@shared_task(name="tuning.optimize_rrg", bind=True)
def optimize_rrg(self, request: dict[str, Any]) -> dict[str, Any]:
    """Auto-tuned relative-rotation graph for an :class:`RRGRequest` payload."""
    req = tuning.RRGRequest.model_validate(request)
    result = tuning.optimize_rrg(req)
    rid = tuning.result_id("rrg", req)
    result.save(_RESULTS_DIR / rid)
    # result.to_dict() handles Timestamp serialization via _serialize_frame internally.
    return {"result_id": rid, **result.to_dict()}


def send_completion_confirmation(task_id: str) -> None:
    resend.Emails.send(
        {
            "from": "onboarding@resend.dev",
            "to": "easyd93@proton.me",
            "subject": "Bestee Task Completed",
            "html": "<p>Congrats on sending your <strong>first email</strong>!</p>",
        }
    )
