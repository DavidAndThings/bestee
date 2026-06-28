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
from dataclasses import asdict
from typing import Any

import polars as pl
import resend
from bestee_compute.workflow import tuning
from celery import shared_task
from dotenv import load_dotenv

load_dotenv()
resend.api_key = os.environ.get("RESEND_API_KEY")


@shared_task(name="tuning.optimize_clustering", bind=True)
def optimize_clustering(self, request: dict[str, Any]) -> dict[str, Any]:
    """Auto-tuned spectral clustering for a :class:`ClusteringRequest` payload."""

    result = tuning.optimize_clustering(
        tuning.ClusteringRequest.model_validate(request)
    )
    send_completion_confirmation(
        self.request.id,
    )
    return asdict(result)


@shared_task(name="tuning.optimize_regime", bind=True)
def optimize_regime(self, request: dict[str, Any]) -> dict[str, Any]:
    """Auto-tuned per-asset regime detection for a :class:`RegimeRequest` payload."""

    result = tuning.optimize_regime(tuning.RegimeRequest.model_validate(request))
    # The fitted ``RegimeResult`` objects carry DataFrames / arrays, so only the
    # serializable summary (current regime per asset + chosen params) is returned.
    return {
        "labels": dict(result.labels),
        "residualization_window": result.residualization_window,
        "normalization_window": result.normalization_window,
        "hmm_lag": result.hmm_lag,
        "mean_bic": result.mean_bic,
    }


@shared_task(name="tuning.optimize_fama_french", bind=True)
def optimize_fama_french(self, request: dict[str, Any]) -> dict[str, Any]:
    """Auto-tuned Fama-French fit for a :class:`FamaFrenchRequest` payload."""

    result = tuning.optimize_fama_french(
        tuning.FamaFrenchRequest.model_validate(request)
    )
    return {
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

    result = tuning.optimize_rrg(tuning.RRGRequest.model_validate(request))
    # Stringify the Timestamp column so the frames are plain JSON records.
    strength = result.relative_strength.with_columns(
        pl.col("Timestamp").cast(pl.String)
    )
    momentum = result.relative_momentum.with_columns(
        pl.col("Timestamp").cast(pl.String)
    )
    return {
        "normalization_window": result.normalization_window,
        "momentum_lookback": result.momentum_lookback,
        "smoothing_span": result.smoothing_span,
        "signal_to_noise": result.signal_to_noise,
        "relative_strength": strength.to_dicts(),
        "relative_momentum": momentum.to_dicts(),
    }


def send_completion_confirmation(task_id: str) -> None:
    resend.Emails.send(
        {
            "from": "onboarding@resend.dev",
            "to": "easyd93@proton.me",
            "subject": "Bestee Task Completed",
            "html": "<p>Congrats on sending your <strong>first email</strong>!</p>",
        }
    )
