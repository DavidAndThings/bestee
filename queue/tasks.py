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

from typing import Any

from celery import shared_task


@shared_task(name="tuning.optimize_clustering")
def optimize_clustering(request: dict[str, Any]) -> dict[str, Any]:
    """Auto-tuned spectral clustering for a :class:`ClusteringRequest` payload."""
    from dataclasses import asdict

    from bestee_compute.workflow import tuning

    result = tuning.optimize_clustering(
        tuning.ClusteringRequest.model_validate(request)
    )
    return asdict(result)


@shared_task(name="tuning.optimize_regime")
def optimize_regime(request: dict[str, Any]) -> dict[str, Any]:
    """Auto-tuned per-asset regime detection for a :class:`RegimeRequest` payload."""
    from bestee_compute.workflow import tuning

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


@shared_task(name="tuning.optimize_fama_french")
def optimize_fama_french(request: dict[str, Any]) -> dict[str, Any]:
    """Auto-tuned Fama-French fit for a :class:`FamaFrenchRequest` payload."""
    from bestee_compute.workflow import tuning

    result = tuning.optimize_fama_french(
        tuning.FamaFrenchRequest.model_validate(request)
    )
    return {
        "factors_to_use": result.factors_to_use,
        "mean_adjusted_r_squared": result.mean_adjusted_r_squared,
        "results": {
            ticker: model.model_dump(mode="json")
            for ticker, model in result.results.items()
        },
    }


@shared_task(name="tuning.optimize_rrg")
def optimize_rrg(request: dict[str, Any]) -> dict[str, Any]:
    """Auto-tuned relative-rotation graph for an :class:`RRGRequest` payload."""
    import polars as pl
    from bestee_compute.workflow import tuning

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
