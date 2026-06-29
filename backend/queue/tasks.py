"""Celery tasks -- one per ``*Request`` model in ``bestee_compute.workflow.tuning``.

Each task validates the incoming payload, runs the matching ``optimize_*`` grid
search, persists the full result to disk, then returns only the stable
``result_id``.  Callers retrieve the full result through the
``GET /results/{result_id}`` API endpoint.

On completion, if the requester's ``user_email`` is known (extracted from the
Clerk session token by the API), a Resend email with the task id and result id
is sent via :func:`send_completion_email`.

Workers must provide the relevant API keys (``MASSIVE_API_KEY``, and
``FRED_API_KEY`` for Fama-French factors), ``RESULTS_DIR``, and -- for the
completion email -- ``RESEND_API_KEY`` (plus an optional ``RESEND_FROM_EMAIL``
verified sender) in their environment.
"""

import logging
import os
from pathlib import Path
from typing import Any

import resend
from bestee_compute.workflow import tuning
from celery import Task, shared_task
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
_RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "./results"))
_EMAIL_FROM = os.environ.get("RESEND_FROM_EMAIL", "onboarding@resend.dev")


def send_completion_email(task_id: str, result_id: str, user_email: str | None) -> None:
    """Email *user_email* the task and result ids once a job completes.

    A no-op when there is no recipient or ``RESEND_API_KEY`` is unset.  Any send
    failure is logged and swallowed so a flaky mailer never fails the task.
    """
    if not user_email:
        return
    api_key = os.environ.get("RESEND_API_KEY")
    if not api_key:
        logger.warning("RESEND_API_KEY not set; skipping completion email.")
        return
    resend.api_key = api_key
    try:
        resend.Emails.send(
            {
                "from": _EMAIL_FROM,
                "to": user_email,
                "subject": "Your bestee analysis is ready",
                "html": (
                    "<p>Your analysis has finished.</p>"
                    f"<p><strong>Task id:</strong> {task_id}<br>"
                    f"<strong>Result id:</strong> {result_id}</p>"
                    f"<p>Fetch it at <code>GET /results/{result_id}</code>.</p>"
                ),
            }
        )
        logger.info("completion email sent to %s (result %s)", user_email, result_id)
    except Exception as exc:
        logger.warning("completion email to %s failed: %s", user_email, exc)


def _finalize(
    analysis: str,
    rid: str,
    result: Any,
    task_id: str,
    user_email: str | None,
) -> dict[str, str]:
    """Persist *result*, log attribution, email the requester, return the id."""
    result.save(_RESULTS_DIR / rid)
    logger.info("%s %s requested by %s", analysis, rid, user_email or "anonymous")
    send_completion_email(task_id, rid, user_email)
    return {"result_id": rid}


@shared_task(name="tuning.optimize_clustering", bind=True)
def optimize_clustering(
    self: Task, request: dict[str, Any], user_email: str | None = None
) -> dict[str, str]:
    """Auto-tuned spectral clustering for a :class:`ClusteringRequest` payload."""
    req = tuning.ClusteringRequest.model_validate(request)
    result = tuning.optimize_clustering(req)
    rid = tuning.result_id("clustering", req)
    return _finalize("clustering", rid, result, self.request.id, user_email)


@shared_task(name="tuning.optimize_regime", bind=True)
def optimize_regime(
    self: Task, request: dict[str, Any], user_email: str | None = None
) -> dict[str, str]:
    """Auto-tuned per-asset regime detection for a :class:`RegimeRequest` payload."""
    req = tuning.RegimeRequest.model_validate(request)
    result = tuning.optimize_regime(req)
    rid = tuning.result_id("regime", req)
    return _finalize("regime", rid, result, self.request.id, user_email)


@shared_task(name="tuning.optimize_fama_french", bind=True)
def optimize_fama_french(
    self: Task, request: dict[str, Any], user_email: str | None = None
) -> dict[str, str]:
    """Auto-tuned Fama-French fit for a :class:`FamaFrenchRequest` payload."""
    req = tuning.FamaFrenchRequest.model_validate(request)
    result = tuning.optimize_fama_french(req)
    rid = tuning.result_id("fama_french", req)
    return _finalize("fama_french", rid, result, self.request.id, user_email)


@shared_task(name="tuning.optimize_rrg", bind=True)
def optimize_rrg(
    self: Task, request: dict[str, Any], user_email: str | None = None
) -> dict[str, str]:
    """Auto-tuned relative-rotation graph for an :class:`RRGRequest` payload."""
    req = tuning.RRGRequest.model_validate(request)
    result = tuning.optimize_rrg(req)
    rid = tuning.result_id("rrg", req)
    return _finalize("rrg", rid, result, self.request.id, user_email)
