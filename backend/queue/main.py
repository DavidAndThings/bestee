from datetime import UTC, datetime
from typing import Any

from bestee_compute.logging_config import configure_logging
from bestee_tasking import build_celery_app
from celery.signals import setup_logging, task_prerun

# Importing ``audit`` registers its task signal handlers (the audit trail).
import audit  # noqa: E402, F401  (side-effect import; must follow celery import)

# Serialization is pinned to JSON by the shared factory; the worker tuning here
# is for the long-running tuning jobs in :mod:`tasks`: ack after completion (so
# a crashed worker's task is redelivered) and prefetch one at a time for fair
# dispatch.
app = build_celery_app(
    "queue",
    include=["tasks"],
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    task_track_started=True,
    worker_prefetch_multiplier=1,
)


@setup_logging.connect
def _configure_logging(**_: Any) -> None:
    """Own the worker's logging so task logs persist to ``LOG_DIR``.

    Connecting a receiver to ``setup_logging`` disables Celery's own logging
    hijack, letting :func:`configure_logging` install the stdout + file handlers
    on the root logger.  Forked children inherit the configuration.
    """
    configure_logging("queue")


@task_prerun.connect
def _record_task_start(
    sender: object = None, task_id: str | None = None, **kwargs: object
) -> None:
    """Write the task start time to Redis so the /jobs API can display it."""
    if task_id is None:
        return
    started_at = datetime.now(UTC).isoformat()
    try:
        app.backend.client.set(  # type: ignore[union-attr]
            f"celery-task-started-{task_id}", started_at, ex=86400
        )
    except Exception:
        pass
