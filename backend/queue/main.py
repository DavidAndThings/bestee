import os
from datetime import UTC, datetime
from typing import Any

from bestee_compute.logging_config import configure_logging
from celery import Celery
from celery.signals import setup_logging, task_prerun
from dotenv import load_dotenv

# Importing ``audit`` registers its task signal handlers (the audit trail).
import audit  # noqa: E402, F401  (side-effect import; must follow celery import)


def _with_ssl_cert_reqs(url: str | None) -> str | None:
    """Ensure a TLS Redis URL (``rediss://``) carries an ``ssl_cert_reqs`` param.

    Celery's Redis result backend rejects a ``rediss://`` URL that omits
    ``ssl_cert_reqs``.  Managed Redis (e.g. DigitalOcean) uses ``rediss://``;
    default to ``CERT_NONE`` (encrypted, no certificate verification -- matching
    kombu's broker fallback) unless the URL already sets it.
    """
    if url and url.startswith("rediss://") and "ssl_cert_reqs" not in url:
        separator = "&" if "?" in url else "?"
        return f"{url}{separator}ssl_cert_reqs=CERT_NONE"
    return url


def get_celery_app() -> Celery:
    """Build and configure the Celery application.

    Reads the Redis broker / result-backend URL from ``DO_REDIS_CONNECTION``
    (loaded from a local ``.env`` if present) and registers the task module.
    Serialization is pinned to JSON to match the JSON-serializable task results,
    and the worker is tuned for the long-running tuning jobs in :mod:`tasks`.
    """
    load_dotenv()
    redis_connection = _with_ssl_cert_reqs(os.environ.get("DO_REDIS_CONNECTION"))
    app = Celery(
        "queue",
        broker=redis_connection,
        backend=redis_connection,
        include=["tasks"],
    )
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        # The tuning jobs are long; ack after completion so a crashed worker's
        # task is redelivered, and prefetch one at a time for fair dispatch.
        task_acks_late=True,
        task_track_started=True,
        worker_prefetch_multiplier=1,
    )
    return app


app = get_celery_app()


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
