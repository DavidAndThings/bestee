import os

from celery import Celery
from dotenv import load_dotenv


def get_celery_app() -> Celery:
    """Build and configure the Celery application.

    Reads the Redis broker / result-backend URL from ``DO_REDIS_CONNECTION``
    (loaded from a local ``.env`` if present) and registers the task module.
    Serialization is pinned to JSON to match the JSON-serializable task results,
    and the worker is tuned for the long-running tuning jobs in :mod:`tasks`.
    """
    load_dotenv()
    redis_connection = os.environ.get("DO_REDIS_CONNECTION")
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
        worker_prefetch_multiplier=1,
    )
    return app


app = get_celery_app()
