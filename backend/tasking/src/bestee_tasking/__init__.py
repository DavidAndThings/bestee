"""Shared Celery wiring for the bestee producer (api) and worker (queue).

The FastAPI producer and the Celery worker must agree on the broker, result
backend, and serialization, so both build their Celery app from the single
:func:`build_celery_app` factory here. Centralizing it also makes this package
the one place that declares the ``celery[redis]`` dependency and handles the
``rediss://`` (managed/TLS Redis) connection quirks.
"""

import os
from typing import Any

from celery import Celery
from dotenv import load_dotenv

__all__ = ["build_celery_app", "redis_url_with_ssl_cert_reqs"]


def redis_url_with_ssl_cert_reqs(url: str | None) -> str | None:
    """Ensure a TLS Redis URL (``rediss://``) carries an ``ssl_cert_reqs`` param.

    Celery's Redis result backend rejects a ``rediss://`` URL that omits
    ``ssl_cert_reqs``. Managed Redis (e.g. DigitalOcean) uses ``rediss://``;
    default to ``CERT_NONE`` (encrypted, no certificate verification -- matching
    kombu's broker fallback) unless the URL already sets it.
    """
    if url and url.startswith("rediss://") and "ssl_cert_reqs" not in url:
        separator = "&" if "?" in url else "?"
        return f"{url}{separator}ssl_cert_reqs=CERT_NONE"
    return url


def build_celery_app(
    name: str, *, include: list[str] | None = None, **conf: Any
) -> Celery:
    """Build a Celery app bound to ``DO_REDIS_CONNECTION`` (broker + backend).

    JSON serialization is pinned so the producer and worker interoperate; extra
    *conf* extends/overrides it (e.g. the worker's acks-late / prefetch tuning).
    *include* names the task modules the worker should import. A local ``.env``
    is loaded if present.

    Args:
        name: The Celery application name.
        include: Task modules to import (worker side), or ``None``.
        **conf: Additional ``app.conf`` settings merged over the JSON defaults.

    Returns:
        A configured :class:`celery.Celery` instance.
    """
    load_dotenv()
    url = redis_url_with_ssl_cert_reqs(os.environ.get("DO_REDIS_CONNECTION"))
    app = Celery(name, broker=url, backend=url, include=include)
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        **conf,
    )
    return app
