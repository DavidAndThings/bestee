# bestee-tasking

Shared Celery wiring for the bestee backend. Both the FastAPI producer
(`api/`) and the Celery worker (`queue/`) build their Celery app from this one
factory so they can never drift on the broker URL, result backend, or
serialization, and so the `celery[redis]` driver is declared in a single place.

```python
from bestee_tasking import build_celery_app

# Producer (api):
celery_app = build_celery_app("api")

# Worker (queue):
app = build_celery_app(
    "queue",
    include=["tasks"],
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)
```

`build_celery_app` reads `DO_REDIS_CONNECTION` (loading a local `.env` if
present) and, for TLS (`rediss://`) managed Redis, appends
`ssl_cert_reqs=CERT_NONE` when the URL doesn't already set it -- Celery's Redis
result backend otherwise refuses such URLs. Override by putting your own
`ssl_cert_reqs` (e.g. `CERT_REQUIRED` with `ssl_ca_certs`) in the URL.
