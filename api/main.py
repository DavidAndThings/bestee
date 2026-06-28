"""FastAPI app exposing the bestee tuning queue over HTTP.

One ``POST /tasks/{analysis}`` endpoint enqueues each tuning optimizer onto the
shared Celery/Redis queue; ``GET /jobs/{task_id}`` polls a job's state and
result. The app is a thin producer -- it dispatches tasks by name and never
imports the analytics stack.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from bestee_compute.logging_config import configure_logging
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from kombu.exceptions import OperationalError
from redis.exceptions import RedisError

from auth import require_auth
from routers import clustering, fama_french, jobs, regime, results, rrg


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Persist logs to ``LOG_DIR`` (stdout + file) for the server's lifetime."""
    configure_logging("api")
    yield


app = FastAPI(title="bestee tuning API", version="0.1.0", lifespan=lifespan)

_auth = [Depends(require_auth)]

app.include_router(clustering.router, dependencies=_auth)
app.include_router(regime.router, dependencies=_auth)
app.include_router(fama_french.router, dependencies=_auth)
app.include_router(rrg.router, dependencies=_auth)
app.include_router(jobs.router, dependencies=_auth)
app.include_router(results.router, dependencies=_auth)


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.exception_handler(OperationalError)
async def handle_broker_unavailable(
    request: Request, exc: OperationalError
) -> JSONResponse:
    """Return 503 when the Celery broker cannot be reached."""
    return JSONResponse(
        status_code=503,
        content={"detail": "Task queue is unavailable."},
    )


@app.exception_handler(RedisError)
async def handle_redis_unavailable(request: Request, exc: RedisError) -> JSONResponse:
    """Return 503 when the Redis result backend cannot be reached."""
    return JSONResponse(
        status_code=503,
        content={"detail": "Result backend is unavailable."},
    )
