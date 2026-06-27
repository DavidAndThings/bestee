"""FastAPI app exposing the bestee tuning queue over HTTP.

One ``POST /tasks/{analysis}`` endpoint enqueues each tuning optimizer onto the
shared Celery/Redis queue; ``GET /jobs/{task_id}`` polls a job's state and
result. The app is a thin producer -- it dispatches tasks by name and never
imports the analytics stack.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from kombu.exceptions import OperationalError

from routers import clustering, fama_french, jobs, regime, rrg

app = FastAPI(title="bestee tuning API", version="0.1.0")

app.include_router(clustering.router)
app.include_router(regime.router)
app.include_router(fama_french.router)
app.include_router(rrg.router)
app.include_router(jobs.router)


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
