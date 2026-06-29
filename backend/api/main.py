"""FastAPI app exposing the bestee tuning queue over HTTP.

One ``POST /tasks/{analysis}`` endpoint enqueues each tuning optimizer onto the
shared Celery/Redis queue; ``GET /jobs/{task_id}`` polls a job's state and
result. The app is a thin producer -- it dispatches tasks by name and never
imports the analytics stack.
"""

import logging
import os
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from bestee_compute.logging_config import configure_logging
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from kombu.exceptions import OperationalError
from redis.exceptions import RedisError

from auth import require_auth
from routers import clustering, fama_french, jobs, regime, results, rrg, sic, ticker

load_dotenv()

# Browser origins allowed to call the API. Comma-separated in CORS_ALLOW_ORIGINS;
# defaults to the local Vite dev server. For a split-origin production deploy set
# it to the UI's domain (a same-origin deploy behind one proxy needs no CORS).
_DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"
_CORS_ALLOW_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("CORS_ALLOW_ORIGINS", _DEFAULT_CORS_ORIGINS).split(",")
    if origin.strip()
]


def _warm_search_catalog() -> None:
    """Build the /search catalog so the first request isn't a cold ~5s fetch."""
    try:
        ticker.get_all_search_terms()
        logging.getLogger("api").info("search catalog warmed")
    except Exception:
        logging.getLogger("api").warning("search catalog warm failed", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Persist logs to ``LOG_DIR`` (stdout + file) for the server's lifetime."""
    configure_logging("api")
    # Warm the (network-bound) search catalog off the request path so the first
    # autocomplete query is fast instead of blocking ~5s on a cold cache.
    threading.Thread(target=_warm_search_catalog, daemon=True).start()
    yield


app = FastAPI(title="bestee tuning API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_auth = [Depends(require_auth)]

app.include_router(clustering.router, dependencies=_auth)
app.include_router(regime.router, dependencies=_auth)
app.include_router(fama_french.router, dependencies=_auth)
app.include_router(rrg.router, dependencies=_auth)
app.include_router(jobs.router, dependencies=_auth)
app.include_router(results.router, dependencies=_auth)
app.include_router(sic.router, dependencies=_auth)
app.include_router(ticker.router, dependencies=_auth)


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
