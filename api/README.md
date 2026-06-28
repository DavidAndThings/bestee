# api

A thin [FastAPI](https://fastapi.tiangolo.com/) producer that enqueues the
`bestee-compute` tuning optimizers onto the shared Celery/Redis queue. It
dispatches tasks **by name** and polls their results -- it never *runs* an
analysis (the `queue` worker does the heavy lifting). It depends on
`bestee-compute` only to reuse the request models, so the API and worker
validate against a single source of truth.

## Endpoints

| Method | Path                     | Body                  | Description                                  |
| ------ | ------------------------ | --------------------- | -------------------------------------------- |
| `POST` | `/tasks/clustering`      | `ClusteringRequest`   | Enqueue auto-tuned spectral clustering.      |
| `POST` | `/tasks/regime`          | `RegimeRequest`       | Enqueue auto-tuned per-asset regime labels.  |
| `POST` | `/tasks/fama-french`     | `FamaFrenchRequest`   | Enqueue an auto-tuned Fama-French fit.       |
| `POST` | `/tasks/rrg`             | `RRGRequest`          | Enqueue an auto-tuned relative-rotation job. |
| `GET`  | `/jobs/{task_id}`        | --                    | Poll a job's state, result, or error.        |
| `GET`  | `/health`                | --                    | Liveness probe.                              |

A successful `POST` returns `202 Accepted` with a `TaskHandle` (`{"task_id": ...}`).
Poll `GET /jobs/{task_id}` until `state` is `SUCCESS` (carries `result`) or
`FAILURE` (carries `error`).

## Environment

| Variable | Required | Description |
|---|---|---|
| `DO_REDIS_CONNECTION` | Yes | Redis URL shared with the `queue` worker (broker + result backend). |
| `CLERK_JWKS_URL` | Yes | Clerk JWKS endpoint used to verify session tokens. Find it in the Clerk dashboard under **API Keys → Advanced → JWKS URL** (format: `https://<instance>.clerk.accounts.dev/.well-known/jwks.json`). |
| `RESULTS_DIR` | No | Directory where tuning results are persisted (default `./results`). Must be the same path the `queue` worker writes to. |

## Run

```sh
uv sync
uv run uvicorn main:app --reload
```

Interactive docs are then served at `http://127.0.0.1:8000/docs`.

## Test

```sh
uv run pytest
```

Tests mock the Celery producer, so no broker is required.
