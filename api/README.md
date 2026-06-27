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

Reads `DO_REDIS_CONNECTION` (the Celery broker/backend URL) from the environment
or a local `.env` file. The same Redis instance must back the `queue` worker.

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
