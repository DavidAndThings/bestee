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
| `GET`  | `/jobs`                  | --                    | List the caller's jobs (newest first, paged).|
| `GET`  | `/jobs/{task_id}`        | --                    | Poll a job's state, inputs, result, or error.|
| `GET`  | `/results`               | --                    | List stored tuning results, newest first.    |
| `GET`  | `/results/{result_id}`   | --                    | Load a full stored tuning result.            |
| `GET`  | `/results/{result_id}/{aspect}` | --             | One analysis-specific aspect, as a table.    |
| `GET`  | `/search`                | --                    | Search SIC titles + company names (`?q=`).   |
| `GET`  | `/sic`                   | --                    | List every SIC code and its industry title.  |
| `GET`  | `/sic/{sic_code}/tickers`| --                    | Tickers classified under a SIC code.         |
| `GET`  | `/health`                | --                    | Liveness probe.                              |

Each submission is persisted per user (keyed by the Clerk `sub`) at dispatch
time, so `GET /jobs` returns the caller's history -- including still-queued jobs
and their recorded inputs (`analysis`, `payload`, deterministic `result_id`) --
immediately and across browsers/devices. Live state, timing, and errors are
merged in from the Celery result backend (which also persists each task's name +
args via `result_extended`).

A successful `POST` returns `202 Accepted` with a `TaskHandle`
(`{"task_id": ..., "result_id": ...}`). The `result_id` is deterministic -- a
hash of the resolved request, identical to what the worker saves under -- so it
is known immediately and a client can build the `/results/{result_id}/{aspect}`
URL before the job finishes (it `404`s until the result is written). Poll
`GET /jobs/{task_id}` until `state` is `SUCCESS` or `FAILURE` (carries `error`).

### Result aspects

`GET /results/{result_id}/{aspect}` returns a `ResultTable`
(`{result_id, aspect, columns, rows}`) -- a focused, analysis-specific view of a
stored result. The available aspect depends on the analysis that produced it:

| Analysis | Aspect | Columns |
| --- | --- | --- |
| clustering | `cluster_label` | `ticker`, `cluster_label` |
| rrg | `coordinates` | `timestamp`, `ticker`, `relative_strength`, `relative_momentum` |
| regime | `regime_label` | `timestamp`, `ticker`, `regime_label` (in-sample path + any out-of-sample dates) |
| fama_french | `ff_residuals` | `date`, `ticker`, `specification`, `residual`, `p_value` |

## Environment

| Variable | Required | Description |
|---|---|---|
| `DO_REDIS_CONNECTION` | Yes | Redis URL shared with the `queue` worker (broker + result backend). |
| `CLERK_JWKS_URL` | Yes | Clerk JWKS endpoint used to verify session tokens. Find it in the Clerk dashboard under **API Keys → Advanced → JWKS URL** (format: `https://<instance>.clerk.accounts.dev/.well-known/jwks.json`). |
| `CORS_ALLOW_ORIGINS` | No | Comma-separated browser origins allowed to call the API (default `http://localhost:5173,http://127.0.0.1:5173`). Set to the UI's domain for a split-origin deploy; unneeded when the UI and API share one origin behind a proxy. |
| `RESULTS_DIR` | No | Directory where tuning results are persisted (default `./results`). Must be the same path the `queue` worker writes to. |
| `LOG_DIR` | No | Directory for persisted logs, `api.log` (default `logs`). |
| `LOG_LEVEL` | No | Root log level (default `INFO`). |
| `LOG_FORMAT` | No | `json` (default, one object per line) or `text`. |

Logs are written to both stdout (captured by `systemd`/`journald`) and
`${LOG_DIR}/api.log`. See [`deploy/`](../../deploy/README.md) for droplet
`systemd` + `logrotate` templates.

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
