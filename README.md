# bestee

Bestee is a small set of cooperating projects:

| Package          | Path        | Language       | Role                                                                 |
|------------------|-------------|----------------|----------------------------------------------------------------------|
| `bestee-compute` | `compute/`  | Python 3.14    | Financial-data DSL and indicator pipeline on top of the Massive API. |
| `bestee-queue`   | `queue/`    | Python 3.14    | Celery worker that runs the `bestee-compute` tuning optimizers.      |
| `bestee-api`     | `api/`      | Python 3.14    | FastAPI producer that enqueues tuning jobs and serves their results. |
| (UI)             | `ui/`       | TypeScript     | React + Vite frontend (Clerk auth).                                  |

## Layout

Each project is self-contained — there is **no Python project at the repository
root**. `compute/`, `queue/`, and `api/` are independent
[uv](https://docs.astral.sh/uv/) projects, each with its own `pyproject.toml`,
`uv.lock`, and `.venv`; `ui/` is a standalone npm/Vite app. They share nothing
but this directory and a few repo-wide config files
(`.pre-commit-config.yaml`, the CI workflows, `.gitignore`).

A top-level `Makefile` fans the common commands out across the Python packages
so you rarely need to `cd` into each.

## Getting started

```bash
make sync     # uv sync in every Python package (compute, queue, api)
make test     # pytest in every Python package
make check    # lint + type-check + test (what CI runs)
make help     # list every target

# …or drive a single project directly:
cd compute && uv sync && uv run pytest
cd queue   && uv sync && uv run pytest
cd api     && uv sync && uv run pytest
cd ui      && npm install && npm run dev
```

## Run the stack locally

The API only *enqueues* work; the Celery worker (`queue/`) runs it, and both
talk to Redis. A functional local stack is therefore **Redis + API + worker + UI**:

```bash
# 1. Redis — skip if DO_REDIS_CONNECTION already points at a managed instance:
make redis-dev    # throwaway Redis on :6379 (needs Docker)

# 2. Everything else in one terminal (Ctrl-C stops all three):
make dev          # API :8000 + Celery worker + UI :5173
```

`make dev` expects:

- **Redis** reachable via `DO_REDIS_CONNECTION` (e.g. `redis://localhost:6379/0`
  with `make redis-dev`, or your managed URL).
- The worker's data keys (`MASSIVE_API_KEY`, plus `FRED_API_KEY` for
  Fama-French) in the repo-root `.env`.
- `VITE_API_BASE_URL=http://localhost:8000` in `ui/.env.local` — otherwise the
  UI uses its built-in mock and never calls the API.

Run the pieces separately with `make api-dev`, `make queue-dev`, and `make ui-dev`.

## Per-package READMEs

- [`compute/README.md`](compute/README.md)
- [`queue/README.md`](queue/README.md)
- [`api/README.md`](api/README.md)
- [`ui/README.md`](ui/README.md)
