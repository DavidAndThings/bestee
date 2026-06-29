# bestee

Bestee is a small set of cooperating projects:

| Package          | Path               | Language    | Role                                                                 |
|------------------|--------------------|-------------|----------------------------------------------------------------------|
| `bestee-compute` | `backend/compute/` | Python 3.14 | Financial-data DSL and indicator pipeline on top of the Massive API. |
| `bestee-tasking` | `backend/tasking/` | Python 3.14 | Shared Celery app factory + Redis wiring for the producer and worker.|
| `bestee-queue`   | `backend/queue/`   | Python 3.14 | Celery worker that runs the `bestee-compute` tuning optimizers.      |
| `bestee-api`     | `backend/api/`     | Python 3.14 | FastAPI producer that enqueues tuning jobs and serves their results. |
| (UI)             | `ui/`              | TypeScript  | React + Vite frontend (Clerk auth).                                  |

## Layout

The Python code lives in `backend/`, a single [uv](https://docs.astral.sh/uv/)
**workspace**: `compute/`, `api/`, and `queue/` are members that share one
`.venv` and one `uv.lock` (`backend/pyproject.toml` defines the workspace).
`ui/` is a standalone npm/Vite app at the repository root. Backend secrets live
in `backend/.env`; the UI's in `ui/.env.local`. Repo-wide config
(`.pre-commit-config.yaml`, the CI workflows, `.gitignore`) stays at the root.

A top-level `Makefile` fans the common commands out across the workspace members
so you rarely need to `cd` into each.

## Getting started

```bash
make sync     # sync the backend uv workspace (one shared env)
make test     # pytest across compute, api, queue
make check    # lint + type-check + test (what CI runs)
make help     # list every target

# …or drive the workspace directly:
cd backend && uv sync --all-packages          # one shared env for all members
cd backend/api && uv run --no-sync pytest -q   # run one member's tests
cd ui && npm install && npm run dev
```

## Run the stack locally

The API only *enqueues* work; the Celery worker (`backend/queue/`) runs it, and
both connect to the shared (remote) Redis at `DO_REDIS_CONNECTION`. Start the
API, worker, and UI together with:

```bash
make dev          # API :8000 + Celery worker + UI :5173 (Ctrl-C stops all three)
```

`make dev` expects, in `backend/.env`:

- `DO_REDIS_CONNECTION` — the shared Redis broker + result backend.
- The worker's data keys: `MASSIVE_API_KEY` (plus `FRED_API_KEY` for
  Fama-French).

and, in `ui/.env.local`, `VITE_API_BASE_URL=http://localhost:8000` — otherwise
the UI uses its built-in mock and never calls the API.

Run the pieces separately with `make api-dev`, `make queue-dev`, and `make ui-dev`.

## Per-package READMEs

- [`backend/compute/README.md`](backend/compute/README.md)
- [`backend/queue/README.md`](backend/queue/README.md)
- [`backend/api/README.md`](backend/api/README.md)
- [`ui/README.md`](ui/README.md)
