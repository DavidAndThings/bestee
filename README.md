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

## Per-package READMEs

- [`compute/README.md`](compute/README.md)
- [`queue/README.md`](queue/README.md)
- [`api/README.md`](api/README.md)
- [`ui/README.md`](ui/README.md)
