# bestee

Bestee is a small set of cooperating projects:

| Package          | Path        | Language       | Role                                                                 |
|------------------|-------------|----------------|----------------------------------------------------------------------|
| `bestee-compute` | `compute/`  | Python 3.14    | Financial-data DSL and indicator pipeline on top of the Massive API. |
| `bestee-chat`    | `chat/`     | Python 3.14    | Conversational chatbot.                                              |
| (UI)             | `ui/`       | TypeScript     | React + Vite frontend (Clerk auth).                                  |
| `bestee-api`     | `api/`      | (planned)      | Back-end service tying the others together.                          |

## Layout

Each project is self-contained — there is **no Python project at the repository
root**. `compute/` and `chat/` are independent [uv](https://docs.astral.sh/uv/)
projects, each with its own `pyproject.toml`, `uv.lock`, and `.venv`; `ui/` is a
standalone npm/Vite app. They share nothing but this directory and a few
repo-wide config files (`.pre-commit-config.yaml`, the CI workflows,
`.gitignore`).

A top-level `Makefile` fans the common commands out across the two Python
packages so you rarely need to `cd` into each.

## Getting started

```bash
make sync     # uv sync in compute + chat
make test     # pytest in compute + chat
make check    # lint + type-check + test (what CI runs)
make help     # list every target

# …or drive a single project directly:
cd compute && uv sync && uv run pytest
cd chat    && uv sync && uv run pytest
cd ui      && npm install && npm run dev
```

## Per-package READMEs

- [`compute/README.md`](compute/README.md)
- [`chat/README.md`](chat/README.md)
- [`ui/README.md`](ui/README.md)
