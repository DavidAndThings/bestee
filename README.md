# bestee

Bestee is a small set of cooperating projects:

| Package          | Path        | Language       | Role                                                                 |
|------------------|-------------|----------------|----------------------------------------------------------------------|
| `bestee-compute` | `compute/`  | Python 3.14    | Financial-data DSL and indicator pipeline on top of the Massive API. |
| `bestee-chat`    | `chat/`     | Python 3.14    | Conversational chatbot.                                              |
| (UI)             | `ui/`       | TypeScript     | React + Vite frontend (Clerk auth).                                  |
| `bestee-api`     | `api/`      | (planned)      | Back-end service tying the others together.                          |

## Layout

This is a **uv workspace**: the root `pyproject.toml` lists `compute` and `chat`
as members, and `uv sync` at the root sets up a single `.venv` with both packages
installed editable. Each package keeps its own `pyproject.toml`, tests, and
console scripts under its subdirectory.

The TypeScript UI lives alongside the Python packages but is managed independently
(`cd ui && npm install`).

## Getting started

```bash
uv sync               # creates .venv with bestee-compute + bestee-chat
uv run pytest compute # tests for the compute package
uv run pytest chat    # tests for the chat package
cd ui && npm install && npm run dev   # local UI
```

## Per-package READMEs

- [`compute/README.md`](compute/README.md)
- [`chat/README.md`](chat/README.md)
- [`ui/README.md`](ui/README.md)
