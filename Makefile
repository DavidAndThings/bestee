# Top-level task runner for the bestee monorepo.
#
# backend/ is a single uv workspace: compute/, api/, and queue/ share one
# .venv and one uv.lock (see backend/pyproject.toml).  ui/ is a standalone
# npm/Vite app.  queue/ runs the bestee-compute tuning optimizers as Celery
# jobs and api/ is the FastAPI producer that enqueues them.  These targets fan
# the common commands out across the workspace members; the UI keeps its own
# npm scripts and has dedicated `ui-*` targets.

PY := compute tasking api queue

.PHONY: help sync test lint fmt typecheck check ui-install ui-dev ui-build ui-preview free-api-port api-dev queue-dev dev

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-12s\033[0m %s\n", $$1, $$2}'

sync:  ## Sync the backend uv workspace (one shared env for compute, api, queue)
	cd backend && uv sync --all-packages

test: sync  ## Run the Python test suites in the shared workspace env
	@for p in $(PY); do echo "==> pytest ($$p)"; (cd backend/$$p && uv run --no-sync pytest -q); done

lint: sync  ## Lint everything: ruff (Python packages) + eslint (ui)
	@for p in $(PY); do echo "==> ruff check ($$p)"; (cd backend/$$p && uv run --no-sync ruff check .); done
	@echo "==> eslint (ui)"; (cd ui && npm run lint)

fmt: sync  ## ruff format every Python package
	@for p in $(PY); do echo "==> ruff format ($$p)"; (cd backend/$$p && uv run --no-sync ruff format .); done

typecheck: sync  ## ty check every Python package
	@for p in $(PY); do echo "==> ty check ($$p)"; (cd backend/$$p && uv run --no-sync ty check); done

check: lint typecheck test  ## Lint + type-check + test (what CI runs)

ui-install:  ## npm install in ui/
	cd ui && npm install

ui-dev:  ## start the UI dev server (Vite)
	cd ui && npm run dev

ui-build:  ## type-check + build the UI
	cd ui && npm run build

ui-preview:  ## preview the production build of the UI
	cd ui && npm run preview

# ---- Local dev orchestration -------------------------------------------------
# These start long-running servers. The API and worker both connect to the
# remote Redis at DO_REDIS_CONNECTION (set in backend/.env). For the UI to call
# the API instead of its built-in mock, set
# VITE_API_BASE_URL=http://localhost:8000 in ui/.env.local.

free-api-port:  ## Stop any stale listener bound to :8000 (orphaned uvicorn, etc.)
	@PIDS="$$(lsof -ti tcp:8000 -sTCP:LISTEN 2>/dev/null)"; \
	if [ -n "$$PIDS" ]; then \
		echo "Freeing port 8000 (stopping PID(s) $$PIDS)"; \
		kill $$PIDS 2>/dev/null || true; sleep 1; \
		PIDS="$$(lsof -ti tcp:8000 -sTCP:LISTEN 2>/dev/null)"; \
		if [ -n "$$PIDS" ]; then kill -9 $$PIDS 2>/dev/null || true; sleep 1; fi; \
	fi

api-dev: sync free-api-port  ## Run the API (uvicorn, autoreload) on :8000
	cd backend/api && uv run --no-sync uvicorn main:app --reload --port 8000

queue-dev: sync  ## Run the Celery worker that processes submitted jobs
	cd backend/queue && uv run --no-sync celery -A main worker --loglevel=info

dev: sync free-api-port  ## Run API + worker + UI together (Ctrl-C stops all). Needs remote Redis.
	@echo "Starting API :8000, Celery worker, and UI :5173. Ctrl-C stops all."
	@echo "Prereqs: backend/.env (DO_REDIS_CONNECTION, MASSIVE_API_KEY) + ui/.env.local VITE_API_BASE_URL=http://localhost:8000"
	@trap 'kill 0' INT TERM HUP EXIT; \
	(cd backend/api && uv run --no-sync uvicorn main:app --reload --port 8000) & \
	(cd backend/queue && uv run --no-sync celery -A main worker --loglevel=info) & \
	(cd ui && npm run dev) & \
	wait
