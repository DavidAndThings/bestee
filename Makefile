# Top-level task runner for the bestee monorepo.
#
# compute/, queue/, and api/ are independent uv projects (each with its
# own .venv and uv.lock); ui/ is a standalone npm/Vite app.  queue/ runs the
# bestee-compute tuning optimizers as Celery jobs and api/ is the FastAPI
# producer that enqueues them.  These targets just fan common commands out
# across the Python packages so you don't have to cd into each.  The UI keeps
# its own npm scripts (see ui/) and has dedicated `ui-*` targets.

PY := compute queue api

.PHONY: help sync test lint fmt typecheck check ui-install ui-dev ui-build ui-preview

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-12s\033[0m %s\n", $$1, $$2}'

sync:  ## Install/sync deps for every Python package
	@for p in $(PY); do echo "==> uv sync ($$p)"; (cd $$p && uv sync); done

test:  ## Run the Python test suites
	@for p in $(PY); do echo "==> pytest ($$p)"; (cd $$p && uv run pytest -q); done

lint:  ## Lint everything: ruff (Python packages) + eslint (ui)
	@for p in $(PY); do echo "==> ruff check ($$p)"; (cd $$p && uv run ruff check .); done
	@echo "==> eslint (ui)"; (cd ui && npm run lint)

fmt:  ## ruff format every Python package
	@for p in $(PY); do echo "==> ruff format ($$p)"; (cd $$p && uv run ruff format .); done

typecheck:  ## ty check every Python package
	@for p in $(PY); do echo "==> ty check ($$p)"; (cd $$p && uv run ty check); done

check: lint typecheck test  ## Lint + type-check + test (what CI runs)

ui-install:  ## npm install in ui/
	cd ui && npm install

ui-dev:  ## start the UI dev server (Vite)
	cd ui && npm run dev

ui-build:  ## type-check + build the UI
	cd ui && npm run build

ui-preview:  ## preview the production build of the UI
	cd ui && npm run preview
