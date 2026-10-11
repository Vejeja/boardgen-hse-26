.PHONY: setup setup-neural setup-full setup-tools studio doctor format lint format-check lock-check test smoke check pre-commit studio-check studio-format audit-pipelines asset-smoke pipeline-smoke research-check

PYTHON := .venv/bin/python
PROJECT ?= $(CURDIR)
STUDIO_PORT ?= 8769
MESH ?=

setup:
	uv sync --locked
setup-neural:
	uv sync --locked --extra neural
setup-full:
	uv sync --locked --all-extras
setup-tools:
	$(PYTHON) -m boardgen.maintenance.setup_external --weights
studio:
	.venv/bin/boardgen-studio --project "$(PROJECT)" --port $(STUDIO_PORT)
doctor:
	.venv/bin/boardgen-project --project "$(PROJECT)" doctor
format:
	.venv/bin/ruff check --fix src tests
	.venv/bin/ruff format src tests
lint:
	.venv/bin/ruff check src tests
format-check:
	.venv/bin/ruff format --check src tests
lock-check:
	uv lock --check --offline
test:
	.venv/bin/pytest -q
smoke:
	.venv/bin/boardgen-project --help
check: lock-check lint format-check test smoke research-check
pre-commit:
	.venv/bin/pre-commit run --all-files
studio-check:
	npm ci --ignore-scripts
	npm run check
	npm run format:check
studio-format:
	npm run format
audit-pipelines:
	$(PYTHON) -m boardgen.maintenance.audit_pipelines
asset-smoke:
	$(if $(strip $(MESH)),,$(error Set MESH=/absolute/path/to/selected.glb))
	$(PYTHON) -m boardgen.maintenance.smoke_asset_bridge --source "$(MESH)"
pipeline-smoke:
	$(if $(strip $(MESH)),,$(error Set MESH=/absolute/path/to/selected.glb))
	$(PYTHON) -m boardgen.maintenance.smoke_pipelines --mesh "$(MESH)"
research-check:
	$(PYTHON) -m boardgen.maintenance.research_report
