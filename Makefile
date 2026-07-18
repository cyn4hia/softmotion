# motion — common commands. Run `make help`.
.DEFAULT_GOAL := help
PY ?= python3

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Install CPU core (Mac mini / render machine)
	$(PY) -m pip install -e .

.PHONY: install-gpu
install-gpu: ## Install GPU + training extras (GPU box)
	$(PY) -m pip install -e ".[gpu,train,data,dev]"

.PHONY: dirs
dirs: ## Create the (gitignored) data/checkpoints/outputs skeleton
	$(PY) scripts/bootstrap_dirs.py

.PHONY: doctor
doctor: ## Report available devices and model backends
	motion doctor

.PHONY: smoke
smoke: ## Run the CPU morph end-to-end on synthetic images
	$(PY) scripts/smoke_test.py

.PHONY: test
test: ## Run the test suite
	$(PY) -m pytest

.PHONY: lint
lint: ## Lint + format check
	ruff check src tests scripts
	ruff format --check src tests scripts

.PHONY: fmt
fmt: ## Auto-format
	ruff format src tests scripts
	ruff check --fix src tests scripts

.PHONY: clean
clean: ## Remove caches
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__
