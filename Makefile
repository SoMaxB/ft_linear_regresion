PYTHON ?= python3
DATA ?= data.csv
MODEL ?= model/theta.json
LEARNING_RATE ?= 0.1
ITERATIONS ?= 1000
PLOT_OUTPUT ?= bonus/output/regression.svg
DIAGNOSTICS ?= bonus/output/training.json

.DEFAULT_GOAL := help

.PHONY: help check test verify train predict plot precision diagnostics

help: ## Show available commands
	@green=""; yellow=""; reset=""; \
	if [ -t 1 ] && command -v tput >/dev/null 2>&1; then \
		green=$$(tput setaf 2 2>/dev/null || true); \
		yellow=$$(tput setaf 3 2>/dev/null || true); \
		reset=$$(tput sgr0 2>/dev/null || true); \
	fi; \
	printf '\n%sUsage:%s\n  %smake <target>%s\n\n' "$$yellow" "$$reset" "$$green" "$$reset"; \
	printf '%sTargets:%s\n' "$$yellow" "$$reset"; \
	MAKE_GREEN="$$green" MAKE_YELLOW="$$yellow" MAKE_RESET="$$reset" \
	awk 'BEGIN { FS = ":.*## " } /^[a-zA-Z0-9][a-zA-Z0-9_-]*:.*## / { printf "  %s%-14s%s %s%s%s\n", ENVIRON["MAKE_GREEN"], $$1, ENVIRON["MAKE_RESET"], ENVIRON["MAKE_YELLOW"], $$2, ENVIRON["MAKE_RESET"] }' $(MAKEFILE_LIST)

check: ## Compile all Python sources
	$(PYTHON) -m compileall -q train.py predict.py src tests bonus

test: ## Run the complete test suite
	$(PYTHON) -m unittest discover -s tests -v

verify: check test ## Run compilation checks and tests

train: ## Train the model with configurable paths and defaults
	$(PYTHON) train.py --data "$(DATA)" --model "$(MODEL)" --learning-rate "$(LEARNING_RATE)" --iterations "$(ITERATIONS)"

predict: ## Predict a price interactively
	$(PYTHON) predict.py --model "$(MODEL)"

plot: ## Generate the dataset plot and regression line
	$(PYTHON) bonus/plot.py --data "$(DATA)" --model "$(MODEL)" --output "$(PLOT_OUTPUT)"

precision: ## Calculate MAE and MSE for the model
	$(PYTHON) bonus/precision.py --data "$(DATA)" --model "$(MODEL)"

diagnostics: ## Train with progress and export diagnostics JSON
	$(PYTHON) train.py --data "$(DATA)" --model "$(MODEL)" --learning-rate "$(LEARNING_RATE)" --iterations "$(ITERATIONS)" --progress --diagnostics "$(DIAGNOSTICS)"
