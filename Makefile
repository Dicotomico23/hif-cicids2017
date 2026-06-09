# Convenience targets. Run `make help` for a summary.

.PHONY: help setup smoke reproduce clean

help:
	@echo "Targets:"
	@echo "  setup      install dependencies (pip install -r requirements.txt)"
	@echo "  smoke      run the fast synthetic smoke test (no dataset needed)"
	@echo "  reproduce  run the full study and write results/ and figures"
	@echo "  clean      remove generated results and caches"

setup:
	pip install -r requirements.txt
	pip install -e .

smoke:
	python tests/test_smoke.py

reproduce:
	python reproduce/run_comparison.py --output results

clean:
	rm -rf results/*.csv results/*.json results/figures/*.png
	find . -type d -name __pycache__ -exec rm -rf {} +
