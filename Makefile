PY := backend/.venv/bin/python
PIP := backend/.venv/bin/pip

.PHONY: help install test train bench episodes simulate dev api web clean
.PHONY: test-m0 test-m1 test-m2 test-m3 test-m4 test-m5 test-m7 test-m8 test-m9 test-m10
.PHONY: test-web demo fixtures

help:
	@grep -E '^[a-z0-9-]+:.*?## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

install: ## create venv, install python + node deps
	python3 -m venv backend/.venv
	$(PIP) install -q --upgrade pip
	cd backend && .venv/bin/pip install -q -e ".[dev]"
	cd frontend && npm install
	cd backend && .venv/bin/python -m orchestrator.forecast.train

test: ## full python test suite
	cd backend && .venv/bin/python -m pytest -q

test-m0: ; cd backend && .venv/bin/python -m pytest -q tests/test_config.py
test-m1: ; cd backend && .venv/bin/python -m pytest -q tests/test_world.py
test-m2: ; cd backend && .venv/bin/python -m pytest -q tests/test_dispatch.py
test-m3: ; cd backend && .venv/bin/python -m pytest -q tests/test_forecast.py
test-m4: ; cd backend && .venv/bin/python -m pytest -q tests/test_m4_criteria.py
test-m5: ; cd backend && .venv/bin/python -m pytest -q tests/test_m5_m6.py
test-m7: ; cd backend && .venv/bin/python -m pytest -q tests/test_m7_simulation.py
test-m8: ; cd backend && .venv/bin/python -m pytest -q tests/test_m8_agent.py
test-m9: ; cd backend && .venv/bin/python -m pytest -q tests/test_api.py
test-m10: test-web ## console: typecheck, unit tests, production build
	cd frontend && npm run build

test-web: ## console unit + render tests against recorded API fixtures
	cd frontend && npx vitest run

fixtures: ## re-record the console's API fixtures from a running backend
	cd frontend && ORCH_API=$${ORCH_API:-http://127.0.0.1:8000} node scripts/capture-fixtures.mjs

demo: ## drive the PRD's 7-minute demo script through a running API
	cd backend && .venv/bin/python scripts/demo_walkthrough.py $${ORCH_API:-http://127.0.0.1:8000}

train: ## retrain + recalibrate the quantile forecasters (nightly job)
	cd backend && .venv/bin/python -m orchestrator.forecast.train

bench: ## optimiser latency against the <1s budget
	cd backend && .venv/bin/python scripts/bench.py

simulate: ## F3 scenario benchmark: 24h x N paths x 5 presets x 3 strategies
	cd backend && .venv/bin/python scripts/run_simulation.py 6

episodes: ## 30 randomised episodes -> invariant report + baseline win rate
	cd backend && .venv/bin/python scripts/run_episodes.py

api: ## backend only, on :8000
	cd backend && .venv/bin/python -m uvicorn orchestrator.api.app:app --reload --port 8000

web: ## frontend only, on :5173
	cd frontend && npm run dev

dev: ## backend :8000 + frontend :5173
	@echo "starting backend on :8000 and frontend on :5173 (Ctrl-C to stop both)"
	@(cd backend && .venv/bin/python -m uvicorn orchestrator.api.app:app --port 8000 &) ; \
	 sleep 2 ; cd frontend && npm run dev

clean:
	rm -rf backend/.venv backend/**/__pycache__ frontend/node_modules frontend/dist
	rm -rf backend/.pytest_cache backend/artifacts
