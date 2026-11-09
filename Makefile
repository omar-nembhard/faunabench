.PHONY: setup test demo replays smoke mcp-smoke check

PYTHON ?= .venv/bin/python

setup:
	uv venv --python 3.12 .venv
	uv pip install --python $(PYTHON) -e '.[dev,analysis,providers]'

test:
	$(PYTHON) -m pytest -q

demo:
	$(PYTHON) -m faunabench.demo --mock --rung 2 --seed 1

replays:
	$(PYTHON) -m faunabench.demo --mock --replay-suite --seeds 10 --output artifacts/replays-v2

smoke:
	$(PYTHON) -m faunabench.inspect_task
	$(PYTHON) -m faunabench.inspect_task --transport mcp --output artifacts/inspect-mcp-smoke

mcp-smoke:
	$(PYTHON) scripts/mcp_smoke.py

check: test demo replays smoke mcp-smoke
