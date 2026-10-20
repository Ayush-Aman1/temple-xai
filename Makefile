# Temple XAI. Targets: setup, test, demo, eval, report, retrain
SHELL := /bin/bash
PY := .venv/bin/python
export PYTHONPATH := $(CURDIR)

.PHONY: setup test demo eval report retrain p4 kits

setup:                         ## Python venv + web deps (idempotent)
	@test -d .venv || (command -v uv >/dev/null && uv venv --python 3.11 .venv || python3.11 -m venv .venv)
	@(command -v uv >/dev/null && uv pip install -p .venv -r requirements.txt) || .venv/bin/pip install -r requirements.txt
	cd web && npm install --no-audit --no-fund

test:                          ## unit + API tests
	$(PY) -m pytest -q

demo:                          ## backend :8000 + frontend :5173 (Ctrl-C stops both)
	@trap 'kill 0' INT TERM EXIT; \
	$(PY) -m uvicorn templexai.api.main:app --port 8000 & \
	(cd web && npm run dev -- --host 127.0.0.1) & \
	wait

p4:                            ## cache detector/backbone outputs once, then train fusion + parts heads (minutes)
	$(PY) -m templexai.models.cache_features
	$(PY) -m templexai.models.fusion

eval:                          ## explanation metrics, ablation, robustness -> experiments/results/
	$(PY) -m templexai.eval.run_explanations
	$(PY) experiments/robustness.py
	$(PY) experiments/ablation.py

report:                        ## regenerate docs/RESULTS.md and the README results block from the metrics files
	$(PY) -m templexai.report

retrain:                       ## full training pipeline detached (see runs/train_all/STATUS)
	rm -rf runs/train_all/stages
	bash tools/bg.sh train_all "bash pipelines/train_all.sh"

kits:                          ## rebuild the human add-on kits in addons/
	$(PY) -m templexai.addons.build_kits
