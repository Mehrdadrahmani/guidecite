# GuideCite. Runs inside the `medguide-rag` conda env (falls back to `conda run`).
ENV_NAME := medguide-rag
ifeq ($(CONDA_DEFAULT_ENV),$(ENV_NAME))
RUN :=
else
RUN := conda run -n $(ENV_NAME) --no-capture-output
endif
PY := $(RUN) python
ARGS ?=

.PHONY: setup lock data index smoke-retrieval eval-questions eval-retrieval eval-ragas eval-mcq report serve test lint format

setup:
	$(RUN) uv pip install -e ".[dev]"

lock:
	$(RUN) uv pip compile pyproject.toml --extra dev -o requirements.lock

data:
	$(PY) scripts/download_data.py $(ARGS)

index:
	$(PY) scripts/build_index.py $(ARGS)

smoke-retrieval:
	$(PY) scripts/smoke_retrieval.py

eval-questions:
	$(PY) -m medguide.eval.build_questions $(ARGS)

eval-retrieval:
	$(PY) -m medguide.eval.retrieval_eval $(ARGS)

eval-ragas:
	$(PY) -m medguide.eval.ragas_eval $(ARGS)

eval-mcq:
	$(PY) -m medguide.eval.mcq_eval $(ARGS)

report:
	$(PY) -m medguide.eval.report

serve:
	$(RUN) uvicorn medguide.api.main:app --host 0.0.0.0 --port 8000

test:
	$(RUN) pytest

lint:
	$(RUN) ruff check src tests scripts
	$(RUN) ruff format --check src tests scripts

format:
	$(RUN) ruff check --fix src tests scripts
	$(RUN) ruff format src tests scripts
