# GuideCite — agent instructions

GuideCite: answers from clinical guidelines, with citations. A RAG assistant over public
WHO / CDC / NICE guidelines (`epfl-llm/guidelines`, ~2023 snapshot). Research/portfolio tool,
**not medical advice**.

## Environment
- Conda env: `medguide-rag` (Python 3.11). If it is not active, prefix commands with
  `conda run -n medguide-rag --no-capture-output` (the Makefile does this automatically).
- Install packages with `uv pip install ...` — **never plain `pip`**.
- Secrets live in `.env` (copy `.env.example`). Provider/model come from `LLM_PROVIDER`,
  `LLM_MODEL`, `JUDGE_PROVIDER`, `JUDGE_MODEL`.
- Default LLM is local Ollama: `OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve`,
  model `ollama pull hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M` (the Ollama registry may be blocked; HF works).

## Commands
```
make setup            # uv pip install -e ".[dev]"
make lock             # regenerate requirements.lock
make data             # download + sample + chunk -> data/processed/*.parquet
make index            # embeddings -> Qdrant (data/index/qdrant), BM25 -> data/index/bm25.pkl
make smoke-retrieval  # 5 queries x 4 retrieval modes
make eval-questions   # synthetic questions -> eval/questions_draft.jsonl (LLM calls)
make eval-retrieval   # Recall@k / MRR per mode -> results/
make eval-ragas       # RAGAS + citation validity + abstention (LLM calls)
make eval-mcq         # MCQ with/without retrieval (LLM calls)
make report           # results/ablation.md + chart
make serve            # FastAPI on :8000
make test             # pytest (offline)
make lint             # ruff check + format check
```
Pass extra flags with `ARGS="--limit 10"`.

## Layout
- `configs/default.yaml` — every model, path and hyper-parameter (incl. `language`).
- `prompts/` — all prompts as text files.
- `src/medguide/` — `config.py`, `data/` (load, clean, chunk), `index/` (embed, qdrant_store,
  bm25), `retrieval/` (dense, bm25, hybrid RRF, rerank, pipeline), `generation/` (llm, answer,
  citations), `eval/`, `api/main.py`.
- `scripts/` — data/index/smoke entry points. `tests/` — offline unit tests.
- `results/` — committed eval outputs. `data/` — gitignored corpus and indexes.
- `docs/DECISIONS.md` — design decisions log.

## Code style
- Type hints, small functions, ruff (line length 100).
- No hardcoded model names in code: read them from config/env.
- Retrieval modes (`dense | bm25 | hybrid | hybrid_rerank`) are switched by one word.

## Project rules
1. Never invent results: every number in README comes from a file in `results/`. Report
   failures and weak results honestly.
2. Never commit the corpus, indexes, `.env`, or model caches; scripts recreate them.
3. Cost control: print an LLM-call estimate and run a 10-item smoke test (`--limit 10`)
   before any full LLM-heavy run. Published results use the local Ollama model ($0); a hosted
   model is set via env, never in code. LLM calls are cached in `data/cache/llm_cache.sqlite`.
   Local runs are slow (~30 s/answer): run long evals under `caffeinate -ims` on macOS.
4. Medical disclaimer in the answer prompt, API responses and README.
5. Log each major design choice in `docs/DECISIONS.md` (choice, alternative, why).
