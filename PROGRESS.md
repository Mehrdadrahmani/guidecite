# PROGRESS

| Phase | Status | Verified |
|---|---|---|
| 0 Environment + scaffold | done | env, uv install, lock, 24 tests, ruff clean |
| 1 Data | done | 300 docs (100/source) -> 8,634 chunks, max 460 tokens, no empty chunks, metadata complete |
| 2 Index + retrieval | done | 8,634 vectors (bge-base, MPS, 210s) + BM25; smoke test passes in all 4 modes |
| 3 Generation | done | 5 real Q&As (results/demo_answers.json): all citations valid, out-of-scope question declined, ~17-27 s/answer on laptop |
| 4 Evaluation | todo | |
| 5 API + Docker | todo | Docker not installed on dev machine |
| 6 Tests + CI | tests drafted | |
| 7 README + docs | todo | |

## Notes
- `medguide-rag` env did not exist; created with `conda create -c conda-forge --override-channels python=3.11`
  (the `defaults` channel returned HTTP 403).
- Dataset schema: id, source, title, clean_text, raw_text, url, overview. Missing values are the string "None".
  Source tags: who (223), cdc (621), nice (1656).
- CDC and WHO docs have no title/url in the dataset; titles are inferred (`title_inferred`), see DECISIONS.md.
- ragas 0.4.3 needs langchain-community<0.4.2 (pinned).

## Paused (2026-10-12, 17:30) — resume here
- Model installed: `hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M` (Ollama, pulled from HF; registry.ollama.ai is blocked).
- Start server: `OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve`
- Phase 4 status: question-generation prompts/filters tuned on smoke tests (15 items look good).
  Full run (`make eval-questions`, ~720 calls, ~35-45 min) was started and stopped after 2 min; restart it.
- Then: `make eval-retrieval` (no LLM) -> smoke `make eval-ragas ARGS="--limit 10"` and
  `make eval-mcq ARGS="--limit 10"` for timings -> full runs overnight (`caffeinate -i`).
- Then: `make report`, README, final AGENTS.md/CLAUDE.md check.
- Docker files are written but untested (Docker not installed).
