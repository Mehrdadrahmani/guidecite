# PROGRESS

| Phase | Status | Verified |
|---|---|---|
| 0 Environment + scaffold | done | env, uv install, lock, 24 tests, ruff clean |
| 1 Data | done | 300 docs (100/source) -> 8,634 chunks, max 460 tokens, no empty chunks, metadata complete |
| 2 Index + retrieval | done | 8,634 vectors (bge-base, MPS, 210s) + BM25; smoke test passes in all 4 modes |
| 3 Generation | code written | needs API key |
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

## Paused (2026-10-12) — resume here
- LLM decision: $0 budget -> local Ollama, `qwen3:8b` (thinking off, num_ctx 8192). Config defaults set.
  Kaggle (vLLM) for final heavy runs is an open option; needs the repo on GitHub.
- Ollama 0.40.3 installed via brew. `registry.ollama.ai` is unreachable (likely geo-block),
  so pull from Hugging Face instead: `ollama pull hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M` (~5 GB),
  then set `LLM_MODEL` in .env to that name.
- Start server: `OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve`
- Next: pull model -> Phase 3 (5 real Q&As, citations resolve) -> Phase 4 smoke tests.
- Judge model not chosen yet (different family preferred, e.g. a Gemma GGUF from HF).
- Smoke retrieval note: no adult-hypertension guideline in the 300-doc sample (coverage limitation).
