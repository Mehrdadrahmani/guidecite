# PROGRESS

| Phase | Status | Verified |
|---|---|---|
| 0 Environment + scaffold | done | env, uv install, lock, 24 tests, ruff clean |
| 1 Data | done | 300 docs (100/source) -> 8,634 chunks, max 460 tokens, no empty chunks, metadata complete |
| 2 Index + retrieval | code written | — |
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
- Budget: user wants $0 spend -> local LLM (Ollama) or free GPU (Kaggle); awaiting decision.
