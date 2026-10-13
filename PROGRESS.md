# PROGRESS

| Phase | Status | Verified |
|---|---|---|
| 0 Environment + scaffold | done | env, uv install, lock, 24 tests, ruff clean |
| 1 Data | done | 300 docs (100/source) -> 8,634 chunks, max 460 tokens, no empty chunks, metadata complete |
| 2 Index + retrieval | done | 8,634 vectors (bge-base, MPS, 210s) + BM25; smoke test passes in all 4 modes |
| 3 Generation | done | 5 real Q&As (results/demo_answers.json): all citations valid, out-of-scope question declined, ~17-27 s/answer on laptop |
| 4 Evaluation | done | retrieval n=201, RAGAS n=50 + 20 out-of-scope, MCQ n=300 x 4 modes; all in results/ (synthetic questions UNVERIFIED) |
| 5 API + Docker | API done, Docker untested | API tested via TestClient; Docker not installed on dev machine |
| 6 Tests + CI | done | 35 offline tests pass, ruff clean; CI workflow not run (no push) |
| 7 README + docs | done | README numbers copied from results/; AGENTS.md, DECISIONS.md updated |

## Notes
- `medguide-rag` env did not exist; created with `conda create -c conda-forge --override-channels python=3.11`
  (the `defaults` channel returned HTTP 403).
- Dataset schema: id, source, title, clean_text, raw_text, url, overview. Missing values are the string "None".
  Source tags: who (223), cdc (621), nice (1656).
- CDC and WHO docs have no title/url in the dataset; titles are inferred (`title_inferred`), see DECISIONS.md.
- ragas 0.4.3 needs langchain-community<0.4.2 (pinned).

## Status (2026-10-13)
- All phases done. Total spend $0 (local Qwen3-8B via Ollama).
- Known issues: question-level prompt injection succeeds (oos10); Docker untested;
  eval questions not human-verified.
- Next (manual): review `eval/review.csv` (>=40 rows) -> `eval/questions_verified.jsonl`,
  review `eval/out_of_scope.jsonl`, then rerun eval-retrieval / eval-ragas / report.
