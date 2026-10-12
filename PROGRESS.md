# PROGRESS

| Phase | Status | Verified |
|---|---|---|
| 0 Environment + scaffold | in progress | env created (conda-forge), uv installed |
| 1 Data | code written | — |
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
