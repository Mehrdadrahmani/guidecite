# Design decisions

Each entry: what was chosen, the main alternative, and why.

## Corpus: WHO + CDC + NICE, 100 docs each
Chosen: a balanced, seeded sample of ~300 documents (300–15,000 words) from three sources.
Alternative: the full 37k-document dataset (dominated by wikidoc).
Why: three authoritative, guideline-style sources with real URLs. Balance stops NICE (1,656
docs) from dominating, and 300 docs keep indexing and evaluation cheap enough to iterate on.

## Chunking: heading-aware packing + recursive token splitter, ~450 tokens / 64 overlap
Chosen: split on markdown headings, greedily pack small consecutive sections, and recursively split
sections that are too large (paragraph → line → sentence). Every chunk gets a "Title / Section" prefix.
Alternative: fixed-size character windows.
Why: guideline recommendations are organised by section, so heading boundaries keep a
recommendation intact. The prefix makes chunks self-explanatory for both the embedder and the LLM.
Token counts use the embedding model's tokenizer, so chunks stay under bge's 512-token limit.

## Embeddings: BAAI/bge-base-en-v1.5 (bge-small on CPU)
Chosen: bge-base with its query instruction, normalised vectors, cosine distance.
Alternative: e5-base-v2, or OpenAI text-embedding-3.
Why: a strong open MTEB retrieval model that runs locally on MPS, with no per-query API cost. It
has a drop-in multilingual successor (bge-m3) for the Persian version. The index manifest records
the model used, and query time uses the same model even if the hardware fallback differs.

## Vector store: Qdrant via qdrant-client (local file mode / server mode)
Chosen: one code path. `vector_store.mode: local|server` switches between an embedded
on-disk DB and the docker-compose server.
Alternative: FAISS, or the langchain-qdrant wrapper.
Why: Qdrant has payloads and a real server for deployment. Calling qdrant-client directly
avoids churn in the LangChain wrapper's API. LangChain is still used for the Embeddings interface,
text splitters and chat models.

## BM25: rank_bm25 Okapi, lowercase regex tokenizer, no stemming
Chosen: tokens like `covid-19`, `mg/kg`, `hba1c`, `6.5` are kept as single tokens, with a small
stopword list.
Alternative: a Porter-stemmed analyser (e.g. Elasticsearch's English analyser).
Why: exact drug names, doses and thresholds are the main thing lexical search adds over dense
retrieval, and stemming or splitting would weaken them.

## Fusion: reciprocal rank fusion, k=60, dense top-30 + BM25 top-30
Chosen: RRF.
Alternative: a weighted sum of normalised scores.
Why: BM25 and cosine scores sit on incomparable scales. RRF uses ranks only, has no weight to
tune (which would be easy to overfit on a small synthetic set), and k=60 is the standard value
from Cormack et al. (2009).

## Reranker: BAAI/bge-reranker-base over the fused top-25, keep top 5
Chosen: a cross-encoder rerank.
Alternative: no reranking, or an LLM reranker.
Why: a cross-encoder reads the query and passage together and usually improves precision
at the top. It runs locally in about 100–300 ms for 25 pairs on MPS, and has a multilingual
successor (bge-reranker-v2-m3).

## Answer prompt: numbered passages, mandatory [n] citations, explicit abstention
Chosen: context is passed as numbered blocks inside `<context>` and marked as data. Every claim
must carry a citation, and when the evidence is insufficient the model gives a fixed
"not enough information" sentence.
Alternative: free-form answers with post-hoc attribution.
Why: inline numbers can be checked mechanically (`citations.py`), so they give a measurable
citation-validity rate. The fixed abstention sentence makes abstention measurable too.

## MCQ benchmark: MedQA (USMLE, 4 options), English
Chosen: `GBaker/MedQA-USMLE-4-options`, test split, 300 questions (seed 42), stratified by exam step.
Alternative: the English side of PersianMedQA.
Why: this release is English-only. PersianMedQA is gated, and it belongs to the later Persian
phase, where `mcq_dataset` is a one-line config change.

## LLM access: provider-agnostic via LangChain init_chat_model (+ OpenRouter)
Chosen: `anthropic | openai | openrouter | ollama`, selected by env. OpenRouter reuses the OpenAI
client with a different base URL.
Alternative: a hard dependency on one vendor SDK.
Why: one OpenRouter key can reach Claude, GPT and open models. That makes it cheap to use a judge
from a different model family than the generator, which reduces same-family bias in RAGAS.
