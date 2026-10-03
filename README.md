# RAG Chatbot over Indian Education & Skilling Policy Documents

A retrieval-augmented chatbot over 10 real Government of India policy PDFs —
free/open-source stack, 2-5s response target, no GPU needed.

**At a glance**
- **Corpus**: 10 real `.gov.in` PDFs, 200+ pages each (NCERT, UGC, AICTE, MSDE, Economic Survey, NITI Aayog)
- **Embeddings**: `multilingual-e5-small` · **Reranker**: `mmarco-mMiniLMv2-L12` · **Vector DB**: LanceDB (IVF_PQ)
- **LLM**: Groq-hosted `gpt-oss-20b` (open-weight, Apache 2.0)
- **UI**: Streamlit — ingestion stats, retrieval visualization, cited answers

**Why this domain:** NavGurukul skills underprivileged youth in India for
tech careers — work that sits inside the Indian education & skilling policy
ecosystem (NCERT curriculum, UGC/AICTE regulation, MSDE programs). This is
the kind of real document set such a team would actually want a research
assistant over, not a generic trivia demo.

## The core tension

**2-5s latency** + **open-source only** + **no GPU** pull against each
other — a quality open-weight LLM on CPU alone typically takes 15-60s,
blowing the budget before retrieval even starts.

**Resolution:** retrieval stays 100% local (embeddings, vector search,
reranker — no API calls), but *generation* runs on [Groq](https://console.groq.com),
which serves the same open-weight model on purpose-built hardware at
300-800+ tok/s, free tier. The model is still open-source — only the
serving infra is hosted. That's what makes 2-5s realistic, not aspirational.

## Corpus

| Document | Pages | Source |
|---|---|---|
| National Curriculum Framework 2023 | 628 | NCERT / Ministry of Education |
| Economic Survey 2023-24 | 524 | Ministry of Finance |
| UGC Annual Report 2021-22 | 380 | UGC |
| AISHE Report 2021-22 | 356 | Ministry of Education |
| UGC Annual Report 2020-21 | 340 | UGC |
| UGC Annual Report 2017-18 | 294 | UGC |
| MSDE Annual Report 2023-24 | 276 | Ministry of Skill Development |
| MSDE Annual Report 2022-23 | 264 | Ministry of Skill Development |
| School Education System in India | 232 | NITI Aayog |
| AICTE Approval Process Handbook | 231 | AICTE |

All byte-verified before download (magic bytes, real page count). Full
provenance: `data/corpus_manifest.json`.

Mostly native-text, but OCR is genuinely exercised: **44 of 3,525 pages**
(charts, scanned tables) had no text layer and went through RapidOCR —
see `data/ingest_report.json` for the breakdown.

Embeddings/reranker are multilingual even though this corpus is English —
NCERT content exists in Hindi/regional languages too, and the pipeline
should handle that without a code change.

## Architecture

```
Ingest:  PDF -> extract (native + OCR fallback) -> clean + chunk (800 tok, 15% overlap)
              -> embed -> LanceDB (IVF_PQ)

Query:   question -> embed -> ANN top-20 -> rerank top-3
                  -> Groq LLM -> cited answer + latency breakdown
```

```
src/        extraction, chunking, embeddings, vectorstore, reranker, generator, pipeline, config
scripts/    download_corpus.py, ingest.py
eval/       generate_eval_set.py, evaluate.py
app/        app.py — Streamlit chat UI
data/       pdfs/, lancedb/, corpus_manifest.json, ingest_report.json
```

## Key decisions

| Decision | Why |
|---|---|
| Groq, not local Ollama | Only way to hit 2-5s on CPU-only hardware |
| LanceDB, not Chroma | No C++ build tools needed here; still a real IVF_PQ ANN index |
| Multilingual models | NCERT exists in Hindi too — English-only would silently degrade later |
| Rerank top-3, not top-5 | Keeps prompts small, well within Groq free-tier limits |
| OCR only when needed | 100-1000x slower than native extraction — don't pay that cost everywhere |
| RapidOCR, not Tesseract | Pure `pip install`, no OS binary — faster to reproduce |
| Hand-rolled chunker | Deterministic IDs + exact page ranges per chunk |
| Retrieve-then-rerank | Bi-encoder is fast but shallow; cross-encoder is accurate but slow — use both |

## How to run

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

copy .env.example .env              # set GROQ_API_KEY (free: console.groq.com)

python scripts/download_corpus.py   # fetch the 10 PDFs
python scripts/ingest.py            # extract -> chunk -> embed -> LanceDB (~15-20 min)

streamlit run app/app.py            # -> http://localhost:8501
```

Already have `data/pdfs/` and `data/lancedb/` (e.g. copied from another
machine)? Skip straight to the last command.

## Evaluation

```bash
python eval/generate_eval_set.py    # auto-generates gold Q&A from real chunks
python eval/evaluate.py             # runs them live -> eval/eval_report.json
```

Metrics (per the spec's "Evaluation & Monitoring" requirement):
- **Latency** p50/p95, total + per-stage
- **Recall@k / MRR** — does the gold chunk surface, and how high
- **Citation accuracy** — do citations trace to a chunk the model actually saw
- **Unsupported-claim rate** — lexical-overlap hallucination proxy (not a solved problem — stated as a proxy, not dressed up as more)

**Latest run:** Recall@20/@3 = 1.0 · MRR 0.72 · citation accuracy 1.0 · p50 latency ~4.8s.

*Eval set kept small — Groq's free tier caps at 200k tokens/day, shared across every call this project makes.*

## What I'd add with more time

- LLM-as-judge hallucination scoring, cross-checked against the lexical proxy
- Query decomposition for multi-hop questions spanning more than one PDF
- Incremental ingestion instead of a full re-run on corpus changes
- A Hindi/regional NCERT document to actually exercise the multilingual models
- Bounding-box-level citations (currently page-level — spec marks this optional)
- A paid Groq tier for full 5-chunk context and higher rate limits
