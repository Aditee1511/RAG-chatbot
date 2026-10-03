"""
Central configuration. Every tunable that affects quality or the 2-5s latency
budget lives here so it can be explained and defended in one place, instead of
scattered as magic numbers across the pipeline.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
PDF_DIR = ROOT / "data" / "pdfs"
VECTOR_DB_DIR = ROOT / "data" / "lancedb"
MANIFEST_PATH = ROOT / "data" / "corpus_manifest.json"
INGEST_REPORT_PATH = ROOT / "data" / "ingest_report.json"
EVAL_SET_PATH = ROOT / "eval" / "eval_set.json"
EVAL_REPORT_PATH = ROOT / "eval" / "eval_report.json"

# --- Chunking ---
# 500-1000 token window per the spec; 800 chosen as the midpoint so chunks are
# big enough to carry a full idea but small enough that the cross-encoder
# reranker (which has a ~512 token budget) can still see the whole thing.
CHUNK_SIZE_TOKENS = 800
CHUNK_OVERLAP_RATIO = 0.15  # 15% overlap: inside the 10-30% spec, biased low
# because our chunks are already large (800 tok) -- more overlap on top of
# that mostly duplicates storage/embedding cost without adding recall.
TOKEN_ENCODING = "cl100k_base"

# OCR fallback trigger: a native-text page with fewer than this many
# characters is treated as "no usable text layer" -> rasterize + OCR.
# 20 chars tolerates a mostly-blank page (e.g. a chapter divider) without
# wastefully OCR'ing it.
MIN_NATIVE_CHARS_PER_PAGE = 20

# Header/footer stripping: a line that repeats on more than this fraction of
# a document's pages is almost certainly running header/footer, not content.
HEADER_FOOTER_REPEAT_THRESHOLD = 0.5

# --- Embeddings ---
# multilingual-e5-small: the corpus is primarily English but also includes
# Hindi and Marathi documents (same pipeline, no per-language special-casing)
# -- an English-only model like bge-small would silently embed those two
# documents poorly. e5-small covers 100+ languages at nearly the same
# parameter count/CPU cost as the English-only alternative, so the
# multilingual coverage doesn't cost meaningful speed or quality on the
# English majority of the corpus.
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
# e5 models are trained with BOTH sides prefixed (unlike bge's query-only
# asymmetry) -- omitting the passage prefix at ingest time would silently
# mismatch the embedding space the query prefix was trained against.
EMBEDDING_QUERY_PREFIX = "query: "
EMBEDDING_PASSAGE_PREFIX = "passage: "
EMBEDDING_DIM = 384
EMBED_BATCH_SIZE = 64

# --- Retrieval / reranking ---
VECTOR_TOP_K = 20          # candidates pulled from ANN search
# Survivors after cross-encoder reranking, sent to the LLM. Capped at 3 (not
# 5) because this account's Groq free tier hard-caps at 8000 tokens/minute
# (discovered empirically: 413 rate_limit_exceeded on a 5-chunk/~4.3k-token
# prompt once two queries land in the same 60s window). 3 chunks of 800
# tokens keeps a single request's prompt safely under ~2.7k tokens, leaving
# real headroom before the per-minute cap, at the cost of narrower context.
RERANK_TOP_N = 3
# Multilingual counterpart to ms-marco-MiniLM -- same reasoning as the
# embedding model swap: an English-only cross-encoder would mis-score
# Hindi/Marathi candidates it can't actually read.
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
VECTOR_TABLE_NAME = "pdf_corpus"
VECTOR_METRIC = "cosine"

# IVF_PQ ANN index (LanceDB). Below ANN_MIN_ROWS_FOR_INDEX, LanceDB just does
# exact KNN, which is still correct -- there just aren't enough vectors yet
# for partitioning to pay off. NUM_SUB_VECTORS must evenly divide
# EMBEDDING_DIM (384 / 16 = 24-dim sub-vectors).
ANN_MIN_ROWS_FOR_INDEX = 256
ANN_NUM_SUB_VECTORS = 16
ANN_MAX_PARTITIONS = 256

# --- Generation ---
# Groq's LPU inference serves open-weight models at 300-800+ tok/s, which is
# what actually makes a 2-5s end-to-end budget achievable on CPU-only
# hardware with zero GPU -- a local model of this size on CPU alone would
# blow the budget on generation before retrieval even starts. gpt-oss-20b
# (OpenAI's Apache-2.0-licensed open-weight model) replaced the originally
# planned Llama 3.1 after Groq deprecated the Llama 3.x chat models from its
# catalog. Started on gpt-oss-120b, switched to the 20b variant after 120b's
# free-tier daily quota (200k tokens/day, account-wide not per-key) was
# exhausted by this session's own testing -- each model has its own separate
# daily budget, so this is a legitimate same-tier fallback, not a downgrade
# under duress with no justification. Both are well inside the latency budget.
GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_API_KEY_ENV = "GROQ_API_KEY"
MAX_ANSWER_TOKENS = 512
GENERATION_TEMPERATURE = 0.1  # low temp: this is a citation-grounded QA task, not creative writing

# --- Latency budget (documented, not just measured after the fact) ---
# Target total: 2-5s. Rough allocation used to justify each design choice:
LATENCY_BUDGET_MS = {
    "embed_query": 50,
    "vector_search": 150,
    "rerank": 600,
    "generate": 2500,
    "overhead": 200,
}
LATENCY_TARGET_MS = (2000, 5000)

GROQ_API_KEY = os.environ.get(GROQ_API_KEY_ENV, "")

# --- Eval harness pacing ---
# The batch eval script fires queries back-to-back, which (unlike a live demo
# with natural pauses between a human's questions) can stack multiple
# requests into the same 60s window and blow the 8000 TPM free-tier cap --
# the SDK then silently retries with ~30s backoff, polluting measured
# per-query latency with queueing time that has nothing to do with the
# pipeline's real speed. 31s spacing guarantees at most 2 requests (~6.4k
# tokens worst-case) land in any rolling 60s window.
EVAL_INTER_QUERY_DELAY_SEC = 31
