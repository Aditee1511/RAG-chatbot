"""
End-to-end RAG orchestration with per-stage latency instrumentation.

Every query reports where its time actually went (embed / search / rerank /
generate), not just a single total -- this is what lets the "2-5s latency"
requirement be demonstrated rather than asserted, and what makes it possible
to spot regressions stage-by-stage instead of just "it got slower".
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import config, embeddings, generator, reranker
from .vectorstore import VectorStore

_store: VectorStore | None = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


@dataclass
class RAGResult:
    query: str
    answer: str
    sources: list[dict] = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return self.timings_ms.get("total", 0.0)


def answer(query: str) -> RAGResult:
    timings = {}
    t_start = time.perf_counter()

    t0 = time.perf_counter()
    query_emb = embeddings.encode_query(query)
    timings["embed_query_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    candidates = get_store().query(query_emb, top_k=config.VECTOR_TOP_K)
    timings["vector_search_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    top_chunks = reranker.rerank(query, candidates, top_n=config.RERANK_TOP_N)
    timings["rerank_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    if top_chunks:
        answer_text = generator.generate_answer(query, top_chunks)
    else:
        answer_text = (
            "I couldn't find anything relevant to that question in the ingested documents."
        )
    timings["generate_ms"] = (time.perf_counter() - t0) * 1000

    timings["total_ms"] = (time.perf_counter() - t_start) * 1000

    sources = [
        {
            "filename": c["metadata"]["filename"],
            "page_start": c["metadata"]["page_start"],
            "page_end": c["metadata"]["page_end"],
            "similarity": round(c["similarity"], 4),
            "rerank_score": round(c.get("rerank_score", 0.0), 4),
            "ocr": c["metadata"]["ocr"],
            "snippet": c["text"][:300],
        }
        for c in top_chunks
    ]

    return RAGResult(query=query, answer=answer_text, sources=sources, timings_ms=timings)
