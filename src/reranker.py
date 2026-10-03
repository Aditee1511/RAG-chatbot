"""
Cross-encoder reranking over the ANN shortlist.

Bi-encoder embeddings (what the vector search uses) score query and passage
independently, which is fast but loses cross-term interaction -- it's good at
"topically similar", not "actually answers this specific question". A
cross-encoder jointly scores (query, passage) pairs, which is much more
accurate but too slow to run over the whole corpus. The standard fix, used
here, is a two-stage retrieve-then-rerank: ANN search narrows millions of
chunks down to VECTOR_TOP_K candidates cheaply, then the cross-encoder spends
its (comparatively expensive) accuracy budget only on those few candidates.
"""
from __future__ import annotations

from sentence_transformers import CrossEncoder

from . import config

_reranker = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(config.RERANKER_MODEL)
    return _reranker


def rerank(query: str, candidates: list[dict], top_n: int) -> list[dict]:
    if not candidates:
        return []
    model = _get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates[:top_n]
