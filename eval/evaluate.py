"""
Runs the eval set end-to-end and reports the four metrics the challenge spec
explicitly asks for: latency (p50/p95), relevance (Recall@k, MRR),
citation accuracy, and an (heuristic) hallucination/unsupported-claim rate.

Honesty about the hallucination metric: true hallucination detection needs
either human judgment or a much stronger LLM-as-judge pass, both too slow for
a 4-hour build. What's implemented here is a cheap but real proxy -- lexical
overlap between each answer sentence and the retrieved context -- reported as
"unsupported-claim rate", not dressed up as a solved problem.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # Windows console default (cp1252) chokes on LLM-generated unicode punctuation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, pipeline

CITATION_RE = re.compile(r"\[([^,\]]+),\s*pp?\.\s*(\d+)(?:-(\d+))?\]")

# gpt-oss renders page ranges with typographic dashes (non-breaking hyphen,
# en dash, etc.) instead of ASCII '-', e.g. "pp.103‑104" -- which silently
# fails CITATION_RE's literal '-' and would undercount real citations as zero.
_DASH_CHARS = "‐‑‒–—−"
_DASH_TRANSLATION = str.maketrans(_DASH_CHARS, "-" * len(_DASH_CHARS))


def _normalize_dashes(text: str) -> str:
    return text.translate(_DASH_TRANSLATION)


def recall_and_rank(expected_chunk_id: str, retrieved_ids: list[str]) -> tuple[bool, int | None]:
    if expected_chunk_id in retrieved_ids:
        rank = retrieved_ids.index(expected_chunk_id) + 1
        return True, rank
    return False, None


def citation_accuracy(answer: str, sources: list[dict]) -> tuple[int, int]:
    """Returns (valid_citations, total_citations). A citation is valid if its
    (filename, page) falls inside the page range of some retrieved chunk that
    was actually passed to the LLM -- i.e. the model didn't invent a source."""
    citations = CITATION_RE.findall(_normalize_dashes(answer))
    if not citations:
        return 0, 0
    valid = 0
    for filename, p_start, p_end in citations:
        cited_pages = range(int(p_start), int(p_end or p_start) + 1)
        if any(
            s["filename"] == filename.strip()
            and set(cited_pages) & set(range(s["page_start"], s["page_end"] + 1))
            for s in sources
        ):
            valid += 1
    return valid, len(citations)


def unsupported_claim_rate(answer: str, sources: list[dict]) -> float:
    context_words = set()
    for s in sources:
        context_words |= set(w.lower() for w in re.findall(r"[a-zA-Z]{4,}", s["snippet"]))
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    if not sentences:
        return 0.0
    unsupported = 0
    for sent in sentences:
        words = set(w.lower() for w in re.findall(r"[a-zA-Z]{4,}", sent))
        if not words:
            continue
        overlap = len(words & context_words) / len(words)
        if overlap < 0.15:
            unsupported += 1
    return unsupported / len(sentences)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * p
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    return values[f] + (values[c] - values[f]) * (k - f)


def main():
    eval_set = json.loads(config.EVAL_SET_PATH.read_text())
    store = pipeline.get_store()

    total_latency_ms, stage_latency = [], {"embed_query_ms": [], "vector_search_ms": [], "rerank_ms": [], "generate_ms": []}
    recall_at_vector_k, recall_at_rerank_n, mrr_ranks = [], [], []
    citation_valid_total, citation_count_total = 0, 0
    unsupported_rates = []
    rows = []

    for i, item in enumerate(eval_set):
        from src import embeddings as emb_mod

        query_emb = emb_mod.encode_query(item["question"])
        candidates = store.query(query_emb, top_k=config.VECTOR_TOP_K)
        candidate_ids = [c["chunk_id"] for c in candidates]
        hit_k, rank = recall_and_rank(item["expected_chunk_id"], candidate_ids)
        recall_at_vector_k.append(hit_k)
        mrr_ranks.append(1.0 / rank if rank else 0.0)

        result = pipeline.answer(item["question"])
        retrieved_in_final = {(s["filename"], s["page_start"]) for s in result.sources}
        hit_rerank = (item["expected_filename"], item["expected_page_start"]) in retrieved_in_final
        recall_at_rerank_n.append(hit_rerank)

        total_latency_ms.append(result.timings_ms["total_ms"])
        for k in stage_latency:
            stage_latency[k].append(result.timings_ms[k])

        valid, count = citation_accuracy(result.answer, result.sources)
        citation_valid_total += valid
        citation_count_total += count
        unsupported_rates.append(unsupported_claim_rate(result.answer, result.sources))

        rows.append(
            {
                "question": item["question"],
                "expected": f"{item['expected_filename']} p.{item['expected_page_start']}",
                "hit_vector_top_k": hit_k,
                "hit_final_top_n": hit_rerank,
                "total_ms": round(result.timings_ms["total_ms"], 1),
            }
        )
        print(f"[{'OK ' if hit_rerank else 'MISS'}] {item['question'][:70]:70s} {result.timings_ms['total_ms']:.0f}ms")

        if i < len(eval_set) - 1:
            time.sleep(config.EVAL_INTER_QUERY_DELAY_SEC)

    report = {
        "n_questions": len(eval_set),
        "latency_ms": {
            "p50_total": round(percentile(total_latency_ms, 0.5), 1),
            "p95_total": round(percentile(total_latency_ms, 0.95), 1),
            **{f"mean_{k}": round(statistics.mean(v), 1) for k, v in stage_latency.items()},
        },
        "retrieval": {
            f"recall_at_{config.VECTOR_TOP_K}_vector": round(statistics.mean(recall_at_vector_k), 3),
            f"recall_at_{config.RERANK_TOP_N}_final": round(statistics.mean(recall_at_rerank_n), 3),
            "mrr": round(statistics.mean(mrr_ranks), 3),
        },
        "citation_accuracy": round(citation_valid_total / citation_count_total, 3) if citation_count_total else None,
        "unsupported_claim_rate": round(statistics.mean(unsupported_rates), 3),
        "rows": rows,
    }

    config.EVAL_REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nLatency p50/p95: {report['latency_ms']['p50_total']}ms / {report['latency_ms']['p95_total']}ms")
    print(f"Recall@{config.VECTOR_TOP_K} (vector): {report['retrieval'][f'recall_at_{config.VECTOR_TOP_K}_vector']}")
    print(f"Recall@{config.RERANK_TOP_N} (final):  {report['retrieval'][f'recall_at_{config.RERANK_TOP_N}_final']}")
    print(f"MRR: {report['retrieval']['mrr']}")
    print(f"Citation accuracy: {report['citation_accuracy']}")
    print(f"Unsupported-claim rate: {report['unsupported_claim_rate']}")
    print(f"\nFull report written to {config.EVAL_REPORT_PATH}")


if __name__ == "__main__":
    main()
