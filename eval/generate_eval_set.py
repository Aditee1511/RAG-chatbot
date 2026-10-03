"""
Auto-generate a gold eval set from the ingested corpus itself.

Hand-writing 20+ Q&A pairs against a 2000+ page corpus within a 4-hour
window isn't realistic, and guessing questions risks asking things the
corpus doesn't actually answer. Instead we sample real chunks and have the
LLM write one question per chunk that is answerable from *that chunk alone*.
Because we pick the chunk first, we know the exact ground-truth
(filename, page range, chunk_id) for every question -- which is what makes
Recall@k / MRR computable at all, instead of hand-waved.

This is a known technique (synthetic/self-supervised retrieval eval) -- it's
not a substitute for real user queries, and the eval report says so.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # Windows console default (cp1252) chokes on LLM-generated unicode punctuation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from groq import Groq

from src import config
from src.vectorstore import VectorStore

N_QUESTIONS = 5  # kept small: Groq free-tier daily token quota (200k/day) is shared
# across this account's generation calls, and each eval question costs one more
MIN_CHUNK_TOKENS = 150  # skip tiny/boilerplate chunks that don't carry a real fact

QUESTION_PROMPT = """Below is a single excerpt from a document. Write exactly ONE specific, \
factual question that can be answered using ONLY the information in this excerpt. \
The question should be the kind a real user would ask, not something generic like \
"what is this document about". Respond with ONLY the question, nothing else.

Excerpt:
{text}
"""


def main():
    random.seed(7)
    store = VectorStore()
    raw = store.get_all(limit=5000)

    candidates = [
        {"chunk_id": cid, "text": doc, "metadata": meta}
        for cid, doc, meta in zip(raw["ids"], raw["documents"], raw["metadatas"])
        if meta["token_count"] >= MIN_CHUNK_TOKENS
    ]
    if not candidates:
        print("No chunks found -- run scripts/ingest.py first.")
        return

    sample = random.sample(candidates, min(N_QUESTIONS, len(candidates)))
    client = Groq(api_key=config.GROQ_API_KEY)

    eval_set = []
    for item in sample:
        resp = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[{"role": "user", "content": QUESTION_PROMPT.format(text=item["text"])}],
            temperature=0.3,
            max_tokens=150,
            # gpt-oss burns completion tokens on a hidden reasoning trace
            # first; left uncapped, a tight max_tokens truncates before any
            # visible content is emitted (finish_reason="length", empty text).
            reasoning_effort="low",
        )
        question = resp.choices[0].message.content.strip()
        meta = item["metadata"]
        eval_set.append(
            {
                "question": question,
                "expected_chunk_id": item["chunk_id"],
                "expected_filename": meta["filename"],
                "expected_page_start": meta["page_start"],
                "expected_page_end": meta["page_end"],
            }
        )
        print(f"  [{meta['filename']} p.{meta['page_start']}] {question}")

    config.EVAL_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.EVAL_SET_PATH.write_text(json.dumps(eval_set, indent=2))
    print(f"\nWrote {len(eval_set)} questions to {config.EVAL_SET_PATH}")


if __name__ == "__main__":
    main()
