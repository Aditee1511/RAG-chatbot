"""
Answer generation via Groq's LPU-hosted open-weight models.

This is the single biggest lever for hitting the 2-5s end-to-end latency
target. Generation is normally the slowest stage of a RAG pipeline by a wide
margin -- a quantized 8B model on CPU alone commonly takes 15-60s for a few
hundred output tokens. Groq serves open-weight models on purpose-built
inference hardware at 300-800+ tokens/sec, which turns generation from the
bottleneck into the cheapest stage, leaving real budget for retrieval +
reranking while still landing well inside 5s. The free tier is enough for a
demo's query volume. The model weights themselves remain open-source
(gpt-oss-20b, Apache 2.0) -- only the serving infrastructure is hosted.
"""
from __future__ import annotations

from groq import Groq

from . import config

_client = None

SYSTEM_PROMPT = """You are a knowledgeable research assistant answering questions \
using ONLY the provided document excerpts. Rules:
1. Answer strictly from the excerpts below. If they don't contain the answer, say so plainly -- do not guess or use outside knowledge.
2. After every claim, cite its source inline like [filename, p.N]. If a claim draws on multiple excerpts, cite all of them.
3. NEVER start a paragraph or bullet with a bolded heading, a bolded label, or a number like "**1. Some Title**". This applies regardless of how the question is phrased.
4. Write in natural sentences, like you're explaining it to a colleague. For most answers, give each distinct point its own short paragraph starting directly with the content (e.g. "First, ..." / "The framework also ..."), separated by a blank line. A plain bulleted list ("- point text [file.pdf, p.N]") is also fine when the question asks for a clear set of items -- just never bold the start of a bullet as a heading.
5. Use **bold** sparingly, only inline on specific numbers/names/terms within a sentence, never on a whole sentence or as a heading.
6. Be concise -- a few short paragraphs at most, don't pad the answer.

Example of the right shape for a multi-point answer:
"The framework calls for organising content thematically, moving from local to global contexts [file.pdf, p.3].

It also recommends reducing content to core essentials so teachers have room for inquiry-based discussion [file.pdf, p.5]."

Example of what NOT to do: "**1. Thematic Organisation** - The framework calls for..."
"""


def _get_client() -> Groq:
    global _client
    if _client is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError(
                f"{config.GROQ_API_KEY_ENV} is not set. Get a free key at "
                "https://console.groq.com and set it as an environment variable."
            )
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


def _build_context(chunks: list[dict]) -> str:
    blocks = []
    for c in chunks:
        meta = c["metadata"]
        page_label = (
            f"p.{meta['page_start']}"
            if meta["page_start"] == meta["page_end"]
            else f"pp.{meta['page_start']}-{meta['page_end']}"
        )
        blocks.append(f"[{meta['filename']}, {page_label}]\n{c['text']}")
    return "\n\n---\n\n".join(blocks)


def generate_answer(query: str, chunks: list[dict]) -> str:
    client = _get_client()
    context = _build_context(chunks)
    user_prompt = f"Document excerpts:\n\n{context}\n\n---\n\nQuestion: {query}\n\nAnswer:"

    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=config.GENERATION_TEMPERATURE,
        max_tokens=config.MAX_ANSWER_TOKENS,
        # gpt-oss spends completion tokens on a hidden reasoning trace before
        # the visible answer; without capping it, max_tokens can be exhausted
        # by reasoning alone (finish_reason="length", empty content).
        reasoning_effort="low",
    )
    return response.choices[0].message.content
