"""
Token-based chunking with page-aware metadata.

We deliberately hand-roll this instead of calling a framework's
`RecursiveCharacterTextSplitter` black box, for two reasons:
  1. Determinism: chunk boundaries depend only on token count, not on a
     library's internal separator-splitting heuristics that can change
     between versions -- re-running ingestion on the same PDFs always
     produces the same chunks and the same chunk IDs.
  2. Citations need page numbers. A chunk that spans a page boundary must
     carry (page_start, page_end), not just "the chunk came from somewhere
     in this document" -- a generic text splitter operating on a single
     flattened string loses page boundaries entirely.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import tiktoken

from . import config

_encoding = tiktoken.get_encoding(config.TOKEN_ENCODING)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    filename: str
    page_start: int
    page_end: int
    text: str
    token_count: int
    ocr: bool  # True if any source page used OCR


def _make_chunk_id(filename: str, page_start: int, index: int) -> str:
    raw = f"{filename}:{page_start}:{index}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def chunk_document(pages, doc_id: str, filename: str) -> list[Chunk]:
    """pages: list of extraction.PageResult, in page order."""
    # Flatten to a token stream while remembering which page each token
    # boundary falls on, so a chunk can report an accurate [page_start, page_end].
    tokens: list[int] = []
    token_page: list[int] = []  # parallel array: which page each token came from
    token_ocr: list[bool] = []

    for page in pages:
        if not page.text.strip():
            continue
        page_tokens = _encoding.encode(page.text)
        tokens.extend(page_tokens)
        token_page.extend([page.page_num] * len(page_tokens))
        token_ocr.extend([page.method == "ocr"] * len(page_tokens))

    chunk_size = config.CHUNK_SIZE_TOKENS
    overlap = int(chunk_size * config.CHUNK_OVERLAP_RATIO)
    stride = chunk_size - overlap

    chunks: list[Chunk] = []
    index = 0
    start = 0
    n = len(tokens)
    while start < n:
        end = min(start + chunk_size, n)
        chunk_tokens = tokens[start:end]
        text = _encoding.decode(chunk_tokens)
        page_start = token_page[start]
        page_end = token_page[end - 1]
        has_ocr = any(token_ocr[start:end])
        chunks.append(
            Chunk(
                chunk_id=_make_chunk_id(filename, page_start, index),
                doc_id=doc_id,
                filename=filename,
                page_start=page_start,
                page_end=page_end,
                text=text,
                token_count=len(chunk_tokens),
                ocr=has_ocr,
            )
        )
        index += 1
        if end == n:
            break
        start += stride

    return chunks
