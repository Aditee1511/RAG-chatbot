"""
sentence-transformers wrapper around intfloat/multilingual-e5-small.

e5 models are trained with an instruction prefix on BOTH queries and
passages (unlike bge's query-only asymmetry) -- "query: " / "passage: ".
Skipping either prefix is a common mistake that silently degrades retrieval
quality (cosine scores still "work", just worse) -- so we make both sides
explicit in the API rather than exposing one generic encode() that's easy to
misuse.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from . import config

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def encode_passages(texts: list[str]) -> np.ndarray:
    model = _get_model()
    prefixed = [config.EMBEDDING_PASSAGE_PREFIX + t for t in texts]
    return model.encode(
        prefixed,
        batch_size=config.EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def encode_query(text: str) -> np.ndarray:
    model = _get_model()
    prefixed = config.EMBEDDING_QUERY_PREFIX + text
    return model.encode([prefixed], normalize_embeddings=True, show_progress_bar=False)[0]
