"""
PDF -> per-page text, with an OCR fallback for scanned pages.

Design choice: we don't OCR every page. We try native extraction first
(cheap, exact) and only rasterize+OCR pages that come back empty, because OCR
is ~100-1000x slower than native text extraction. Running OCR unconditionally
over a 10x200-page corpus would turn a 2-minute ingest into an hour-plus one
for no quality gain on the 95% of pages that already have a text layer.

OCR engine: RapidOCR (onnxruntime) instead of Tesseract. Tesseract needs a
separate OS-level binary installed outside of pip, which isn't reproducible
on a machine where that can't be guaranteed (no admin rights, no internet
during a demo, etc). RapidOCR ships its models via pip and runs on CPU with
no extra install step -- a pure `pip install -r requirements.txt` reproduces
the whole stack on any machine, which matters a lot for a 4-hour, run-it-
anywhere challenge.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

import fitz  # PyMuPDF
from langdetect import LangDetectException, detect

from . import config

logger = logging.getLogger(__name__)

_ocr_engine = None


def _get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR

        _ocr_engine = RapidOCR()
    return _ocr_engine


@dataclass
class PageResult:
    page_num: int  # 1-indexed
    text: str
    method: str  # "native" | "ocr"


@dataclass
class DocumentResult:
    filename: str
    pages: list[PageResult] = field(default_factory=list)
    language: str = "unknown"
    ocr_page_count: int = 0


def _ocr_page(page: "fitz.Page") -> str:
    """Rasterize a page to an image and run OCR on it."""
    # 200 DPI is a quality/speed middle ground: Tesseract/RapidOCR accuracy
    # degrades noticeably below ~150 DPI on small body text, while going much
    # higher (300+) roughly doubles OCR time for negligible accuracy gain on
    # typical scanned reports.
    zoom = 200 / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img_bytes = pix.tobytes("png")
    engine = _get_ocr_engine()
    result, _ = engine(img_bytes)
    if not result:
        return ""
    return "\n".join(line[1] for line in result)


def _strip_repeated_headers_footers(pages: list[PageResult]) -> None:
    """Mutates `pages` in place, removing lines that repeat across most pages.

    A running header/footer ("Company Confidential | Page X of Y") adds
    noise to every chunk it appears in and pollutes embeddings. We detect it
    structurally (repeats on >50% of pages) rather than hardcoding patterns,
    since every PDF's header/footer text is different.
    """
    if len(pages) < 4:
        return
    first_lines = Counter()
    last_lines = Counter()
    for p in pages:
        lines = [l.strip() for l in p.text.splitlines() if l.strip()]
        if lines:
            first_lines[lines[0]] += 1
            last_lines[lines[-1]] += 1
    threshold = len(pages) * config.HEADER_FOOTER_REPEAT_THRESHOLD
    noisy_lines = {l for l, c in first_lines.items() if c >= threshold}
    noisy_lines |= {l for l, c in last_lines.items() if c >= threshold}
    if not noisy_lines:
        return
    for p in pages:
        lines = [l for l in p.text.splitlines() if l.strip() not in noisy_lines]
        p.text = "\n".join(lines)


def extract_pdf(path) -> DocumentResult:
    doc = fitz.open(path)
    result = DocumentResult(filename=path.name)
    sample_for_lang = []

    for i, page in enumerate(doc, start=1):
        text = page.get_text("text")
        method = "native"
        if len(text.strip()) < config.MIN_NATIVE_CHARS_PER_PAGE:
            text = _ocr_page(page)
            method = "ocr"
            result.ocr_page_count += 1
        result.pages.append(PageResult(page_num=i, text=text, method=method))
        if method == "native" and len(sample_for_lang) < 5 and text.strip():
            sample_for_lang.append(text)

    doc.close()

    _strip_repeated_headers_footers(result.pages)

    try:
        result.language = detect(" ".join(sample_for_lang)[:3000]) if sample_for_lang else "unknown"
    except LangDetectException:
        result.language = "unknown"

    return result
