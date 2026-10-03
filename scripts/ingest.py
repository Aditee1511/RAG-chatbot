"""
Ingestion pipeline: data/pdfs/*.pdf -> extract -> chunk -> embed -> LanceDB.

Run from the project root:
    python scripts/ingest.py

Note: unlike the Chroma version this replaced, LanceDB's `add` appends rather
than upserts by chunk id -- re-running ingest on an unchanged data/pdfs/ will
duplicate rows. Fine for a one-shot 4-hour build; a real incremental-ingest
story would need an explicit delete-by-id-then-add step.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

from src import chunking, config, embeddings, extraction
from src.vectorstore import VectorStore


def main():
    config.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    pdf_paths = sorted(config.PDF_DIR.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {config.PDF_DIR}. Run scripts/download_corpus.py first.")
        return

    store = VectorStore()
    report = []
    overall_start = time.perf_counter()

    for pdf_path in pdf_paths:
        doc_start = time.perf_counter()
        print(f"\n[ingest] {pdf_path.name}")

        doc = extraction.extract_pdf(pdf_path)
        print(f"  pages={len(doc.pages)} ocr_pages={doc.ocr_page_count} lang={doc.language}")

        doc_id = pdf_path.stem
        chunks = chunking.chunk_document(doc.pages, doc_id=doc_id, filename=pdf_path.name)
        print(f"  chunks={len(chunks)}")

        for batch_start in tqdm(range(0, len(chunks), config.EMBED_BATCH_SIZE), desc="  embedding"):
            batch = chunks[batch_start : batch_start + config.EMBED_BATCH_SIZE]
            vecs = embeddings.encode_passages([c.text for c in batch])
            store.add_chunks(batch, vecs)

        report.append(
            {
                "filename": pdf_path.name,
                "pages": len(doc.pages),
                "ocr_pages": doc.ocr_page_count,
                "language": doc.language,
                "chunks": len(chunks),
                "ingest_seconds": round(time.perf_counter() - doc_start, 2),
            }
        )

    total_seconds = round(time.perf_counter() - overall_start, 2)
    print(f"\n[ingest] done in {total_seconds}s. Total chunks in store: {store.count()}")

    index_status = store.build_ann_index()
    print(f"[ingest] ANN index: {index_status}")

    config.INGEST_REPORT_PATH.write_text(
        json.dumps(
            {
                "total_seconds": total_seconds,
                "total_chunks": store.count(),
                "ann_index": index_status,
                "documents": report,
            },
            indent=2,
        )
    )
    print(f"[ingest] report written to {config.INGEST_REPORT_PATH}")


if __name__ == "__main__":
    main()
