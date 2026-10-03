"""
Re-downloads the 10-PDF corpus from the source URLs recorded in
data/corpus_manifest.json -- makes the corpus reproducible on a different
machine without needing to re-run the sourcing/verification search.

Run from the project root: python scripts/download_corpus.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from src import config

RETRIES = 3
TIMEOUT_S = 60


def download_one(url: str, dest: Path) -> None:
    for attempt in range(1, RETRIES + 1):
        try:
            resp = requests.get(url, timeout=TIMEOUT_S, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            return
        except Exception as e:
            print(f"    attempt {attempt}/{RETRIES} failed: {e}")
            if attempt < RETRIES:
                time.sleep(3)
    raise RuntimeError(f"Failed to download {url} after {RETRIES} attempts")


def main():
    config.PDF_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))

    present, fetched = 0, 0
    for entry in manifest:
        dest = config.PDF_DIR / entry["filename"]
        if dest.exists():
            print(f"[skip] {entry['filename']} already present")
            present += 1
            continue
        print(f"[fetch] {entry['filename']} <- {entry['source_url']}")
        download_one(entry["source_url"], dest)
        fetched += 1

    print(f"\n{present + fetched}/{len(manifest)} files present ({fetched} freshly downloaded).")


if __name__ == "__main__":
    main()
