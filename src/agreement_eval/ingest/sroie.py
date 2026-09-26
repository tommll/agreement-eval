"""SROIE (ICDAR 2019 Task 3) ingestion from Hugging Face.

Source: https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE

Each row carries exactly what this study needs:
    entities  -> {company, date, address, total}   the gold labels
    words     -> OCR tokens in reading order        the text input
    image     -> the scanned receipt                the vision input

Two fetch paths:
    "api"     - the HF datasets-server rows endpoint, paginated 100 at a time.
                No large download; right for the usual few-hundred-document run.
    "parquet" - downloads the split's parquet (~200-350 MB) and reads it with
                pyarrow. Right for the full split, or for offline re-runs.
"""

from __future__ import annotations

import io
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from ..documents import Document
from ..schemas import get_schema

HF_DATASET = "jsdnrs/ICDAR2019-SROIE"
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
PARQUET_URL = "https://huggingface.co/datasets/{ds}/resolve/main/data/{split}-00000-of-00001.parquet"
PAGE_SIZE = 100
FIELDS = ("company", "date", "address", "total")
USER_AGENT = "agreement-eval/0.1 (+https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE)"


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _headers() -> Dict[str, str]:
    headers = {"User-Agent": USER_AGENT}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _iter_api_rows(split: str, limit: Optional[int]) -> Iterator[Dict[str, Any]]:
    offset, seen = 0, 0
    while True:
        query = urllib.parse.urlencode(
            {"dataset": HF_DATASET, "config": "default", "split": split,
             "offset": offset, "length": PAGE_SIZE}
        )
        payload = json.loads(_get(f"{ROWS_ENDPOINT}?{query}"))
        rows = payload.get("rows", [])
        if not rows:
            return
        for entry in rows:
            yield entry["row"]
            seen += 1
            if limit and seen >= limit:
                return
        offset += len(rows)
        if offset >= payload.get("num_rows_total", offset):
            return
        time.sleep(0.2)  # be polite to the public endpoint


def _clean(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _ocr_text(words: Optional[List[str]]) -> Optional[str]:
    if not words:
        return None
    lines = [w.strip() for w in words if w and w.strip()]
    return "\n".join(lines) or None


def _download_image(url: str, dest: Path) -> Optional[str]:
    if dest.exists() and dest.stat().st_size > 0:
        return str(dest)
    try:
        data = _get(url, timeout=120)
    except Exception:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return str(dest)


def load_sroie(
    split: str = "test",
    limit: Optional[int] = None,
    source: str = "api",
    images_dir: Optional[Path] = None,
    download_images: bool = False,
    parquet_path: Optional[Path] = None,
) -> List[Document]:
    schema = get_schema("sroie")
    images_dir = Path(images_dir) if images_dir else Path("data/raw/sroie/images") / split
    if source == "parquet":
        rows = _iter_parquet_rows(split, limit, parquet_path, images_dir if download_images else None)
    elif source == "api":
        rows = _iter_api_rows(split, limit)
    else:
        raise ValueError(f"unknown source {source!r}; use 'api' or 'parquet'")

    docs: List[Document] = []
    for row in rows:
        key = _clean(row.get("key")) or f"sroie-{len(docs):05d}"
        entities = row.get("entities") or {}
        gold = {f: _clean(entities.get(f)) for f in FIELDS}
        assert set(gold) == set(schema.field_names), "SROIE gold must cover the schema"

        image_uri, image_path = None, row.get("image_path")
        image = row.get("image")
        if isinstance(image, dict):
            image_uri = image.get("src")
        if download_images and image_uri and not image_path:
            image_path = _download_image(image_uri, images_dir / f"{key}.jpg")

        docs.append(
            Document(
                doc_id=f"sroie/{split}/{key}",
                dataset="sroie",
                split=split,
                gold=gold,
                ocr_text=_ocr_text(row.get("words")),
                image_uri=image_uri,
                image_path=image_path,
                meta={
                    "source_key": key,
                    "image_size": row.get("image_size"),
                    "n_words": len(row.get("words") or []),
                    "hf_dataset": HF_DATASET,
                },
            )
        )
    return docs


def _iter_parquet_rows(
    split: str,
    limit: Optional[int],
    parquet_path: Optional[Path],
    images_dir: Optional[Path],
) -> Iterator[Dict[str, Any]]:
    import pyarrow.parquet as pq  # local import: only the parquet path needs it

    path = Path(parquet_path) if parquet_path else Path("data/raw/sroie") / f"{split}.parquet"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = PARQUET_URL.format(ds=HF_DATASET, split=split)
        print(f"downloading {url} -> {path} (this is a few hundred MB)")
        req = urllib.request.Request(url, headers=_headers())
        with urllib.request.urlopen(req, timeout=600) as resp, open(path, "wb") as fh:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)

    table = pq.read_table(path)
    seen = 0
    for batch in table.to_batches(max_chunksize=64):
        for row in batch.to_pylist():
            image_path = None
            image = row.get("image")
            if images_dir is not None and isinstance(image, dict) and image.get("bytes"):
                dest = Path(images_dir) / f"{row.get('key')}.jpg"
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    dest.write_bytes(image["bytes"])
                image_path = str(dest)
            row["image"] = None
            row["image_path"] = image_path
            yield row
            seen += 1
            if limit and seen >= limit:
                return
