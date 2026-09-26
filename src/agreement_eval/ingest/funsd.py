"""FUNSD ingestion (noisy scanned forms) from a local extract of the dataset.

FUNSD has no fixed schema - each form is a graph of linked question/answer
entities. We project every form onto the fixed schema in schemas.FUNSD by
matching question text against aliases. Most fields are absent on most forms,
which is precisely the null-heavy regime the populated/null split is about.

Expected layout (the standard release):
    <root>/dataset/training_data/annotations/*.json
    <root>/dataset/training_data/images/*.png
    <root>/dataset/testing_data/...
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..documents import Document
from ..schemas import get_schema

# Question-text aliases per schema field. Matching is on lowercased text with
# punctuation stripped, so "TO:" and "To" both hit "to".
ALIASES: Dict[str, List[str]] = {
    "date": ["date", "date:", "dated", "date of request", "submission date"],
    "to": ["to", "to:", "recipient", "send to", "attn", "attention"],
    "from": ["from", "from:", "sender", "originator"],
    "subject": ["subject", "subject:", "re", "re:", "regarding"],
    "company": ["company", "company name", "firm", "organization", "corporation"],
    "fax_number": ["fax", "fax number", "fax no", "fax #", "telefax"],
    "phone_number": ["phone", "telephone", "phone number", "tel", "tel no", "phone no"],
    "total_pages": ["pages", "total pages", "no of pages", "number of pages", "page count"],
}

_PUNCT = re.compile(r"[^a-z0-9 ]+")


def _key(text: str) -> str:
    return re.sub(r"\s+", " ", _PUNCT.sub(" ", (text or "").lower())).strip()


_ALIAS_INDEX = {alias: field for field, aliases in ALIASES.items() for alias in aliases}


def _reading_order(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def sort_key(e: Dict[str, Any]):
        box = e.get("box") or [0, 0, 0, 0]
        return (round(box[1] / 25), box[0])  # band rows ~25px, then left-to-right
    return sorted(entities, key=sort_key)


def _parse_form(path: Path, split: str, images_dir: Optional[Path]) -> Document:
    payload = json.loads(path.read_text())
    entities = payload.get("form", [])
    by_id = {e["id"]: e for e in entities}

    gold: Dict[str, Optional[str]] = {f: None for f in get_schema("funsd").field_names}
    for entity in entities:
        if entity.get("label") != "question":
            continue
        field = _ALIAS_INDEX.get(_key(entity.get("text", "")))
        if not field or gold.get(field):
            continue
        for link in entity.get("linking", []) or []:
            for other_id in link:
                other = by_id.get(other_id)
                if other is not None and other is not entity and other.get("label") == "answer":
                    text = (other.get("text") or "").strip()
                    if text:
                        gold[field] = text
                        break
            if gold.get(field):
                break

    ocr_text = "\n".join(
        e["text"].strip() for e in _reading_order(entities) if (e.get("text") or "").strip()
    )
    stem = path.stem
    image_path = None
    if images_dir:
        for ext in (".png", ".jpg", ".jpeg", ".tif"):
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists():
                image_path = str(candidate)
                break

    return Document(
        doc_id=f"funsd/{split}/{stem}",
        dataset="funsd",
        split=split,
        gold=gold,
        ocr_text=ocr_text or None,
        image_path=image_path,
        meta={"source_file": str(path), "n_entities": len(entities),
              "n_populated_gold": sum(1 for v in gold.values() if v)},
    )


def load_funsd(root: Path, split: str = "testing", limit: Optional[int] = None) -> List[Document]:
    root = Path(root)
    split_dir_name = {"test": "testing_data", "testing": "testing_data",
                      "train": "training_data", "training": "training_data"}.get(split, split)
    candidates = [root / "dataset" / split_dir_name, root / split_dir_name, root]
    base = next((c for c in candidates if (c / "annotations").is_dir()), None)
    if base is None:
        raise FileNotFoundError(
            f"no annotations/ directory under {root} - expected <root>/dataset/{split_dir_name}/annotations"
        )
    images_dir = base / "images" if (base / "images").is_dir() else None

    files = sorted((base / "annotations").glob("*.json"))
    if limit:
        files = files[:limit]
    return [_parse_form(f, split, images_dir) for f in files]
