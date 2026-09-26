"""The ingest-side document record, shared by every loader."""

from __future__ import annotations

import json
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, Optional


@dataclass
class Document:
    doc_id: str
    dataset: str
    gold: Dict[str, Optional[str]]
    split: Optional[str] = None
    ocr_text: Optional[str] = None
    image_uri: Optional[str] = None     # remote URL (lazily downloaded for vision configs)
    image_path: Optional[str] = None    # local path, once downloaded
    meta: Dict[str, Any] = dc_field(default_factory=dict)

    def to_row(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "dataset": self.dataset,
            "split": self.split,
            "gold": json.dumps(self.gold),
            "ocr_text": self.ocr_text,
            "image_uri": self.image_uri,
            "image_path": self.image_path,
            "meta": json.dumps(self.meta),
        }

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "Document":
        gold = row["gold"]
        meta = row.get("meta") or {}
        return cls(
            doc_id=row["doc_id"],
            dataset=row["dataset"],
            gold=json.loads(gold) if isinstance(gold, str) else dict(gold),
            split=row.get("split"),
            ocr_text=row.get("ocr_text"),
            image_uri=row.get("image_uri"),
            image_path=row.get("image_path"),
            meta=json.loads(meta) if isinstance(meta, str) else dict(meta),
        )
