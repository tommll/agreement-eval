"""Prompt variants.

Three variants that differ in the ways that plausibly change *where* a model
is wrong, not just how often - which is what makes agreement between them
informative:

    terse    - bare schema, no guidance. Models fall back to their own priors,
               especially about when to emit null.
    detailed - explicit null policy, verbatim-copy rule, format rules.
    evidence - detailed, plus a required verbatim evidence span per field.
               Grounding each value is meant to suppress invented values.

The prompt text is versioned by hash; the version is stored with every
extraction so a prompt edit can never be silently mixed into an old run.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from ..documents import Document
from ..schemas import DatasetSchema

_BASE_RULES = """\
Rules:
- Copy values verbatim from the document. Do not reformat, reorder, complete, or correct them.
- If a field is not present in the document, its value is null. Never guess, never infer a
  plausible value, and never carry a value over from a different field.
- A field that is present but unreadable is also null.
- Return every field in the schema exactly once."""

_EVIDENCE_RULES = """\
For each field also return `evidence`: the verbatim line(s) from the document that the value was
read from. If you cannot point to a supporting line, the value must be null and evidence must be
null. Decide the evidence first, then the value."""


@dataclass(frozen=True)
class Prompt:
    name: str
    with_evidence: bool
    system_fn: Callable[[DatasetSchema], str]
    user_fn: Callable[[DatasetSchema, Document, int], str]

    def system(self, schema: DatasetSchema) -> str:
        return self.system_fn(schema)

    def user(self, schema: DatasetSchema, doc: Document, max_ocr_chars: int = 8000) -> str:
        return self.user_fn(schema, doc, max_ocr_chars)

    def version(self, schema: DatasetSchema) -> str:
        """Short hash over the rendered prompt text for this schema."""
        probe = Document(doc_id="probe", dataset=schema.name, gold={}, ocr_text="<DOC>")
        blob = f"{self.name}|{self.system(schema)}|{self.user_fn(schema, probe, 8000)}"
        return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _document_block(doc: Document, max_ocr_chars: int) -> str:
    if not doc.ocr_text:
        return "(no OCR text available - read the attached image)"
    text = doc.ocr_text
    if len(text) > max_ocr_chars:
        # Receipts put the merchant at the top and the total at the bottom, so
        # keep both ends rather than truncating one side.
        head = text[: max_ocr_chars // 2]
        tail = text[-max_ocr_chars // 2:]
        text = f"{head}\n...[{len(doc.ocr_text) - max_ocr_chars} characters elided]...\n{tail}"
    return text


def _terse_system(schema: DatasetSchema) -> str:
    return (
        f"Extract structured fields from a {schema.name} document. "
        "Return JSON matching the provided schema. Use null for fields that are absent."
    )


def _terse_user(schema: DatasetSchema, doc: Document, max_ocr_chars: int) -> str:
    return f"Document:\n<document>\n{_document_block(doc, max_ocr_chars)}\n</document>\n\nExtract: {', '.join(schema.field_names)}."


def _detailed_system(schema: DatasetSchema) -> str:
    return (
        "You are an information extraction system for scanned documents. You read noisy OCR text "
        "and scanned images and return exactly the requested fields.\n\n"
        f"Fields:\n{schema.describe_fields()}\n\n{_BASE_RULES}"
    )


def _detailed_user(schema: DatasetSchema, doc: Document, max_ocr_chars: int) -> str:
    return (
        f"<document id=\"{doc.doc_id}\">\n{_document_block(doc, max_ocr_chars)}\n</document>\n\n"
        "Extract the fields defined in the schema from this document."
    )


def _evidence_system(schema: DatasetSchema) -> str:
    return f"{_detailed_system(schema)}\n\n{_EVIDENCE_RULES}"


def _evidence_user(schema: DatasetSchema, doc: Document, max_ocr_chars: int) -> str:
    return (
        f"<document id=\"{doc.doc_id}\">\n{_document_block(doc, max_ocr_chars)}\n</document>\n\n"
        "For every field, quote the supporting line as `evidence`, then give the `value` read from it."
    )


PROMPTS: Dict[str, Prompt] = {
    "terse": Prompt("terse", False, _terse_system, _terse_user),
    "detailed": Prompt("detailed", False, _detailed_system, _detailed_user),
    "evidence": Prompt("evidence", True, _evidence_system, _evidence_user),
}


def get_prompt(name: str) -> Prompt:
    try:
        return PROMPTS[name]
    except KeyError:
        raise KeyError(f"unknown prompt variant {name!r}; have {sorted(PROMPTS)}") from None
