"""Field schemas for each dataset.

A schema is the contract shared by three things: what we ask the models for,
what the gold labels contain, and how each field gets normalized before it is
compared. Keeping them in one place is what makes per-field analysis possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

# How a field's value should be normalized before comparison. See normalize.py.
FIELD_TYPES = ("text", "org", "address", "date", "money", "integer", "phone")


@dataclass(frozen=True)
class FieldSpec:
    name: str
    type: str
    description: str
    # Fields that are frequently absent in the source document. The populated
    # vs. null split in the analysis uses the gold label, not this flag; this
    # is only a hint carried into the prompt.
    often_absent: bool = False

    def __post_init__(self) -> None:
        if self.type not in FIELD_TYPES:
            raise ValueError(f"unknown field type {self.type!r} for {self.name!r}")


@dataclass(frozen=True)
class DatasetSchema:
    name: str
    fields: Tuple[FieldSpec, ...]

    @property
    def field_names(self) -> Tuple[str, ...]:
        return tuple(f.name for f in self.fields)

    def get(self, name: str) -> FieldSpec:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(name)

    def field_type(self, name: str) -> str:
        return self.get(name).type

    def empty_record(self) -> Dict[str, Optional[str]]:
        return {name: None for name in self.field_names}

    def describe_fields(self) -> str:
        lines = []
        for f in self.fields:
            hint = " (often absent from the document)" if f.often_absent else ""
            lines.append(f"- {f.name} [{f.type}]: {f.description}{hint}")
        return "\n".join(lines)

    def json_schema(self, with_evidence: bool = False) -> Dict[str, Any]:
        """JSON schema for output_config.format / structured outputs.

        Every field is required and nullable: the model must make an explicit
        null decision rather than omitting a key, otherwise "absent" and
        "not extracted" become indistinguishable in the analysis.
        """
        props: Dict[str, Any] = {}
        for f in self.fields:
            # anyOf (not a type array) - structured outputs documents anyOf as
            # supported, and it is the portable spelling for "string or null".
            value_schema = {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "description": f.description,
            }
            if with_evidence:
                props[f.name] = {
                    "type": "object",
                    "properties": {
                        "evidence": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "description": "Verbatim snippet from the document supporting the value, or null.",
                        },
                        "value": value_schema,
                    },
                    "required": ["evidence", "value"],
                    "additionalProperties": False,
                }
            else:
                props[f.name] = value_schema
        return {
            "type": "object",
            "properties": props,
            "required": list(self.field_names),
            "additionalProperties": False,
        }


SROIE = DatasetSchema(
    name="sroie",
    fields=(
        FieldSpec("company", "org", "Merchant/company name as printed on the receipt header."),
        FieldSpec("date", "date", "Date of the transaction."),
        FieldSpec("address", "address", "Full merchant address."),
        FieldSpec("total", "money", "Grand total amount paid, numeric only."),
    ),
)

# FUNSD has no fixed schema; we project each form onto a fixed set of common
# keys via question-text aliases (see ingest/funsd.py). Most fields are null on
# most forms, which is exactly the regime the populated/null split probes.
FUNSD = DatasetSchema(
    name="funsd",
    fields=(
        FieldSpec("date", "date", "Date written on the form.", often_absent=True),
        FieldSpec("to", "text", "Recipient of the form/fax.", often_absent=True),
        FieldSpec("from", "text", "Sender of the form/fax.", often_absent=True),
        FieldSpec("subject", "text", "Subject or RE line.", often_absent=True),
        FieldSpec("company", "org", "Company named on the form.", often_absent=True),
        FieldSpec("fax_number", "phone", "Fax number.", often_absent=True),
        FieldSpec("phone_number", "phone", "Telephone number.", often_absent=True),
        FieldSpec("total_pages", "integer", "Total page count of the fax/form.", often_absent=True),
    ),
)

# Synthetic offline receipts, same shape as SROIE plus optional fields so the
# demo exercises the populated/null split without any download.
SYNTHETIC = DatasetSchema(
    name="synthetic",
    fields=(
        FieldSpec("company", "org", "Merchant/company name as printed on the receipt header."),
        FieldSpec("date", "date", "Date of the transaction."),
        FieldSpec("address", "address", "Full merchant address."),
        FieldSpec("total", "money", "Grand total amount paid, numeric only."),
        FieldSpec("tax", "money", "Tax amount charged.", often_absent=True),
        FieldSpec("phone", "phone", "Merchant phone number.", often_absent=True),
    ),
)

SCHEMAS: Dict[str, DatasetSchema] = {
    s.name: s for s in (SROIE, FUNSD, SYNTHETIC)
}


def get_schema(name: str) -> DatasetSchema:
    try:
        return SCHEMAS[name]
    except KeyError:
        raise KeyError(f"unknown dataset schema {name!r}; have {sorted(SCHEMAS)}") from None


def iter_schemas() -> Iterable[DatasetSchema]:
    return SCHEMAS.values()
