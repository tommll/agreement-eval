"""Field-type-aware normalization.

Every comparison in this project - prediction vs. gold, and prediction vs.
prediction - goes through the same normalizer for the field's type. That is
deliberate: if accuracy and agreement used different notions of equality, the
central question ("does agreement predict correctness?") would be answered by
an artifact of the metric rather than by the models.

`normalize(value, field_type)` returns either None (the value means "absent")
or a canonical string. Two values agree iff their normalized forms are equal.
`jaccard_similarity` adds a graded score on the same normalized forms, for
partial credit on free-text fields.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

from dateutil import parser as date_parser

# Strings that models and annotators use to mean "this field is not present".
NULL_TOKENS = {
    "", "-", "--", "n/a", "na", "n.a.", "none", "null", "nil", "unknown",
    "not available", "not found", "not present", "not specified", "missing",
    "no", "no value", "empty", "[]", "{}", "?",
}

_ORG_SUFFIXES = [
    "sdn bhd", "sdn. bhd.", "sdn bhd.", "bhd", "pte ltd", "pte. ltd.", "pty ltd",
    "co ltd", "co. ltd.", "company limited", "limited", "ltd", "llc", "l.l.c.",
    "inc", "incorporated", "corp", "corporation", "gmbh", "s.a.", "plc",
]

_ADDRESS_ABBREV = {
    "jalan": "jln", "jln.": "jln", "road": "rd", "rd.": "rd", "street": "st",
    "st.": "st", "avenue": "ave", "ave.": "ave", "boulevard": "blvd",
    "blvd.": "blvd", "number": "no", "no.": "no", "lorong": "lrg",
    "taman": "tmn", "bandar": "bdr", "floor": "flr", "suite": "ste",
    "apartment": "apt", "building": "bldg", "drive": "dr", "dr.": "dr",
}

_WS = re.compile(r"\s+")
_PUNCT_EDGE = re.compile(r"^[\s\.,;:\-_'\"()\[\]]+|[\s\.,;:\-_'\"()\[\]]+$")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_MONEY_CLEAN = re.compile(r"[^0-9.,\-]")
_DIGITS = re.compile(r"\d")


def _basic(value: str) -> str:
    """Unicode-normalize, collapse whitespace, trim edge punctuation."""
    text = unicodedata.normalize("NFKC", value)
    text = text.replace("’", "'").replace("–", "-").replace("—", "-")
    text = text.replace("&", " and ")  # "NO 2 & 4" == "NO 2 AND 4"
    text = _WS.sub(" ", text).strip()
    return _PUNCT_EDGE.sub("", text)


def is_null(value: Optional[object]) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and value != value:  # NaN
        return True
    text = str(value).strip().lower()
    return text in NULL_TOKENS


def normalize_text(value: str) -> str:
    return _NON_ALNUM.sub(" ", _basic(value).lower()).strip()


def normalize_org(value: str) -> str:
    text = normalize_text(value)
    changed = True
    while changed:  # e.g. "ABC ENTERPRISE SDN BHD." -> "abc enterprise"
        changed = False
        for suffix in _ORG_SUFFIXES:
            suf = _NON_ALNUM.sub(" ", suffix).strip()
            if suf and text.endswith(" " + suf):
                text = text[: -len(suf) - 1].strip()
                changed = True
    return text


def normalize_address(value: str) -> str:
    tokens = normalize_text(value).split()
    tokens = [_ADDRESS_ABBREV.get(tok, tok) for tok in tokens]
    return " ".join(tokens)


# Country calling codes we strip so that "07-388 2218" == "+60 7 388 2218".
# A prefix is only stripped when what remains is a plausible national number
# (7-10 digits), which keeps US/local numbers that merely start with these
# digits intact.
_COUNTRY_CODES = ("852", "886", "971", "44", "60", "61", "62", "63", "65", "66",
                  "81", "84", "86", "91", "1")


def normalize_phone(value: str) -> str:
    digits = re.sub(r"\D", "", _basic(value))
    for code in _COUNTRY_CODES:
        if digits.startswith(code) and 7 <= len(digits) - len(code) <= 10:
            digits = digits[len(code):]
            break
    return digits.lstrip("0")


def normalize_money(value: str) -> Optional[str]:
    """Return a canonical '123.45' string, or None if no number is present.

    Works off the raw string rather than _basic(), which strips edge
    punctuation and would swallow the sign of a refund line.
    """
    raw = unicodedata.normalize("NFKC", str(value)).strip()
    text = _MONEY_CLEAN.sub("", raw)
    if not _DIGITS.search(text):
        return None
    negative = bool(re.match(r"^\s*[-(]", raw)) or raw.endswith("-") or text.startswith("-")
    text = text.replace("-", "")
    # Decide which separator is the decimal point: the last one, if it is
    # followed by exactly 1-2 digits. Otherwise both are thousand separators.
    last_dot, last_comma = text.rfind("."), text.rfind(",")
    sep = max(last_dot, last_comma)
    if sep >= 0 and len(text) - sep - 1 in (1, 2):
        whole = re.sub(r"[.,]", "", text[:sep])
        frac = text[sep + 1:]
    else:
        whole, frac = re.sub(r"[.,]", "", text), ""
    whole = whole or "0"
    try:
        amount = float(f"{whole}.{frac or '0'}")
    except ValueError:
        return None
    if negative:
        amount = -amount
    return f"{amount:.2f}"


def normalize_integer(value: str) -> Optional[str]:
    digits = re.findall(r"-?\d+", _basic(value))
    return str(int(digits[0])) if digits else None


def normalize_date(value: str) -> Optional[str]:
    """Return ISO 'YYYY-MM-DD', or None if unparseable.

    Receipt dates are overwhelmingly day-first (SROIE is Malaysian), so
    dayfirst=True is tried before the US ordering.
    """
    text = _basic(value)
    if not _DIGITS.search(text):
        return None
    text = re.sub(r"\b(\d{1,2}):(\d{2})(:(\d{2}))?\s*(am|pm)?\b", " ", text, flags=re.I).strip()
    for dayfirst in (True, False):
        try:
            parsed = date_parser.parse(text, dayfirst=dayfirst, fuzzy=True)
            return parsed.date().isoformat()
        except (ValueError, OverflowError, TypeError):
            continue
    return None


_NORMALIZERS = {
    "text": normalize_text,
    "org": normalize_org,
    "address": normalize_address,
    "date": normalize_date,
    "money": normalize_money,
    "integer": normalize_integer,
    "phone": normalize_phone,
}


def normalize(value: Optional[object], field_type: str = "text") -> Optional[str]:
    """Canonical form of `value` for its field type, or None for 'absent'.

    A value that cannot be parsed as its type (e.g. "N/A" for money) also
    normalizes to None - it carries no more information than an omission.
    """
    if is_null(value):
        return None
    fn = _NORMALIZERS.get(field_type, normalize_text)
    result = fn(str(value))
    if result is None or not str(result).strip():
        return None
    return str(result)


def values_equal(a: Optional[object], b: Optional[object], field_type: str = "text") -> bool:
    """Exact match after normalization. None == None is True (both absent)."""
    return normalize(a, field_type) == normalize(b, field_type)


# --------------------------------------------------------------------------
# Jaccard: a graded companion to exact match
# --------------------------------------------------------------------------

# Field types whose value is atomic. A money amount that shares a token with
# the right one is still the wrong amount, so these score 1 or 0 - token
# overlap would reward "438.20" for "436.20" if tokenized by character.
ATOMIC_TYPES = frozenset({"date", "money", "integer", "phone"})

# Default cut-off for scoring a prediction correct under Jaccard. At 0.8 a
# 12-token address survives one wrong token (an OCR typo in the label), but not
# a dropped street line.
DEFAULT_JACCARD_THRESHOLD = 0.8


def jaccard_similarity(a_norm: Optional[str], b_norm: Optional[str], field_type: str = "text") -> float:
    """Token-set Jaccard |A & B| / |A | B| between two *normalized* values.

    Null handling mirrors exact match: both absent is full agreement, absent
    vs. present is none. Atomic types (dates, amounts, ids) are all-or-nothing.
    The distance 1 - J is a metric, so the score is safe to average or cluster
    on - but a threshold on it is not transitive, which is why agreement
    clusters (majority vote, Dawid-Skene) stay on exact match.
    """
    if a_norm is None or b_norm is None:
        return 1.0 if a_norm is None and b_norm is None else 0.0
    if field_type in ATOMIC_TYPES:
        return 1.0 if a_norm == b_norm else 0.0
    a, b = set(str(a_norm).split()), set(str(b_norm).split())
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def jaccard_distance(a_norm: Optional[str], b_norm: Optional[str], field_type: str = "text") -> float:
    return 1.0 - jaccard_similarity(a_norm, b_norm, field_type)
