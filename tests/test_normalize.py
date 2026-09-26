import pytest

from agreement_eval.normalize import (is_null, jaccard_distance, jaccard_similarity, normalize,
                                      values_equal)


@pytest.mark.parametrize("raw,expected", [
    ("RM 1,234.50", "1234.50"), ("193.00", "193.00"), ("$12.3", "12.30"),
    ("12,50", "12.50"), ("1.234,56", "1234.56"), ("-5.00", "-5.00"),
    ("N/A", None), ("", None), ("no total", None),
])
def test_money(raw, expected):
    assert normalize(raw, "money") == expected


@pytest.mark.parametrize("raw", ["15/01/2019", "15/01/2019 11:05:16 AM", "2019-01-15", "Jan 15, 2019"])
def test_date_is_day_first_iso(raw):
    assert normalize(raw, "date") == "2019-01-15"


def test_date_unparseable():
    assert normalize("sometime last week", "date") is None


def test_org_suffixes_stripped():
    assert normalize("OJC MARKETING SDN BHD", "org") == normalize("Ojc Marketing", "org")
    assert normalize("ACME Corp.", "org") == "acme"


def test_address_abbreviations_and_ampersand():
    assert values_equal("NO 2 & 4, JALAN BAYU 4", "no 2 and 4 jln bayu 4", "address")


def test_phone_country_code():
    assert values_equal("TEL:07-388 2218", "+60 7 388 2218", "phone")
    # A local number that merely starts with a country code's digits is kept.
    assert not values_equal("(212) 555-1234", "718-555-1234", "phone")


@pytest.mark.parametrize("raw", [None, "", " ", "N/A", "none", "-", "not found"])
def test_null_tokens(raw):
    assert is_null(raw)
    assert normalize(raw, "text") is None


def test_both_null_are_equal():
    assert values_equal(None, "N/A", "money")


def test_unparseable_typed_value_is_null_not_a_string():
    # "N/A" in a money field carries no more information than an omission;
    # collapsing it to None keeps the populated/null split honest.
    assert normalize("N/A", "money") is None


def _jac(a, b, field_type):
    return jaccard_similarity(normalize(a, field_type), normalize(b, field_type), field_type)


def test_jaccard_null_semantics_match_exact_match():
    assert jaccard_similarity(None, None, "address") == 1.0
    assert jaccard_similarity(None, "jln bayu", "address") == 0.0
    assert jaccard_similarity("jln bayu", None, "text") == 0.0


def test_jaccard_is_all_or_nothing_for_atomic_types():
    # One wrong digit is a wrong amount, not a nearly-right one.
    assert _jac("438.20", "436.20", "money") == 0.0
    assert _jac("RM 193.00", "193.00", "money") == 1.0
    assert _jac("15/01/2019 11:05 AM", "2019-01-15", "date") == 1.0


def test_jaccard_gives_partial_credit_on_free_text():
    # Model read the receipt's 81750; the label carries the OCR typo B1750.
    # 11 of 13 distinct tokens are shared -> above the 0.8 default threshold.
    j = _jac("NO 2 & 4, JALAN BAYU 4, BANDAR SERI ALAM, 81750 MASAI, JOHOR",
             "NO 2 & 4, JALAN BAYU 4, BANDAR SERI ALAM, B1750 MASAI, JOHOR", "address")
    assert j == pytest.approx(11 / 13)
    # A dropped street line and state is a real miss, and stays below it.
    j = _jac("NO 2 & 4 BANDAR SERI ALAM 81750 MASAI",
             "NO 2 & 4, JALAN BAYU 4, BANDAR SERI ALAM, B1750 MASAI, JOHOR", "address")
    assert j == pytest.approx(8 / 13)


def test_jaccard_ignores_order_and_duplicates():
    assert jaccard_similarity("a b c", "c b a b", "text") == 1.0


def test_jaccard_distance_is_complement():
    assert jaccard_distance("a b", "b c", "text") == pytest.approx(1 - 1 / 3)
