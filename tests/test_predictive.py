import numpy as np
import pandas as pd
import pytest

from agreement_eval.metrics import predictive as pv


def _items(rows):
    """rows: (pairwise_agreement, unanimous, consensus_correct, gold_populated)"""
    return pd.DataFrame([
        {"doc_id": f"d{i}", "field": "total", "pairwise_agreement": a, "unanimous": u,
         "consensus_correct": c, "gold_populated": g, "consensus_label": "x" if g else "∅",
         "consensus_frac": a, "entropy": 1 - a}
        for i, (a, u, c, g) in enumerate(rows)
    ])


def test_roc_auc_known_value():
    assert pv.roc_auc([0.1, 0.4, 0.35, 0.8], [False, False, True, True]) == 0.75


def test_roc_auc_all_ties_is_chance():
    assert pv.roc_auc([1, 1, 1, 1], [True, False, True, False]) == 0.5


def test_roc_auc_single_class_is_nan():
    assert np.isnan(pv.roc_auc([0.1, 0.9], [True, True]))


def test_wilson_ci_brackets_the_estimate():
    low, high = pv.wilson_ci(95, 100)
    assert low < 0.95 < high and high <= 1.0


def test_spearman_matches_perfect_rank_agreement():
    assert pv.spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert pv.spearman([1, 2, 3], [30, 20, 10]) == pytest.approx(-1.0)


def test_bucketize_uses_distinct_values_when_few():
    buckets = pv.bucketize(pd.Series([0.0, 0.5, 1.0, 1.0]))
    assert list(buckets.categories) == ["0%", "50%", "100%"]


def test_trust_rule_counts_and_coverage():
    items = _items([
        (1.0, True, True, True),
        (1.0, True, False, True),     # unanimously wrong
        (0.3, False, True, True),
        (0.3, False, False, False),
    ])
    row = pv.trust_rule(items).set_index("split").loc["all"]
    assert row["n_items"] == 4
    assert row["coverage"] == 0.5
    assert row["p_correct_when_rule_fires"] == 0.5
    assert row["errors_missed"] == 1      # the unanimously wrong item slips through
    assert row["errors_caught"] == 1
    assert row["errors_per_1000_accepted"] == 500


def test_trust_rule_splits_populated_and_null():
    items = _items([(1.0, True, True, False), (1.0, True, True, True), (0.2, False, False, True)])
    table = pv.trust_rule(items).set_index("split")
    assert set(table.index) == {"all", "gold_populated=True", "gold_populated=False"}
    assert table.loc["gold_populated=False", "n_items"] == 1


def test_calibration_gap_is_signed():
    items = _items([(1.0, True, False, True)] * 10)
    table, ece = pv.calibration(items, n_bins=5)
    assert table.iloc[0]["gap"] == -1.0   # agreement said 100%, accuracy was 0
    assert ece == pytest.approx(1.0)


def test_headline_splits_by_gold_state():
    items = _items([(1.0, True, True, False)] * 5 + [(0.2, False, False, True)] * 5)
    head = pv.headline(items)
    assert head["n_items"] == 10
    assert head["null"]["n"] == 5 and head["populated"]["n"] == 5
    assert head["p_correct_given_unanimous"] == 1.0
    assert head["p_correct_given_split"] == 0.0


def test_discrimination_inverts_entropy_to_a_common_scale():
    items = _items([(1.0, True, True, True)] * 5 + [(0.0, False, False, True)] * 5)
    table = pv.discrimination(items).set_index(["score", "split"])
    assert table.loc[("pairwise_agreement", "all"), "auc"] == 1.0
    assert table.loc[("entropy (inverted)", "all"), "auc"] == 1.0
