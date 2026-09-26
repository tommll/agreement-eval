import pytest
import pandas as pd

from agreement_eval.metrics.agreement import (NULL_LABEL, cluster_labels, item_agreement,
                                              item_stats, pairwise_matrix, pairwise_table)


def _preds(rows):
    return pd.DataFrame([
        {"doc_id": d, "field": f, "field_type": "text", "config_id": c,
         "value_raw": v, "value_norm": v, "gold_norm": g, "is_null": v is None}
        for d, f, c, v, g in rows
    ])


def test_item_stats_unanimous():
    stats = item_stats({"a": "x", "b": "x", "c": "x"})
    assert stats["unanimous"] and stats["pairwise_agreement"] == 1.0
    assert stats["consensus_label"] == "x" and stats["consensus_support"] == 3


def test_item_stats_split():
    stats = item_stats({"a": "x", "b": "x", "c": "y", "d": NULL_LABEL})
    assert stats["n_pairs"] == 6
    assert stats["n_agreeing_pairs"] == 1          # only (a, b)
    assert stats["pairwise_agreement"] == 1 / 6
    assert stats["consensus_label"] == "x"
    assert stats["n_null"] == 1 and stats["any_null"] and not stats["all_null"]


def test_cluster_labels_merges_judged_equivalents():
    labels = {"a": "9.00", "b": "rm9", "c": "10.00"}
    merged = cluster_labels(labels, [("9.00", "rm9")])
    assert merged["a"] == merged["b"] != merged["c"]


def test_cluster_labels_never_merges_null_with_a_value():
    merged = cluster_labels({"a": NULL_LABEL, "b": "x"}, [(NULL_LABEL, "x")])
    assert merged["a"] != merged["b"]


def test_item_agreement_scores_consensus_against_gold():
    preds = _preds([
        ("d1", "total", "a", "9.00", "9.00"),
        ("d1", "total", "b", "9.00", "9.00"),
        ("d1", "total", "c", "8.00", "9.00"),
    ])
    items = item_agreement(preds)
    row = items.iloc[0]
    assert row["consensus_label"] == "9.00"
    assert bool(row["consensus_correct"]) is True
    assert row["pairwise_agreement"] == 1 / 3
    assert row["n_correct"] == 2


def test_item_agreement_with_judge_merges_into_sem_columns():
    preds = _preds([
        ("d1", "total", "a", "9.00", "9.00"),
        ("d1", "total", "b", "rm9", "9.00"),
        ("d1", "total", "c", "8.00", "9.00"),
    ])
    judgments = pd.DataFrame([{
        "doc_id": "d1", "field": "total", "left_value": "9.00", "right_value": "rm9",
        "equivalent": True, "kind": "pair",
    }])
    row = item_agreement(preds, judgments).iloc[0]
    assert row["pairwise_agreement"] == 0.0        # exact match sees three answers
    assert row["sem_pairwise_agreement"] == 1 / 3  # judge merges two of them
    assert bool(row["sem_consensus_correct"]) is True


def test_pairwise_table_and_matrix():
    preds = _preds([
        ("d1", "total", "a", "9.00", "9.00"),
        ("d1", "total", "b", "9.00", "9.00"),
        ("d1", "total", "c", "8.00", "9.00"),
    ])
    pairs = pairwise_table(preds)
    assert len(pairs) == 3
    assert pairs.set_index(["config_a", "config_b"]).loc[("a", "b"), "exact_agree"]
    matrix = pairwise_matrix(pairs)
    assert matrix.loc["a", "b"] == 1.0 and matrix.loc["a", "c"] == 0.0
    assert matrix.loc["a", "a"] == 1.0


def test_item_agreement_adds_graded_jaccard_alongside_exact():
    # Exact match sees three different answers; Jaccard sees near-agreement.
    preds = _preds([
        ("d1", "addr", "a", "no 2 jln bayu 4 masai", "no 2 jln bayu 4 masai johor"),
        ("d1", "addr", "b", "no 2 jln bayu 4 masai johor", "no 2 jln bayu 4 masai johor"),
        ("d1", "addr", "c", "no 2 jln bayu masai johor", "no 2 jln bayu 4 masai johor"),
    ])
    item = item_agreement(preds, jaccard_threshold=0.8).iloc[0]
    assert item["pairwise_agreement"] == 0.0 and not item["unanimous"]
    # pairs: (a,b)=6/7, (a,c)=5/7, (b,c)=6/7
    assert item["jaccard_agreement"] == pytest.approx((6 / 7 + 5 / 7 + 6 / 7) / 3)
    # The exact consensus is a (lexicographic tie-break); 6/7 >= 0.8 to gold.
    assert item["consensus_correct_jaccard"] and not item["consensus_correct"]


def test_pairwise_table_carries_jaccard():
    preds = _preds([("d1", "f", "a", "x y", "x y"), ("d1", "f", "b", "x z", "x y")])
    pairs = pairwise_table(preds)
    assert pairs["jaccard"].iloc[0] == pytest.approx(1 / 3)
    assert pairwise_matrix(pairs, column="jaccard").loc["a", "b"] == pytest.approx(1 / 3)
