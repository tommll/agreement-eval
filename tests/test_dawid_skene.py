import random

import pytest

from agreement_eval.metrics.agreement import NULL_LABEL
from agreement_eval.metrics.dawid_skene import (binary_dawid_skene, majority_vote,
                                                one_coin_dawid_skene, score_aggregation)


def _correlated_items(n=300, seed=0):
    """Two configs that err *together*, one that errs alone but rarely.

    Majority vote loses whenever the correlated pair is wrong; Dawid-Skene can
    learn that the third config is the reliable one.
    """
    rng = random.Random(seed)
    items, gold = {}, {}
    for i in range(n):
        key = (f"d{i}", "total")
        truth = f"v{i}"
        gold[key] = truth
        pair_wrong = rng.random() < 0.35
        bad = f"w{i}"
        items[key] = {
            "corr-a": bad if pair_wrong else truth,
            "corr-b": bad if pair_wrong else truth,
            "good": truth if rng.random() > 0.05 else f"z{i}",
        }
    return items, gold


def _one_reliable_rater(n=300, seed=1):
    """One accurate config, two noisy ones whose wrong answers differ.

    Majority vote has nothing to work with when all three disagree; Dawid-Skene
    can learn which config to follow.
    """
    rng = random.Random(seed)
    items, gold = {}, {}
    for i in range(n):
        key = (f"d{i}", "total")
        truth = f"v{i}"
        gold[key] = truth
        items[key] = {
            "reliable": truth if rng.random() > 0.05 else f"x{i}",
            "noisy1": truth if rng.random() > 0.5 else f"n1-{i}-{rng.randint(0, 9)}",
            "noisy2": truth if rng.random() > 0.5 else f"n2-{i}-{rng.randint(0, 9)}",
        }
    return items, gold


def test_majority_vote_is_beaten_by_the_correlated_pair():
    items, gold = _correlated_items()
    mv = score_aggregation(majority_vote(items), gold)
    assert mv["accuracy"] < 0.75  # the pair carries every vote it takes part in


def test_one_coin_dawid_skene_beats_majority_when_one_config_is_reliable():
    items, gold = _one_reliable_rater()
    ds = one_coin_dawid_skene(items)
    mv = majority_vote(items)
    assert ds.worker_quality["reliable"] > ds.worker_quality["noisy1"]
    assert ds.worker_quality["reliable"] > ds.worker_quality["noisy2"]
    assert score_aggregation(ds, gold)["accuracy"] > score_aggregation(mv, gold)["accuracy"] + 0.1


def test_dawid_skene_inherits_the_correlated_error_blind_spot():
    """Documents a limitation, not a bug.

    Dawid-Skene assumes raters err independently given the truth. Two configs
    that make the *same* mistake (same base model, same prompt family) break
    that assumption, and DS then ratifies their agreement exactly as majority
    vote does. No aggregation method recovers a truth that only one rater saw
    if the others are correlated - this is why the study reports the ceiling
    ('any config correct') next to every aggregator.
    """
    items, gold = _correlated_items()
    ds = score_aggregation(one_coin_dawid_skene(items), gold)["accuracy"]
    mv = score_aggregation(majority_vote(items), gold)["accuracy"]
    assert abs(ds - mv) < 0.05


def test_one_coin_matches_majority_when_errors_are_independent():
    rng = random.Random(3)
    items, gold = {}, {}
    for i in range(200):
        key = (f"d{i}", "f")
        truth = f"v{i}"
        gold[key] = truth
        items[key] = {w: (truth if rng.random() > 0.2 else f"{w}-wrong{i}") for w in ("a", "b", "c")}
    ds = score_aggregation(one_coin_dawid_skene(items), gold)
    mv = score_aggregation(majority_vote(items), gold)
    assert ds["accuracy"] >= mv["accuracy"] - 0.02


def _lazy_raters(n=300, seed=5):
    """One capable config, two that answer "absent" more often than not.

    The lazy pair agrees constantly - on null - without that agreement carrying
    any information. This is the shape real extraction runs take when a prompt
    makes a model give up, and it is what the null-aware emission model exists
    to handle.
    """
    rng = random.Random(seed)
    items, gold = {}, {}
    for i in range(n):
        key = (f"d{i}", "total")
        truth = f"v{i}"
        gold[key] = truth
        items[key] = {
            "capable": truth if rng.random() > 0.2 else f"x{i}",
            "lazy1": NULL_LABEL if rng.random() < 0.6 else truth,
            "lazy2": NULL_LABEL if rng.random() < 0.6 else truth,
        }
    return items, gold


def test_shared_nulls_do_not_make_lazy_configs_look_reliable():
    items, gold = _lazy_raters()
    aware = one_coin_dawid_skene(items, null_aware=True)
    assert aware.worker_quality["capable"] > aware.worker_quality["lazy1"]
    assert aware.worker_quality["capable"] > aware.worker_quality["lazy2"]
    assert score_aggregation(aware, gold)["accuracy"] > score_aggregation(majority_vote(items), gold)["accuracy"]


def test_textbook_model_is_fooled_by_shared_nulls():
    """Why null_aware defaults to True - the plain model inverts the ranking."""
    items, gold = _lazy_raters()
    naive = one_coin_dawid_skene(items, null_aware=False)
    aware = one_coin_dawid_skene(items, null_aware=True)
    assert naive.worker_quality["capable"] < max(naive.worker_quality["lazy1"], naive.worker_quality["lazy2"])
    assert score_aggregation(naive, gold)["accuracy"] < score_aggregation(aware, gold)["accuracy"]


def test_binary_dawid_skene_learns_an_asymmetric_config():
    rng = random.Random(11)
    items, gold = {}, {}
    for i in range(300):
        key = (f"d{i}", "f")
        truth_populated = rng.random() < 0.6
        gold[key] = "v" if truth_populated else NULL_LABEL
        votes = {}
        for w, p_fill in (("careful", 0.02), ("eager", 0.45), ("normal", 0.1)):
            if truth_populated:
                votes[w] = "v" if rng.random() > 0.05 else NULL_LABEL
            else:
                votes[w] = "v" if rng.random() < p_fill else NULL_LABEL
        items[key] = votes
    agg = binary_dawid_skene(items)
    # 'eager' invents values on absent fields: its null->null rate must be worst.
    assert agg.worker_quality["eager|p(null->null)"] < agg.worker_quality["careful|p(null->null)"]
    assert score_aggregation(agg, gold, binary=True)["accuracy"] > 0.85


def test_empty_input_is_handled():
    assert one_coin_dawid_skene({}).labels == {}
    assert binary_dawid_skene({}).labels == {}
