"""Label aggregation: majority vote vs. Dawid-Skene.

Dawid-Skene (1979) treats each rater's reliability as a latent parameter and
estimates it from the pattern of (dis)agreements alone - no labels. The
promise over majority vote is that a config that is right when it disagrees
gets to outvote two configs that are wrong together.

Two variants are implemented, for two reasons:

*one-coin* (`one_coin_dawid_skene`) - extraction answers are open vocabulary,
so the candidate set differs per item and a full per-class confusion matrix is
not identifiable. The one-coin model gives each config a single reliability
p_w: with probability p_w it emits the true value, otherwise it spreads its
mass uniformly over the other candidates seen for that item.

*binary* (`binary_dawid_skene`) - the populated/null decision *is* a fixed
two-class problem across items, so the full Dawid-Skene confusion matrix is
identifiable there. This is where a config's asymmetry (eager to fill fields
vs. eager to say null) actually shows up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .agreement import NULL_LABEL, cluster_labels, judge_equivalences, label_of

ItemKey = Tuple[str, str]
ItemLabels = Dict[ItemKey, Dict[str, str]]


@dataclass
class Aggregation:
    """Aggregated answers plus what the model learned about each config."""

    method: str
    labels: Dict[ItemKey, str]
    confidence: Dict[ItemKey, float]
    worker_quality: Dict[str, float] = field(default_factory=dict)
    n_iter: int = 0
    log_likelihood: float = float("nan")

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {"doc_id": k[0], "field": k[1], "method": self.method,
             "label": v, "confidence": self.confidence.get(k, float("nan"))}
            for k, v in self.labels.items()
        ]
        return pd.DataFrame(rows).sort_values(["doc_id", "field"]).reset_index(drop=True)


def items_from_predictions(
    preds: pd.DataFrame,
    judgments: Optional[pd.DataFrame] = None,
) -> Tuple[ItemLabels, Dict[ItemKey, str]]:
    """(item -> {config: label}, item -> gold label), judge-merged if available."""
    equivalences = judge_equivalences(judgments)
    items: ItemLabels = {}
    gold: Dict[ItemKey, str] = {}
    for (doc_id, field), group in preds.groupby(["doc_id", "field"], sort=True):
        key = (doc_id, field)
        labels = {r.config_id: label_of(r.value_norm) for r in group.itertuples()}
        gold_label = label_of(group["gold_norm"].iloc[0])
        if equivalences:
            merged = cluster_labels({**labels, "__gold__": gold_label}, equivalences.get(key, []))
            gold_label = merged.pop("__gold__")
            labels = merged
        items[key] = labels
        gold[key] = gold_label
    return items, gold


def majority_vote(items: ItemLabels) -> Aggregation:
    labels: Dict[ItemKey, str] = {}
    confidence: Dict[ItemKey, float] = {}
    for key, votes in items.items():
        counts: Dict[str, int] = {}
        for value in votes.values():
            counts[value] = counts.get(value, 0) + 1
        # Ties broken lexicographically so the method is deterministic.
        winner = min(sorted(counts), key=lambda v: (-counts[v], v))
        labels[key] = winner
        confidence[key] = counts[winner] / len(votes)
    quality = {
        w: float(np.mean([labels[k] == votes[w] for k, votes in items.items() if w in votes]))
        for w in {w for votes in items.values() for w in votes}
    }
    return Aggregation("majority_vote", labels, confidence, quality)


def _vocabulary_sizes(items: ItemLabels) -> Dict[str, int]:
    """Distinct answers seen per field - the size of the space a wrong config picks from."""
    seen: Dict[str, set] = {}
    for (_, field), votes in items.items():
        seen.setdefault(field, set()).update(votes.values())
    return {field: max(len(values), 2) for field, values in seen.items()}


def one_coin_dawid_skene(
    items: ItemLabels,
    max_iter: int = 200,
    tol: float = 1e-6,
    vocabulary_size: Optional[int] = None,
    null_aware: bool = True,
    min_reliability: float = 0.05,
    max_reliability: float = 0.999,
) -> Aggregation:
    """EM for the one-coin (homogeneous-error) Dawid-Skene model.

    With probability p_w config w emits the true value; otherwise it emits some
    other value from a vocabulary of size V. `vocabulary_size` is that V; left
    as None it is estimated per field as the number of distinct answers that
    field received across the whole run.

    V is not cosmetic. It sets how improbable it is for two configs to land on
    the *same* wrong value by chance, which is what decides whether one
    confident dissenter may override a pair that agrees. Taking V to be just
    the candidates present in an item (the naive choice) makes coincidental
    agreement free, and EM then runs away to a degenerate fixed point where one
    config is declared infallible and everyone else is noise.

    `null_aware` prices the one answer every config can always reach. "Field is
    absent" is not one of V arbitrary strings: a config that gives up often
    emits it constantly, so two such configs agree on null all the time without
    that agreement carrying information. Folding null into the uniform 1/V term
    treats shared silence as though it were a coincidence too unlikely to
    ignore, and EM duly crowns the laziest raters. Instead each config gets a
    second parameter nu_w - given that it is wrong, how often it answers null -
    estimated alongside p_w. Set null_aware=False for the textbook model.
    """
    workers = sorted({w for votes in items.values() for w in votes})
    if not workers:
        return Aggregation("dawid_skene_one_coin", {}, {})

    warm = majority_vote(items)
    reliability = {
        w: float(np.clip(warm.worker_quality.get(w, 0.7), 0.5, 0.95)) for w in workers
    }
    # Warm start for nu_w: how often this config answers null at all.
    null_rate = {}
    for w in workers:
        answers = [votes[w] for votes in items.values() if w in votes]
        observed = np.mean([a == NULL_LABEL for a in answers]) if answers else 0.3
        null_rate[w] = float(np.clip(observed, 0.02, 0.9))

    candidates = {key: sorted(set(votes.values())) for key, votes in items.items()}
    field_vocab = _vocabulary_sizes(items)
    posteriors: Dict[ItemKey, Dict[str, float]] = {}
    log_likelihood = float("nan")
    iteration = 0

    def log_emission(worker: str, observed: str, truth: str, vocab: int) -> float:
        p = reliability[worker]
        if observed == truth:
            return float(np.log(p))
        if not null_aware:
            return float(np.log((1.0 - p) / max(vocab - 1, 1)))
        if truth == NULL_LABEL:
            # Truth is "absent"; every wrong answer is some value.
            return float(np.log((1.0 - p) / max(vocab - 1, 1)))
        nu = null_rate[worker]
        if observed == NULL_LABEL:
            return float(np.log((1.0 - p) * nu))
        return float(np.log((1.0 - p) * (1.0 - nu) / max(vocab - 2, 1)))

    for iteration in range(1, max_iter + 1):
        # --- E step: posterior over the true label of each item -------------
        log_likelihood = 0.0
        for key, votes in items.items():
            cands = candidates[key]
            m = len(cands)
            if m == 1:
                posteriors[key] = {cands[0]: 1.0}
                continue
            vocab = max(vocabulary_size or field_vocab.get(key[1], m), m)
            log_probs = []
            for c in cands:
                total = np.log(1.0 / m)  # uniform prior over the observed candidates
                for w, value in votes.items():
                    total += log_emission(w, value, c, vocab)
                log_probs.append(total)
            arr = np.array(log_probs)
            shifted = arr - arr.max()
            weights = np.exp(shifted)
            norm = weights.sum()
            log_likelihood += float(arr.max() + np.log(norm))
            posteriors[key] = {c: float(w_) for c, w_ in zip(cands, weights / norm)}

        # --- M step ----------------------------------------------------------
        numer = {w: 0.0 for w in workers}
        denom = {w: 0.0 for w in workers}
        null_numer = {w: 0.0 for w in workers}
        null_denom = {w: 0.0 for w in workers}
        for key, votes in items.items():
            post = posteriors[key]
            # P(the true answer is a value rather than "absent")
            value_mass = sum(prob for c, prob in post.items() if c != NULL_LABEL)
            for w, value in votes.items():
                correct = post.get(value, 0.0)
                numer[w] += correct
                denom[w] += 1.0
                if value == NULL_LABEL:
                    null_numer[w] += value_mass
                    null_denom[w] += value_mass
                else:
                    null_denom[w] += max(value_mass - correct, 0.0)

        updated = {
            w: float(np.clip(numer[w] / denom[w] if denom[w] else 0.5, min_reliability, max_reliability))
            for w in workers
        }
        delta = max(abs(updated[w] - reliability[w]) for w in workers)
        reliability = updated
        if null_aware:
            for w in workers:
                if null_denom[w] > 1e-9:
                    null_rate[w] = float(np.clip(null_numer[w] / null_denom[w], 0.01, 0.99))
        if delta < tol:
            break

    labels = {key: max(sorted(post), key=lambda c: post[c]) for key, post in posteriors.items()}
    confidence = {key: post[labels[key]] for key, post in posteriors.items()}
    quality = dict(reliability)
    return Aggregation("dawid_skene_one_coin", labels, confidence, quality, iteration, log_likelihood)


def binary_dawid_skene(
    items: ItemLabels,
    max_iter: int = 200,
    tol: float = 1e-6,
    smoothing: float = 1.0,
) -> Aggregation:
    """Full Dawid-Skene over the two-class populated/null decision.

    Returns labels in {'populated', NULL_LABEL} and per-config 2x2 confusion
    matrices in `worker_quality` flattened as
    {'<config>|p(null->null)': ..., '<config>|p(pop->pop)': ...}.
    """
    workers = sorted({w for votes in items.values() for w in votes})
    keys = sorted(items)
    if not workers or not keys:
        return Aggregation("dawid_skene_binary", {}, {})

    observed = {
        key: {w: (0 if value == NULL_LABEL else 1) for w, value in items[key].items()} for key in keys
    }
    # Warm start from the observed vote share.
    posterior = np.zeros((len(keys), 2))
    for i, key in enumerate(keys):
        votes = list(observed[key].values())
        frac = float(np.mean(votes)) if votes else 0.5
        posterior[i] = [1 - frac, frac]

    confusion = {w: np.full((2, 2), 0.5) for w in workers}
    prior = np.array([0.5, 0.5])
    log_likelihood = float("nan")

    for iteration in range(1, max_iter + 1):
        # --- M step ---------------------------------------------------------
        prior = posterior.mean(axis=0)
        for w in workers:
            counts = np.full((2, 2), smoothing)
            for i, key in enumerate(keys):
                if w not in observed[key]:
                    continue
                obs = observed[key][w]
                counts[0, obs] += posterior[i, 0]
                counts[1, obs] += posterior[i, 1]
            confusion[w] = counts / counts.sum(axis=1, keepdims=True)

        # --- E step ---------------------------------------------------------
        new_posterior = np.zeros_like(posterior)
        log_likelihood = 0.0
        for i, key in enumerate(keys):
            log_p = np.log(np.clip(prior, 1e-12, None))
            for w, obs in observed[key].items():
                log_p = log_p + np.log(np.clip(confusion[w][:, obs], 1e-12, None))
            shifted = log_p - log_p.max()
            weights = np.exp(shifted)
            norm = weights.sum()
            log_likelihood += float(log_p.max() + np.log(norm))
            new_posterior[i] = weights / norm
        delta = float(np.abs(new_posterior - posterior).max())
        posterior = new_posterior
        if delta < tol:
            break

    labels, confidence = {}, {}
    for i, key in enumerate(keys):
        cls = int(np.argmax(posterior[i]))
        labels[key] = NULL_LABEL if cls == 0 else "populated"
        confidence[key] = float(posterior[i, cls])
    quality = {}
    for w in workers:
        quality[f"{w}|p(null->null)"] = float(confusion[w][0, 0])
        quality[f"{w}|p(pop->pop)"] = float(confusion[w][1, 1])
    return Aggregation("dawid_skene_binary", labels, confidence, quality, iteration, log_likelihood)


def score_aggregation(
    aggregation: Aggregation,
    gold: Dict[ItemKey, str],
    binary: bool = False,
) -> Dict[str, float]:
    keys = [k for k in aggregation.labels if k in gold]
    if not keys:
        return {"n": 0, "accuracy": float("nan")}

    def truth(key: ItemKey) -> str:
        if not binary:
            return gold[key]
        return NULL_LABEL if gold[key] == NULL_LABEL else "populated"

    correct = np.array([aggregation.labels[k] == truth(k) for k in keys], dtype=float)
    populated = np.array([gold[k] != NULL_LABEL for k in keys], dtype=bool)
    return {
        "n": int(len(keys)),
        "accuracy": float(correct.mean()),
        "acc_populated": float(correct[populated].mean()) if populated.any() else float("nan"),
        "acc_null": float(correct[~populated].mean()) if (~populated).any() else float("nan"),
    }


def compare_aggregators(
    preds: pd.DataFrame,
    judgments: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, Aggregation]]:
    """Single configs vs. majority vote vs. Dawid-Skene, on identical items."""
    items, gold = items_from_predictions(preds, judgments)
    if not items:
        return pd.DataFrame(), {}

    rows: List[Dict[str, object]] = []
    for config in sorted({w for votes in items.values() for w in votes}):
        single = Aggregation(
            method=f"single:{config}",
            labels={k: v[config] for k, v in items.items() if config in v},
            confidence={k: 1.0 for k in items},
        )
        rows.append({"method": single.method, **score_aggregation(single, gold)})

    aggregations: Dict[str, Aggregation] = {
        "majority_vote": majority_vote(items),
        "dawid_skene_one_coin": one_coin_dawid_skene(items),
    }
    for name, agg in aggregations.items():
        rows.append({"method": name, **score_aggregation(agg, gold)})

    binary_agg = binary_dawid_skene(items)
    aggregations["dawid_skene_binary"] = binary_agg
    binary_mv = Aggregation(
        "majority_vote_binary",
        {k: (NULL_LABEL if v == NULL_LABEL else "populated") for k, v in aggregations["majority_vote"].labels.items()},
        aggregations["majority_vote"].confidence,
    )
    rows.append({"method": "majority_vote (null/populated only)", **score_aggregation(binary_mv, gold, binary=True)})
    rows.append({"method": "dawid_skene_binary (null/populated only)", **score_aggregation(binary_agg, gold, binary=True)})

    # Ceiling: how often *any* config had the right answer. The gap between
    # this and the best aggregator is what better aggregation could still buy.
    oracle = Aggregation(
        "oracle_any_config",
        {k: (gold[k] if any(v == gold[k] for v in votes.values()) else "__none__") for k, votes in items.items()},
        {k: 1.0 for k in items},
    )
    rows.append({"method": "oracle: any config correct", **score_aggregation(oracle, gold)})

    frame = pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
    return frame, aggregations
