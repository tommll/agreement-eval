"""Agreement between configs, at item (document x field) grain.

Two notions of agreement are computed from the same code path:

    exact     - equality of normalized values (see normalize.py)
    semantic  - exact, then values an LLM judge called equivalent are merged
                into one cluster before anything is counted

plus a graded one alongside them:

    jaccard   - mean pairwise token-set Jaccard of the normalized values. It
                is a score, not a clustering: a Jaccard threshold is not
                transitive, so unanimity, majority and Dawid-Skene stay exact.

Merging judged-equivalent values into clusters (rather than patching pairwise
counts) keeps every downstream statistic - unanimity, majority, entropy,
Dawid-Skene input - consistent with the judge's verdicts.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..normalize import DEFAULT_JACCARD_THRESHOLD, jaccard_similarity

NULL_LABEL = "∅"  # the empty set: "this field is absent"


def label_of(value: Optional[str]) -> str:
    return NULL_LABEL if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)


def _value_of(label: str) -> Optional[str]:
    return None if label == NULL_LABEL else label


def label_jaccard(a: str, b: str, field_type: str) -> float:
    """Jaccard similarity between two labels (NULL_LABEL means absent)."""
    return jaccard_similarity(_value_of(a), _value_of(b), field_type)


def mean_pairwise_jaccard(labels: Sequence[str], field_type: str) -> float:
    pairs = list(itertools.combinations(labels, 2))
    if not pairs:
        return float("nan")
    return sum(label_jaccard(a, b, field_type) for a, b in pairs) / len(pairs)


class _DSU:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Canonical representative is the lexicographically smaller label,
            # so cluster ids are stable across runs.
            lo, hi = sorted((ra, rb))
            self.parent[hi] = lo


def cluster_labels(labels: Dict[str, str], equivalences: Sequence[Tuple[str, str]] = ()) -> Dict[str, str]:
    """Map config -> cluster label, merging judged-equivalent values.

    NULL is never merged with a populated value: "absent" and "present" is the
    distinction the whole populated/null analysis rests on.
    """
    dsu = _DSU()
    for value in labels.values():
        dsu.find(value)
    for a, b in equivalences:
        if NULL_LABEL in (a, b) and a != b:
            continue
        dsu.union(a, b)
    return {config: dsu.find(value) for config, value in labels.items()}


def _entropy(counts: Sequence[int]) -> float:
    total = sum(counts)
    if total <= 1 or len(counts) <= 1:
        return 0.0
    probs = [c / total for c in counts if c]
    raw = -sum(p * math.log(p) for p in probs)
    return raw / math.log(len(counts)) if len(counts) > 1 else 0.0


def item_stats(labels: Dict[str, str]) -> Dict[str, object]:
    """Agreement statistics for one item, from config -> cluster label."""
    values = list(labels.values())
    n = len(values)
    counts: Dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1

    pairs = list(itertools.combinations(values, 2))
    n_pairs = len(pairs)
    n_agree = sum(1 for a, b in pairs if a == b)
    populated = [v for v in values if v != NULL_LABEL]
    pop_pairs = list(itertools.combinations(populated, 2))

    # Deterministic majority: highest count, ties broken lexicographically.
    consensus = min(sorted(counts), key=lambda v: (-counts[v], v)) if counts else NULL_LABEL
    support = counts.get(consensus, 0)

    return {
        "n_raters": n,
        "n_pairs": n_pairs,
        "n_agreeing_pairs": n_agree,
        "pairwise_agreement": (n_agree / n_pairs) if n_pairs else float("nan"),
        "pairwise_agreement_populated": (
            sum(1 for a, b in pop_pairs if a == b) / len(pop_pairs) if pop_pairs else float("nan")
        ),
        # One surviving rater is not a consensus. When other configs errored on
        # a document, the item has nothing to agree about and must not be
        # counted as unanimous - that would credit agreement for silence.
        "unanimous": n > 1 and len(counts) == 1,
        "n_distinct": len(counts),
        "consensus_label": consensus,
        "consensus_support": support,
        "consensus_frac": support / n if n else float("nan"),
        "entropy": _entropy(list(counts.values())),
        "n_null": sum(1 for v in values if v == NULL_LABEL),
        "null_frac": sum(1 for v in values if v == NULL_LABEL) / n if n else float("nan"),
        "all_null": all(v == NULL_LABEL for v in values),
        "any_null": any(v == NULL_LABEL for v in values),
    }


def judge_equivalences(judgments: Optional[pd.DataFrame]) -> Dict[Tuple[str, str], List[Tuple[str, str]]]:
    """(doc_id, field) -> list of equivalent value pairs, from judge verdicts."""
    if judgments is None or judgments.empty:
        return {}
    out: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    for row in judgments[judgments["equivalent"] == True].itertuples():  # noqa: E712
        key = (row.doc_id, row.field)
        out.setdefault(key, []).append((label_of(row.left_value), label_of(row.right_value)))
    return out


def item_agreement(
    preds: pd.DataFrame,
    judgments: Optional[pd.DataFrame] = None,
    value_col: str = "value_norm",
    jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD,
) -> pd.DataFrame:
    """One row per (doc_id, field) with agreement statistics and the gold answer.

    When `judgments` is given, the same statistics are recomputed on judge-merged
    clusters and returned with a `sem_` prefix. `jaccard_agreement` is the graded
    agreement score; `consensus_correct_jaccard` scores the (exact) consensus
    against the label with Jaccard partial credit.
    """
    if preds.empty:
        return pd.DataFrame()

    equivalences = judge_equivalences(judgments)
    rows: List[Dict[str, object]] = []
    for (doc_id, field), group in preds.groupby(["doc_id", "field"], sort=True):
        labels = {row.config_id: label_of(getattr(row, value_col)) for row in group.itertuples()}
        gold_label = label_of(group["gold_norm"].iloc[0])
        stats = item_stats(labels)

        row: Dict[str, object] = {
            "doc_id": doc_id,
            "field": field,
            "field_type": group["field_type"].iloc[0],
            "gold_label": gold_label,
            "gold_populated": gold_label != NULL_LABEL,
            **stats,
        }
        row["consensus_correct"] = row["consensus_label"] == gold_label
        field_type = row["field_type"]
        row["jaccard_agreement"] = mean_pairwise_jaccard(list(labels.values()), field_type)
        row["consensus_jaccard_gold"] = label_jaccard(row["consensus_label"], gold_label, field_type)
        row["consensus_correct_jaccard"] = row["consensus_jaccard_gold"] >= jaccard_threshold
        row["any_correct"] = any(v == gold_label for v in labels.values())
        row["n_correct"] = sum(1 for v in labels.values() if v == gold_label)

        if equivalences:
            merged = cluster_labels(labels, equivalences.get((doc_id, field), []))
            # Gold goes through the same merge so a judged-equivalent consensus
            # is scored as correct.
            gold_cluster = cluster_labels({**labels, "__gold__": gold_label},
                                          equivalences.get((doc_id, field), []))["__gold__"]
            sem = item_stats(merged)
            row.update({f"sem_{k}": v for k, v in sem.items()})
            row["sem_consensus_correct"] = sem["consensus_label"] == gold_cluster
        rows.append(row)

    return pd.DataFrame(rows)


def pairwise_table(preds: pd.DataFrame, value_col: str = "value_norm") -> pd.DataFrame:
    """One row per (doc, field, config_a, config_b) - the judge's work queue."""
    rows: List[Dict[str, object]] = []
    for (doc_id, field), group in preds.groupby(["doc_id", "field"], sort=True):
        ordered = group.sort_values("config_id")
        records = [(r.config_id, label_of(getattr(r, value_col)), r.value_raw) for r in ordered.itertuples()]
        for (ca, la, ra), (cb, lb, rb) in itertools.combinations(records, 2):
            rows.append({
                "doc_id": doc_id,
                "field": field,
                "field_type": group["field_type"].iloc[0],
                "config_a": ca,
                "config_b": cb,
                "label_a": la,
                "label_b": lb,
                "value_a": ra,
                "value_b": rb,
                "exact_agree": la == lb,
                "jaccard": label_jaccard(la, lb, group["field_type"].iloc[0]),
                "both_null": la == NULL_LABEL and lb == NULL_LABEL,
                "one_null": (la == NULL_LABEL) != (lb == NULL_LABEL),
            })
    return pd.DataFrame(rows)


def pairwise_matrix(pairs: pd.DataFrame, column: str = "exact_agree") -> pd.DataFrame:
    """Config x config agreement rates (the metric a team usually quotes)."""
    if pairs.empty:
        return pd.DataFrame()
    configs = sorted(set(pairs["config_a"]) | set(pairs["config_b"]))
    matrix = pd.DataFrame(np.nan, index=configs, columns=configs, dtype=float)
    for (a, b), group in pairs.groupby(["config_a", "config_b"]):
        rate = float(group[column].mean())
        matrix.loc[a, b] = rate
        matrix.loc[b, a] = rate
    for c in configs:
        matrix.loc[c, c] = 1.0
    return matrix


def per_prediction_agreement(preds: pd.DataFrame, judgments: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Per prediction: how many *other* configs back this exact answer.

    This is the "can I trust this particular output?" view, as opposed to the
    item-level "can I trust the consensus?" view.
    """
    equivalences = judge_equivalences(judgments)
    rows: List[Dict[str, object]] = []
    for (doc_id, field), group in preds.groupby(["doc_id", "field"], sort=True):
        labels = {r.config_id: label_of(r.value_norm) for r in group.itertuples()}
        if equivalences:
            labels = cluster_labels(labels, equivalences.get((doc_id, field), []))
        n = len(labels)
        for config_id, label in labels.items():
            others = [v for c, v in labels.items() if c != config_id]
            rows.append({
                "doc_id": doc_id,
                "field": field,
                "config_id": config_id,
                "support_frac": (sum(1 for v in others if v == label) / len(others)) if others else float("nan"),
                "n_raters": n,
            })
    return pd.DataFrame(rows)
