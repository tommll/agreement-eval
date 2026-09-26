"""Does agreement predict correctness?

The study's central question, answered three ways:

1. *Discrimination* - AUC of an agreement score for predicting whether the
   consensus answer is correct.
2. *Calibration* - P(correct | agreement bucket), which is what a team is
   implicitly assuming when it quotes an agreement number.
3. *Operating point* - if you auto-accept unanimous items and review the rest,
   what coverage do you get and what error rate slips through?

Every one of them is reported split by gold-populated vs. gold-null, because
a field that is absent from the document (and that every config correctly
calls null) inflates all three at once.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .agreement import NULL_LABEL

DEFAULT_BINS = (-0.001, 0.25, 0.5, 0.75, 0.999, 1.0)
BIN_LABELS = ("0-25%", "25-50%", "50-75%", "75-99%", "unanimous")


def roc_auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Rank-based AUC with tie correction. NaN when one class is missing."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    keep = ~np.isnan(scores)
    scores, labels = scores[keep], labels[keep]
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0  # average rank for ties
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Rank correlation without pulling in scipy."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    if len(x) < 2:
        return float("nan")

    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        out = np.empty(len(values), dtype=float)
        i = 0
        while i < len(values):
            j = i
            while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
                j += 1
            out[order[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return out

    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def wilson_ci(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval - behaves at p near 0 or 1, where these rates live."""
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return (float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2)))


def _rate_row(mask: pd.Series, outcome: pd.Series, label: str) -> Dict[str, object]:
    subset = outcome[mask]
    n = int(len(subset))
    k = int(subset.sum())
    low, high = wilson_ci(k, n)
    return {"group": label, "n": n, "n_correct": k,
            "p_correct": (k / n) if n else float("nan"), "ci_low": low, "ci_high": high}


def bucketize(scores: pd.Series) -> pd.Series:
    """Bucket an agreement score for tabulation.

    With a handful of raters the score only takes a handful of values (with 4
    raters: 0, 1/6, 1/3, 1/2, 2/3, 1), so fixed-width bins produce empty rows
    and merge meaningfully different patterns. Use the distinct values as
    buckets while there are few of them, and fall back to bins otherwise.
    """
    distinct = sorted(scores.dropna().unique())
    if 0 < len(distinct) <= 8:
        labels = [f"{v:.0%}" for v in distinct]
        mapping = dict(zip(distinct, labels))
        return pd.Categorical(scores.map(mapping), categories=labels, ordered=True)
    return pd.cut(scores, bins=list(DEFAULT_BINS), labels=list(BIN_LABELS), include_lowest=True)


def add_derived(items: pd.DataFrame, score_col: str = "pairwise_agreement") -> pd.DataFrame:
    """Add the conditioning columns the analysis splits on."""
    df = items.copy()
    df["consensus_populated"] = df["consensus_label"] != NULL_LABEL
    if score_col in df.columns:
        df["agreement_bucket"] = bucketize(df[score_col])
    return df


def bucket_table(
    items: pd.DataFrame,
    score_col: str = "pairwise_agreement",
    outcome_col: str = "consensus_correct",
    split_col: Optional[str] = "gold_populated",
) -> pd.DataFrame:
    """P(correct | agreement bucket), optionally split by a boolean column."""
    df = add_derived(items, score_col)
    frames: List[pd.DataFrame] = []
    splits: List[Tuple[str, pd.Series]] = [("all", pd.Series(True, index=df.index))]
    if split_col:
        splits += [
            (f"{split_col}=True", df[split_col].astype(bool)),
            (f"{split_col}=False", ~df[split_col].astype(bool)),
        ]
    for split_name, mask in splits:
        sub = df[mask]
        if sub.empty:
            continue
        grouped = sub.groupby("agreement_bucket", observed=False)[outcome_col]
        table = grouped.agg(n="size", n_correct="sum").reset_index()
        table["p_correct"] = table["n_correct"] / table["n"].replace(0, np.nan)
        cis = [wilson_ci(int(k), int(n)) for k, n in zip(table["n_correct"], table["n"])]
        table["ci_low"] = [c[0] for c in cis]
        table["ci_high"] = [c[1] for c in cis]
        table.insert(0, "split", split_name)
        table.insert(0, "score", score_col)
        frames.append(table)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def trust_rule(
    items: pd.DataFrame,
    rule_col: str = "unanimous",
    outcome_col: str = "consensus_correct",
    split_col: Optional[str] = "gold_populated",
) -> pd.DataFrame:
    """The operating-point view: auto-accept when the rule fires, review the rest.

    Reports coverage (share auto-accepted), precision inside and outside the
    rule, the base rate, and lift - plus errors per 1,000 auto-accepted items,
    which is the number an operations team actually budgets against.
    """
    df = add_derived(items)
    rows: List[Dict[str, object]] = []
    splits: List[Tuple[str, pd.Series]] = [("all", pd.Series(True, index=df.index))]
    if split_col:
        splits += [
            (f"{split_col}=True", df[split_col].astype(bool)),
            (f"{split_col}=False", ~df[split_col].astype(bool)),
        ]
    for split_name, mask in splits:
        sub = df[mask]
        if sub.empty:
            continue
        fires = sub[rule_col].astype(bool)
        outcome = sub[outcome_col].astype(bool)
        n = len(sub)
        n_fire = int(fires.sum())
        p_fire = float(outcome[fires].mean()) if n_fire else float("nan")
        p_not = float(outcome[~fires].mean()) if n - n_fire else float("nan")
        base = float(outcome.mean())
        low, high = wilson_ci(int(outcome[fires].sum()), n_fire)
        rows.append({
            "split": split_name,
            "rule": rule_col,
            "n_items": n,
            "coverage": n_fire / n if n else float("nan"),
            "p_correct_when_rule_fires": p_fire,
            "ci_low": low,
            "ci_high": high,
            "p_correct_when_rule_does_not_fire": p_not,
            "base_rate": base,
            "lift_over_base": (p_fire / base) if base else float("nan"),
            "errors_per_1000_accepted": (1 - p_fire) * 1000 if n_fire else float("nan"),
            "errors_missed": int(((~outcome) & fires).sum()),
            "errors_caught": int(((~outcome) & ~fires).sum()),
        })
    return pd.DataFrame(rows)


def calibration(
    items: pd.DataFrame,
    score_col: str = "pairwise_agreement",
    outcome_col: str = "consensus_correct",
    n_bins: int = 5,
) -> Tuple[pd.DataFrame, float]:
    """Reliability table plus expected calibration error.

    Read literally, "the configs agree 75% of the time" is a claim that such
    items are correct 75% of the time. ECE says how wrong that reading is.
    """
    df = items.dropna(subset=[score_col]).copy()
    if df.empty:
        return pd.DataFrame(), float("nan")
    edges = np.linspace(0, 1, n_bins + 1)
    df["bin"] = np.clip(np.digitize(df[score_col], edges[1:-1], right=True), 0, n_bins - 1)
    rows, ece, total = [], 0.0, len(df)
    for b, group in df.groupby("bin"):
        mean_score = float(group[score_col].mean())
        observed = float(group[outcome_col].mean())
        rows.append({
            "bin": f"[{edges[b]:.2f}, {edges[b + 1]:.2f}]",
            "n": int(len(group)),
            "mean_agreement": mean_score,
            "observed_accuracy": observed,
            "gap": observed - mean_score,
        })
        ece += len(group) / total * abs(observed - mean_score)
    return pd.DataFrame(rows), float(ece)


def discrimination(
    items: pd.DataFrame,
    score_cols: Sequence[str] = ("pairwise_agreement", "consensus_frac", "entropy", "jaccard_agreement"),
    outcome_col: str = "consensus_correct",
    split_col: Optional[str] = "gold_populated",
) -> pd.DataFrame:
    """AUC per agreement score, overall and inside each split."""
    df = add_derived(items)
    rows: List[Dict[str, object]] = []
    splits: List[Tuple[str, pd.Series]] = [("all", pd.Series(True, index=df.index))]
    if split_col:
        splits += [
            (f"{split_col}=True", df[split_col].astype(bool)),
            (f"{split_col}=False", ~df[split_col].astype(bool)),
        ]
    for score in score_cols:
        if score not in df.columns:
            continue
        for split_name, mask in splits:
            sub = df[mask]
            if sub.empty:
                continue
            outcome = sub[outcome_col].astype(bool)
            auc = roc_auc(sub[score], outcome)
            # Entropy points the other way; report it on the same scale.
            if score == "entropy" and not np.isnan(auc):
                auc = 1 - auc
            rows.append({
                "score": score if score != "entropy" else "entropy (inverted)",
                "split": split_name,
                "n": int(len(sub)),
                "base_rate": float(outcome.mean()),
                "auc": auc,
            })
    return pd.DataFrame(rows)


def per_field_breakdown(
    items: pd.DataFrame,
    rule_col: str = "unanimous",
    outcome_col: str = "consensus_correct",
) -> pd.DataFrame:
    """The same trust-rule numbers, per field - where the averages hide things."""
    rows: List[Dict[str, object]] = []
    for field_name, group in items.groupby("field"):
        fires = group[rule_col].astype(bool)
        outcome = group[outcome_col].astype(bool)
        n_fire = int(fires.sum())
        rows.append({
            "field": field_name,
            "n_items": int(len(group)),
            "gold_populated_rate": float(group["gold_populated"].mean()),
            "unanimity_rate": float(fires.mean()),
            "p_correct_when_unanimous": float(outcome[fires].mean()) if n_fire else float("nan"),
            "p_correct_when_split": float(outcome[~fires].mean()) if len(group) - n_fire else float("nan"),
            "base_rate": float(outcome.mean()),
            "auc": roc_auc(group["pairwise_agreement"], outcome),
        })
    return pd.DataFrame(rows).sort_values("field").reset_index(drop=True)


def prediction_level(
    preds: pd.DataFrame,
    support: pd.DataFrame,
    outcome_col: str = "correct",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-prediction view: does 'the others back me' predict *my* correctness?

    Returns (per-config AUC table, support-bucket table).
    """
    merged = preds.merge(support, on=["doc_id", "field", "config_id"], how="inner")
    rows = []
    for config_id, group in merged.groupby("config_id"):
        outcome = group[outcome_col].astype(bool)
        rows.append({
            "config_id": config_id,
            "n": int(len(group)),
            "accuracy": float(outcome.mean()),
            "auc_support_frac": roc_auc(group["support_frac"], outcome),
            "p_correct_when_all_others_agree": float(outcome[group["support_frac"] >= 0.999].mean())
            if (group["support_frac"] >= 0.999).any() else float("nan"),
            "p_correct_when_none_agree": float(outcome[group["support_frac"] <= 0.001].mean())
            if (group["support_frac"] <= 0.001).any() else float("nan"),
        })
    per_config = pd.DataFrame(rows).sort_values("config_id").reset_index(drop=True)

    merged["support_bucket"] = bucketize(merged["support_frac"])
    buckets = (
        merged.groupby(["support_bucket", "gold_populated"], observed=False)[outcome_col]
        .agg(n="size", n_correct="sum").reset_index()
    )
    buckets["p_correct"] = buckets["n_correct"] / buckets["n"].replace(0, np.nan)
    return per_config, buckets


def headline(items: pd.DataFrame, outcome_col: str = "consensus_correct") -> Dict[str, object]:
    """The few numbers you would actually put in front of a team."""
    df = add_derived(items)
    outcome = df[outcome_col].astype(bool)
    unanimous = df["unanimous"].astype(bool)
    populated = df["gold_populated"].astype(bool)

    def safe(mask: pd.Series) -> float:
        return float(outcome[mask].mean()) if mask.any() else float("nan")

    return {
        "n_items": int(len(df)),
        "unanimity_rate": float(unanimous.mean()),
        "consensus_accuracy": float(outcome.mean()),
        "p_correct_given_unanimous": safe(unanimous),
        "p_correct_given_split": safe(~unanimous),
        "auc_pairwise_agreement": roc_auc(df["pairwise_agreement"], outcome),
        "populated": {
            "n": int(populated.sum()),
            "unanimity_rate": float(unanimous[populated].mean()) if populated.any() else float("nan"),
            "p_correct_given_unanimous": safe(unanimous & populated),
            "p_correct_given_split": safe(~unanimous & populated),
            "auc": roc_auc(df.loc[populated, "pairwise_agreement"], outcome[populated]),
        },
        "null": {
            "n": int((~populated).sum()),
            "unanimity_rate": float(unanimous[~populated].mean()) if (~populated).any() else float("nan"),
            "p_correct_given_unanimous": safe(unanimous & ~populated),
            "p_correct_given_split": safe(~unanimous & ~populated),
            "auc": roc_auc(df.loc[~populated, "pairwise_agreement"], outcome[~populated]),
        },
    }
