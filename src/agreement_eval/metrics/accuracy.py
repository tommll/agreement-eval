"""Per-field accuracy against the labels.

Grain: one row per (document, field, config) - a *prediction*. Everything
else in the analysis is built from this frame.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from ..normalize import DEFAULT_JACCARD_THRESHOLD, jaccard_similarity, normalize
from ..schemas import DatasetSchema

PREDICTION_KEYS = ["doc_id", "field", "config_id"]
ITEM_KEYS = ["doc_id", "field"]


def _none(value) -> Optional[str]:
    """Coerce pandas' NaN/NaT/'' to None so comparisons stay 3-valued-free."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def build_predictions(
    raw: pd.DataFrame,
    schema: DatasetSchema,
    jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD,
) -> pd.DataFrame:
    """Attach normalized gold and correctness to the raw prediction frame.

    Two correctness columns: `correct` is exact match after normalization (what
    agreement, majority vote and Dawid-Skene are scored on); `correct_jaccard`
    gives partial credit on free-text fields when token-set Jaccard to the
    label is at least `jaccard_threshold`.
    """
    if raw.empty:
        return raw.assign(gold_norm=None, correct=False, gold_populated=False, pred_populated=False,
                          jaccard_gold=0.0, correct_jaccard=False)

    df = raw.copy()
    df["value_norm"] = df["value_norm"].map(_none)
    df["gold_raw"] = df["gold_raw"].map(_none)
    df["field_type"] = df["field"].map(lambda f: schema.field_type(f))
    df["gold_norm"] = [normalize(g, t) for g, t in zip(df["gold_raw"], df["field_type"])]

    df["gold_populated"] = df["gold_norm"].notna()
    df["pred_populated"] = df["value_norm"].notna()
    # Exact match after normalization; both-absent counts as correct, because
    # "this field is not on the document" is a real answer the model can get
    # right or wrong.
    df["correct"] = df["value_norm"].fillna("\x00NULL") == df["gold_norm"].fillna("\x00NULL")
    df["jaccard_gold"] = [
        jaccard_similarity(v, g, t) for v, g, t in zip(df["value_norm"], df["gold_norm"], df["field_type"])
    ]
    # Exact matches score 1.0, so this can only ever upgrade `correct`.
    df["correct_jaccard"] = df["jaccard_gold"] >= jaccard_threshold

    # Error taxonomy - the three mistakes have different operational costs.
    df["error_kind"] = np.select(
        [
            df["correct"],
            ~df["gold_populated"] & df["pred_populated"],
            df["gold_populated"] & ~df["pred_populated"],
        ],
        ["none", "hallucination", "miss"],
        default="wrong_value",
    )
    return df


def per_config_field(preds: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per (config, field), with the error taxonomy broken out."""
    grouped = preds.groupby(["config_id", "field"], dropna=False)
    out = grouped.agg(
        n=("correct", "size"),
        accuracy=("correct", "mean"),
        accuracy_jaccard=("correct_jaccard", "mean"),
        mean_jaccard=("jaccard_gold", "mean"),
        n_gold_populated=("gold_populated", "sum"),
        n_pred_populated=("pred_populated", "sum"),
    ).reset_index()

    for kind in ("hallucination", "miss", "wrong_value"):
        rate = (
            preds.assign(flag=preds["error_kind"] == kind)
            .groupby(["config_id", "field"], dropna=False)["flag"].mean()
            .rename(f"{kind}_rate")
        )
        out = out.merge(rate, on=["config_id", "field"], how="left")

    # Accuracy conditioned on the gold state - the headline split of the study.
    for name, mask in (("acc_populated", preds["gold_populated"]), ("acc_null", ~preds["gold_populated"])):
        sub = preds[mask].groupby(["config_id", "field"], dropna=False)["correct"].mean().rename(name)
        out = out.merge(sub, on=["config_id", "field"], how="left")
    return out.sort_values(["field", "config_id"]).reset_index(drop=True)


def per_config(preds: pd.DataFrame) -> pd.DataFrame:
    out = preds.groupby("config_id", dropna=False).agg(
        n=("correct", "size"),
        accuracy=("correct", "mean"),
        accuracy_jaccard=("correct_jaccard", "mean"),
        mean_jaccard=("jaccard_gold", "mean"),
    ).reset_index()
    for name, mask in (("acc_populated", preds["gold_populated"]), ("acc_null", ~preds["gold_populated"])):
        sub = preds[mask].groupby("config_id", dropna=False)["correct"].mean().rename(name)
        out = out.merge(sub, on="config_id", how="left")
    for kind in ("hallucination", "miss", "wrong_value"):
        sub = (preds["error_kind"] == kind).groupby(preds["config_id"]).mean().rename(f"{kind}_rate")
        out = out.merge(sub, on="config_id", how="left")
    return out.sort_values("accuracy", ascending=False).reset_index(drop=True)


def per_field(preds: pd.DataFrame) -> pd.DataFrame:
    out = preds.groupby("field", dropna=False).agg(
        n=("correct", "size"),
        accuracy=("correct", "mean"),
        accuracy_jaccard=("correct_jaccard", "mean"),
        mean_jaccard=("jaccard_gold", "mean"),
        gold_populated_rate=("gold_populated", "mean"),
    ).reset_index()
    for name, mask in (("acc_populated", preds["gold_populated"]), ("acc_null", ~preds["gold_populated"])):
        sub = preds[mask].groupby("field", dropna=False)["correct"].mean().rename(name)
        out = out.merge(sub, on="field", how="left")
    return out.sort_values("field").reset_index(drop=True)


def apply_judge_correctness(preds: pd.DataFrame, gold_judgments: pd.DataFrame) -> pd.DataFrame:
    """Add `correct_judge`: exact match OR the judge calling it equivalent.

    The judge only ever *upgrades* an exact-match miss (formatting, synonymy);
    it is never allowed to overturn an exact match, so this stays an upper
    bound on lenient accuracy.
    """
    df = preds.copy()
    df["correct_judge"] = df["correct"]
    if gold_judgments is None or gold_judgments.empty:
        return df
    verdicts = (
        gold_judgments.rename(columns={"left_config": "config_id"})
        .set_index(["doc_id", "field", "config_id"])["equivalent"].to_dict()
    )
    keys = list(zip(df["doc_id"], df["field"], df["config_id"]))
    df["correct_judge"] = [
        bool(exact) or bool(verdicts.get(key, False)) for exact, key in zip(df["correct"], keys)
    ]
    return df
