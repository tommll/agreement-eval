"""Runs the whole evaluation over a stored run and materializes every table."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from . import db
from .metrics import accuracy, agreement, dawid_skene, predictive
from .normalize import DEFAULT_JACCARD_THRESHOLD
from .schemas import get_schema


@dataclass
class AnalysisResult:
    run_id: str
    schema_name: str
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    headline: Dict[str, Any] = field(default_factory=dict)
    aggregations: Dict[str, dawid_skene.Aggregation] = field(default_factory=dict)

    def write_csvs(self, outdir: Path) -> Path:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        for name, frame in self.tables.items():
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                frame.to_csv(outdir / f"{name}.csv", index=name.endswith("_matrix"))
        return outdir


def analyze(
    conn,
    run_id: str,
    schema_name: Optional[str] = None,
    jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD,
) -> AnalysisResult:
    info = db.run_info(conn, run_id)
    if info is None and schema_name is None:
        raise KeyError(f"run {run_id!r} not found and no schema given")
    schema_name = schema_name or info["schema_name"]
    schema = get_schema(schema_name)

    raw = db.load_predictions(conn, run_id)
    if raw.empty:
        raise ValueError(f"run {run_id!r} has no stored field outputs - run `extract` first")

    preds = accuracy.build_predictions(raw, schema, jaccard_threshold=jaccard_threshold)
    judgments = db.load_judgments(conn, run_id)
    pair_judgments = judgments[judgments["kind"] == "pair"] if not judgments.empty else pd.DataFrame()
    gold_judgments = judgments[judgments["kind"] == "gold"] if not judgments.empty else pd.DataFrame()
    preds = accuracy.apply_judge_correctness(preds, gold_judgments)

    # Item-level agreement, exact and (if judged) judge-merged.
    items = agreement.item_agreement(preds, pair_judgments if not pair_judgments.empty else None,
                                     jaccard_threshold=jaccard_threshold)
    pairs = agreement.pairwise_table(preds)
    support = agreement.per_prediction_agreement(preds)

    tables: Dict[str, pd.DataFrame] = {
        "predictions": preds,
        "items": items,
        "pairs": pairs,
        "accuracy_per_config": accuracy.per_config(preds),
        "accuracy_per_config_field": accuracy.per_config_field(preds),
        "accuracy_per_field": accuracy.per_field(preds),
        "agreement_matrix": agreement.pairwise_matrix(pairs),
        "jaccard_matrix": agreement.pairwise_matrix(pairs, column="jaccard"),
        "extraction_stats": db.load_extraction_stats(conn, run_id),
        "local_speed": db.load_local_timings(conn, run_id),
    }

    # --- the central question ------------------------------------------------
    # Agreement is undefined for items only one config answered (the others
    # errored), so they are excluded here and counted in the headline instead.
    comparable = items[items["n_raters"] >= 2]
    n_single = int(len(items) - len(comparable))

    tables["agreement_buckets"] = predictive.bucket_table(comparable)
    tables["trust_rule_unanimous"] = predictive.trust_rule(comparable, rule_col="unanimous")
    tables["discrimination"] = predictive.discrimination(comparable)
    # The same question with partial credit: does graded agreement predict a
    # consensus that is Jaccard-close to the label?
    tables["discrimination_jaccard"] = predictive.discrimination(
        comparable, score_cols=("jaccard_agreement", "pairwise_agreement"),
        outcome_col="consensus_correct_jaccard",
    )
    tables["per_field_trust"] = predictive.per_field_breakdown(comparable)

    calibration_table, ece = predictive.calibration(comparable)
    tables["calibration"] = calibration_table

    # The split a production system can actually condition on: it does not know
    # the gold state, only what the consensus said.
    tables["trust_rule_by_consensus_state"] = predictive.trust_rule(
        predictive.add_derived(comparable), rule_col="unanimous", split_col="consensus_populated"
    )
    per_config_support, support_buckets = predictive.prediction_level(preds, support)
    tables["prediction_level_support"] = per_config_support
    tables["support_buckets"] = support_buckets

    # --- aggregation: majority vote vs Dawid-Skene ---------------------------
    comparison, aggregations = dawid_skene.compare_aggregators(
        preds, pair_judgments if not pair_judgments.empty else None
    )
    tables["aggregator_comparison"] = comparison

    headline_rank_corr = None
    ds = aggregations.get("dawid_skene_one_coin")
    if ds is not None:
        true_acc = tables["accuracy_per_config"].set_index("config_id")["accuracy"].to_dict()
        quality = pd.DataFrame(
            [{"config_id": c, "ds_estimated_reliability": q, "true_accuracy": true_acc.get(c)}
             for c, q in sorted(ds.worker_quality.items())]
        )
        tables["ds_worker_quality"] = quality
        # Dawid-Skene estimates reliability with no labels at all; this says
        # whether that estimate actually tracks measured accuracy.
        headline_rank_corr = predictive.spearman(
            quality["ds_estimated_reliability"], quality["true_accuracy"]
        )
        # Does Dawid-Skene's posterior beat raw agreement as a trust signal?
        ds_conf = pd.DataFrame(
            [{"doc_id": k[0], "field": k[1], "ds_confidence": v} for k, v in ds.confidence.items()]
        )
        items_ds = comparable.merge(ds_conf, on=["doc_id", "field"], how="left")
        items_ds["ds_correct"] = [
            ds.labels.get((d, f)) == g
            for d, f, g in zip(items_ds["doc_id"], items_ds["field"], items_ds["gold_label"])
        ]
        tables["ds_signal"] = predictive.discrimination(
            items_ds, score_cols=("pairwise_agreement", "ds_confidence"), outcome_col="ds_correct"
        )

    binary = aggregations.get("dawid_skene_binary")
    if binary is not None:
        tables["ds_binary_quality"] = pd.DataFrame(
            [{"parameter": k, "value": v} for k, v in sorted(binary.worker_quality.items())]
        )

    # --- judge effect ---------------------------------------------------------
    headline = predictive.headline(comparable)
    headline["n_items_total"] = int(len(items))
    headline["n_items_single_rater_excluded"] = n_single
    headline["ece_agreement_vs_accuracy"] = ece
    if headline_rank_corr is not None:
        headline["ds_reliability_rank_corr_with_accuracy"] = headline_rank_corr
    headline["jaccard"] = {
        "threshold": jaccard_threshold,
        "exact_accuracy": float(preds["correct"].mean()),
        "jaccard_accuracy": float(preds["correct_jaccard"].mean()),
        "mean_jaccard_to_gold": float(preds["jaccard_gold"].mean()),
        "consensus_accuracy_jaccard": float(comparable["consensus_correct_jaccard"].mean())
        if len(comparable) else float("nan"),
        "auc_jaccard_agreement": predictive.roc_auc(
            comparable["jaccard_agreement"], comparable["consensus_correct"].astype(bool)
        ) if len(comparable) else float("nan"),
    }
    headline["exact_vs_judge"] = {
        "exact_accuracy": float(preds["correct"].mean()),
        "judge_lenient_accuracy": float(preds["correct_judge"].mean()),
        "n_gold_judgments": int(len(gold_judgments)),
        "n_pair_judgments": int(len(pair_judgments)),
    }
    if "sem_unanimous" in comparable.columns:
        semantic = predictive.headline(
            comparable.rename(columns={
                "unanimous": "exact_unanimous", "sem_unanimous": "unanimous",
                "pairwise_agreement": "exact_pairwise_agreement",
                "sem_pairwise_agreement": "pairwise_agreement",
                "consensus_correct": "exact_consensus_correct",
                "sem_consensus_correct": "consensus_correct",
                "consensus_label": "exact_consensus_label",
                "sem_consensus_label": "consensus_label",
            })
        )
        headline["semantic_agreement"] = {
            "unanimity_rate": semantic["unanimity_rate"],
            "p_correct_given_unanimous": semantic["p_correct_given_unanimous"],
            "auc": semantic["auc_pairwise_agreement"],
            "populated_p_correct_given_unanimous": semantic["populated"]["p_correct_given_unanimous"],
        }
    if not comparison.empty:
        best_single = comparison[comparison["method"].str.startswith("single:")]["accuracy"].max()
        def _acc(name: str) -> float:
            row = comparison[comparison["method"] == name]
            return float(row["accuracy"].iloc[0]) if len(row) else float("nan")
        headline["aggregation"] = {
            "best_single_config": float(best_single) if pd.notna(best_single) else float("nan"),
            "majority_vote": _acc("majority_vote"),
            "dawid_skene_one_coin": _acc("dawid_skene_one_coin"),
            "ds_minus_mv": _acc("dawid_skene_one_coin") - _acc("majority_vote"),
            "oracle_any_config": _acc("oracle: any config correct"),
        }

    # If the judge was validated on controls, that verdict belongs in the
    # report: it says whether the judge-merged numbers can be believed at all.
    judge_check = db.load_analysis(conn, run_id, "judge_check")
    if judge_check:
        headline["judge_check"] = {k: v for k, v in judge_check.items() if k != "examples"}

    result = AnalysisResult(run_id=run_id, schema_name=schema_name, tables=tables,
                            headline=headline, aggregations=aggregations)
    db.save_analysis(conn, run_id, "headline", headline)
    # predictions + accuracy_* make the path extraction -> accuracy queryable in SQL.
    for name in ("trust_rule_unanimous", "discrimination", "aggregator_comparison", "per_field_trust",
                 "predictions", "accuracy_per_config", "accuracy_per_field", "accuracy_per_config_field"):
        frame = tables.get(name)
        if frame is not None and not frame.empty:
            db.save_analysis(conn, run_id, name, frame.to_dict(orient="records"))
    conn.commit()
    return result
