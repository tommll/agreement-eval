"""Markdown report: the numbers a team would be shown, in reading order."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .analysis import AnalysisResult


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if np.isnan(value):
            return "-"
        return f"{value:.3f}" if abs(value) < 1000 else f"{value:.1f}"
    if isinstance(value, (np.bool_, bool)):
        return "yes" if value else "no"
    return str(value)


def to_markdown(frame: pd.DataFrame, max_rows: int = 40, index: bool = False) -> str:
    """Small markdown table renderer (keeps `tabulate` out of the dependencies)."""
    if frame is None or frame.empty:
        return "_(no rows)_\n"
    df = frame.head(max_rows)
    if index:
        df = df.reset_index()
    columns = list(df.columns)
    header = "| " + " | ".join(str(c) for c in columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, rule]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    if len(frame) > max_rows:
        lines.append(f"| ... | _{len(frame) - max_rows} more rows in the CSV_ |")
    return "\n".join(lines) + "\n"


def _verdict(headline: Dict[str, Any]) -> str:
    """One paragraph that answers the question the study was run to answer."""
    pop = headline.get("populated", {})
    null = headline.get("null", {})
    overall_auc = headline.get("auc_pairwise_agreement", float("nan"))
    pop_auc = pop.get("auc", float("nan"))
    p_unanimous = headline.get("p_correct_given_unanimous", float("nan"))
    p_unanimous_pop = pop.get("p_correct_given_unanimous", float("nan"))
    base = headline.get("consensus_accuracy", float("nan"))

    excluded = headline.get("n_items_single_rater_excluded") or 0
    lines = [
        (f"_{excluded} of {headline.get('n_items_total')} items had only one config answer "
         f"(the others errored) and are excluded below: with a single rater there is nothing to "
         f"agree about._\n" if excluded else ""),
        f"Across **{headline.get('n_items', 0)} items**, the configs were unanimous on "
        f"**{_fmt(headline.get('unanimity_rate'))}** of them. Unanimous items were correct "
        f"**{_fmt(p_unanimous)}** of the time against a base rate of **{_fmt(base)}**; split items "
        f"were correct **{_fmt(headline.get('p_correct_given_split'))}** of the time.",
        "",
        f"Restricted to fields the label says are **populated**, unanimity is worth "
        f"{_fmt(p_unanimous_pop)} (n={pop.get('n', 0)}); on **null** fields it is "
        f"{_fmt(null.get('p_correct_given_unanimous'))} (n={null.get('n', 0)}). Pairwise agreement "
        f"discriminates correct from incorrect consensus at AUC {_fmt(overall_auc)} overall and "
        f"{_fmt(pop_auc)} on populated fields only.",
    ]
    if isinstance(overall_auc, float) and isinstance(pop_auc, float) and not np.isnan(pop_auc) and not np.isnan(overall_auc):
        if overall_auc - pop_auc > 0.03:
            lines += ["", "> The overall number is flattered by null fields: agreement is a "
                          "markedly weaker signal once fields that are genuinely absent are excluded."]
    if null.get("n", 0) == 0:
        lines += ["", "> Every labelled field in this dataset is populated (SROIE labels all four "
                      "fields on every receipt), so there is no gold-null arm to split on. The null "
                      "side of the question is answered here by `trust_rule_by_consensus_state` - "
                      "items where the configs *decided* the field was absent - and by running the "
                      "same study on FUNSD, where most fields genuinely are absent."]
    elif pop.get("n", 0) == 0:
        lines += ["", "> Every labelled field in this dataset is null; the populated arm is empty."]
    ece = headline.get("ece_agreement_vs_accuracy")
    if isinstance(ece, float) and not np.isnan(ece):
        lines += ["", f"Reading an agreement rate *as if it were* an accuracy estimate is off by "
                      f"{_fmt(ece)} on average (expected calibration error)."]
    agg = headline.get("aggregation") or {}
    if agg:
        delta = agg.get("ds_minus_mv", float("nan"))
        direction = "beats" if isinstance(delta, float) and delta > 0 else "does not beat"
        lines += ["", f"Dawid-Skene {direction} majority vote "
                      f"({_fmt(agg.get('dawid_skene_one_coin'))} vs {_fmt(agg.get('majority_vote'))}); "
                      f"the best single config scores {_fmt(agg.get('best_single_config'))} and the "
                      f"ceiling (any config correct) is {_fmt(agg.get('oracle_any_config'))}."]
    return "\n".join(lines)


SECTIONS: Sequence[tuple] = (
    ("Does agreement predict correctness?", ("trust_rule_unanimous", "agreement_buckets", "discrimination",
                                             "discrimination_jaccard")),
    ("Calibration", ("calibration",)),
    ("Per field", ("per_field_trust", "accuracy_per_field")),
    ("Per prediction (is *my* answer backed by the others?)", ("prediction_level_support", "support_buckets")),
    ("Aggregation: majority vote vs. Dawid-Skene", ("aggregator_comparison", "ds_worker_quality", "ds_signal", "ds_binary_quality")),
    ("Per config accuracy", ("accuracy_per_config", "accuracy_per_config_field")),
    ("Raw agreement between configs", ("agreement_matrix", "jaccard_matrix")),
    ("Run cost and reliability", ("extraction_stats", "local_speed")),
)

TABLE_NOTES = {
    "discrimination_jaccard": "Partial credit: the outcome is a consensus within the Jaccard threshold of the "
                              "label, and `jaccard_agreement` is the mean pairwise token-set Jaccard.",
    "jaccard_matrix": "Mean pairwise token-set Jaccard between configs (atomic fields score 1 or 0).",
    "trust_rule_unanimous": "Auto-accept unanimous items, review the rest. `errors_per_1000_accepted` "
                            "is the error budget that policy spends.",
    "agreement_buckets": "P(consensus correct) by agreement bucket, split by whether the label is populated.",
    "discrimination": "AUC of each agreement score for predicting consensus correctness. 0.5 = no signal.",
    "calibration": "If agreement were an accuracy estimate, `gap` would be zero.",
    "per_field_trust": "The same trust rule per field - averages hide fields where agreement is useless.",
    "support_buckets": "Prediction-level: how often a single config's answer is right given how many "
                       "other configs back it.",
    "aggregator_comparison": "Same items, different ways of combining the configs.",
    "ds_worker_quality": "Dawid-Skene estimates reliability without labels; compare to measured accuracy.",
    "ds_signal": "Is the Dawid-Skene posterior a better trust signal than raw agreement?",
    "agreement_matrix": "Pairwise exact-match agreement rate between configs.",
    "local_speed": "llama.cpp's own timings for locally served configs. Prefill is compute-bound "
                   "and decode is memory-bandwidth-bound, so they move independently.",
}


def render(result: AnalysisResult, experiment: Optional[Dict[str, Any]] = None) -> str:
    headline = result.headline
    parts = [
        f"# Agreement vs. accuracy - run `{result.run_id}`",
        "",
        f"Dataset schema: `{result.schema_name}`. "
        f"Configs: {len(result.tables.get('accuracy_per_config', pd.DataFrame()))}. "
        f"Items (document x field): {headline.get('n_items', 0)}.",
        "",
        "## Verdict",
        "",
        _verdict(headline),
        "",
    ]
    exact_vs_judge = headline.get("exact_vs_judge") or {}
    if exact_vs_judge.get("n_pair_judgments") or exact_vs_judge.get("n_gold_judgments"):
        parts += [
            "## Judge",
            "",
            f"Exact-match accuracy {_fmt(exact_vs_judge.get('exact_accuracy'))} -> "
            f"judge-lenient accuracy {_fmt(exact_vs_judge.get('judge_lenient_accuracy'))} "
            f"over {exact_vs_judge.get('n_gold_judgments', 0)} judged predictions and "
            f"{exact_vs_judge.get('n_pair_judgments', 0)} judged pairs. The gap is disagreement "
            "that was only ever about formatting.",
            "",
        ]
        semantic = headline.get("semantic_agreement")
        if semantic:
            parts += [
                f"After merging judge-equivalent values, unanimity rises to "
                f"{_fmt(semantic.get('unanimity_rate'))} and P(correct | unanimous) is "
                f"{_fmt(semantic.get('p_correct_given_unanimous'))} "
                f"({_fmt(semantic.get('populated_p_correct_given_unanimous'))} on populated fields), "
                f"AUC {_fmt(semantic.get('auc'))}.",
                "",
            ]

    check = headline.get("judge_check")
    if check:
        by_kind = check.get("by_kind") or {}
        parts += [
            "## Judge validation",
            "",
            f"Before trusting the judge to merge values, it was run on {check.get('n', 0)} controls with "
            f"known answers ({check.get('provider')}/{check.get('model')}): identical strings, cosmetic "
            f"rewrites, and values taken from different documents.",
            "",
            to_markdown(pd.DataFrame([
                {"control": kind, "n": v.get("n"), "judge accuracy": v.get("accuracy")}
                for kind, v in sorted(by_kind.items())
            ])),
            "",
            f"Overall {_fmt(check.get('accuracy'))}; false-equivalence rate on values that are genuinely "
            f"different: {_fmt(check.get('false_equivalence_rate'))}.",
            "",
        ]
        identical = (by_kind.get("identical") or {}).get("accuracy")
        false_eq = check.get("false_equivalence_rate")
        if (isinstance(identical, float) and identical < 0.95) or (isinstance(false_eq, float) and false_eq > 0.1):
            parts += ["> This judge does not pass its own controls, so its verdicts are not used in the "
                      "numbers below. A judge that cannot recognise two identical strings - or that merges "
                      "unrelated values - would manufacture exactly the agreement this study is trying to "
                      "measure.", ""]

    for title, names in SECTIONS:
        available = [n for n in names if isinstance(result.tables.get(n), pd.DataFrame) and not result.tables[n].empty]
        if not available:
            continue
        parts += [f"## {title}", ""]
        for name in available:
            parts += [f"**{name}**", ""]
            if name in TABLE_NOTES:
                parts += [f"_{TABLE_NOTES[name]}_", ""]
            parts += [to_markdown(result.tables[name], index=name.endswith("_matrix")), ""]

    parts += [
        "## Method notes",
        "",
        "- Agreement and accuracy use the *same* normalizer per field type, so neither is measured "
        "more leniently than the other.",
        "- `null` means normalized-absent: 'N/A', '-' and '' all count as null, on both sides.",
        "- Items are (document, field) pairs; a config that errored on a document contributes no "
        "rows for it, so item rater counts can differ.",
        "- Dawid-Skene over open-vocabulary values uses the one-coin model (one reliability per "
        "config); the populated/null decision additionally gets a full 2x2 confusion-matrix model.",
        "",
    ]
    if experiment:
        parts += ["<details><summary>Experiment config</summary>", "", "```yaml",
                  _yaml_ish(experiment), "```", "", "</details>", ""]
    return "\n".join(parts)


def _yaml_ish(payload: Dict[str, Any], indent: int = 0) -> str:
    lines = []
    pad = " " * indent
    for key, value in payload.items():
        if isinstance(value, dict):
            lines.append(f"{pad}{key}:")
            lines.append(_yaml_ish(value, indent + 2))
        elif isinstance(value, list):
            lines.append(f"{pad}{key}:")
            for item in value:
                if isinstance(item, dict):
                    lines.append(f"{pad}  -")
                    lines.append(_yaml_ish(item, indent + 4))
                else:
                    lines.append(f"{pad}  - {item}")
        else:
            lines.append(f"{pad}{key}: {value}")
    return "\n".join(lines)


def write_report(result: AnalysisResult, outdir: Path, experiment: Optional[Dict[str, Any]] = None) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "report.md"
    path.write_text(render(result, experiment))
    return path
