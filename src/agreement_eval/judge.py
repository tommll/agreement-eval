"""LLM judge for semantic equivalence.

Exact match after normalization is strict by construction: "TAN WOON YANN" vs
"TAN WOON YANN." is a disagreement, and so is "9.00" vs "RM9.00" if the
normalizer misses a currency form. The judge exists to separate *formatting*
disagreement from *substantive* disagreement, on two queues:

    pair - two model outputs for the same field: do they say the same thing?
    gold - a model output vs. the label: is the model's answer acceptable?

Only pairs that exact match already rejected are sent, and only when both
sides are populated: null-vs-populated is a real disagreement about whether
the field exists, never a formatting difference, so a judge must not be able
to wave it away. That keeps judge cost proportional to disagreement, not to
dataset size.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import pandas as pd
from tqdm import tqdm

from . import db
from .config import JudgeConfig
from .extract.providers import (DEFAULT_LLAMACPP_BASE_URL, LlamaCppError, estimate_cost,
                                llamacpp_chat_json)
from .metrics.agreement import NULL_LABEL, pairwise_table

JUDGE_SYSTEM = """\
You compare two values extracted from the same field of a scanned document and decide whether they
mean the same thing.

Say equivalent=true only when a careful reviewer would accept both as the same answer: the same
amount written differently (9.00 vs RM 9.00), the same date in another format, the same name with
different punctuation, spacing or case, or the same address abbreviated.

Say equivalent=false when the values refer to different things: different amounts, different dates,
different entities, one value being a different field of the document, or one being a truncated
fragment that loses identifying information.

Judge only the two values and the field description you are given. Do not speculate about what the
document might have said."""

def verdict_schema(ids: Sequence[str], include_rationale: bool = True) -> Dict[str, Any]:
    """Schema for one batch of verdicts.

    Three choices here are what make a small local model usable as a judge:

    * the array is pinned to exactly one entry per comparison (min/max), so the
      model cannot quietly answer four of five;
    * `id` is an *enum* of this batch's ids, not a free string. Under a grammar
      a free string is an open invitation: a 4B model wanders into repeating
      its own prompt and runs to the token cap, which destroys every verdict in
      the batch. An enum can only emit one of the literals;
    * `reason` (and `confidence`) are optional for the same reason. A frontier
      judge should keep the rationale - it is the audit trail. A local one
      should not; with rationales off the whole response is booleans and
      literals, and there is nothing left to degenerate into.
    """
    properties: Dict[str, Any] = {
        "id": {"enum": list(ids)},
        "equivalent": {"type": "boolean"},
    }
    required = ["id", "equivalent"]
    if include_rationale:
        properties["confidence"] = {"type": "number"}
        properties["reason"] = {"type": "string"}
        required += ["confidence", "reason"]
    return {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "minItems": len(ids),
                "maxItems": len(ids),
                "items": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            }
        },
        "required": ["verdicts"],
        "additionalProperties": False,
    }


def match_verdicts(items: Sequence["JudgeItem"], verdicts: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Align returned verdicts to the batch: by id when the ids line up, else by position.

    Small models drop or invent the id field even under a grammar. Since the
    batch length is pinned and the prompt is ordered, position is a sound
    fallback - but only when the count matches exactly.
    """
    by_id = {v.get("id"): v for v in verdicts if isinstance(v, dict) and v.get("id")}
    if all(item.id in by_id for item in items):
        return {item.id: by_id[item.id] for item in items}
    if len(verdicts) == len(items):
        return {item.id: verdicts[i] for i, item in enumerate(items)}
    return {item.id: by_id.get(item.id, {}) for item in items}


@dataclass
class JudgeItem:
    id: str
    doc_id: str
    field: str
    field_type: str
    kind: str          # 'pair' | 'gold'
    left_config: str
    right_config: str
    left_value: Optional[str]
    right_value: Optional[str]


class LLMJudge:
    """Runs verdict batches against whichever provider the judge config names.

    The judge can be a local model too: `provider: llamacpp` points it at the
    same llama.cpp server the raters use. That is the honest configuration to
    study on a laptop - but note it makes the judge a peer of the raters rather
    than a stronger referee, which the report should say out loud.
    """

    def __init__(self, cfg: JudgeConfig, client: Any = None) -> None:
        self.cfg = cfg
        self.cost_usd = 0.0
        self._anthropic = None
        self.client = client
        if cfg.provider == "anthropic":
            import anthropic

            self._anthropic = anthropic
            self.client = client or anthropic.Anthropic()
        elif cfg.provider != "llamacpp":
            raise ValueError(f"unsupported judge provider {cfg.provider!r}; use 'anthropic' or 'llamacpp'")

    def _render(self, items: Sequence[JudgeItem]) -> str:
        lines = [f"Compare each of the {len(items)} pairs below. "
                 f"Return exactly {len(items)} verdicts, in this order, one per id.", ""]
        for item in items:
            side = "label" if item.kind == "gold" else "value B"
            lines += [
                f"id: {item.id}",
                f"  field: {item.field} ({item.field_type})",
                f"  value A: {item.left_value!r}",
                f"  {side}: {item.right_value!r}",
            ]
        return "\n".join(lines)

    def _error_verdicts(self, items: Sequence[JudgeItem], message: str) -> List[Dict[str, Any]]:
        return [{"id": item.id, "equivalent": None, "confidence": None, "reason": message}
                for item in items]

    def judge_batch(self, items: Sequence[JudgeItem]) -> List[Dict[str, Any]]:
        if not items:
            return []
        if self.cfg.provider == "llamacpp":
            return self._judge_batch_llamacpp(items)
        return self._judge_batch_anthropic(items)

    def _judge_batch_llamacpp(self, items: Sequence[JudgeItem]) -> List[Dict[str, Any]]:
        params = dict(self.cfg.params or {})
        try:
            text, _usage, _timings = llamacpp_chat_json(
                JUDGE_SYSTEM,
                self._render(items),
                verdict_schema([i.id for i in items], self.cfg.include_rationale),
                base_url=params.get("base_url", DEFAULT_LLAMACPP_BASE_URL),
                model=self.cfg.model,
                max_tokens=self.cfg.max_tokens,
                schema_name="judge_verdicts",
                timeout=float(params.get("timeout", 1800.0)),
                retries=int(params.get("retries", 1)),
                options=params,
            )
        except LlamaCppError as exc:
            return self._error_verdicts(items, f"judge error: {exc}")
        try:
            return json.loads(text).get("verdicts", [])
        except json.JSONDecodeError as exc:
            return self._error_verdicts(items, f"unparseable judge output: {exc}")

    def _judge_batch_anthropic(self, items: Sequence[JudgeItem]) -> List[Dict[str, Any]]:
        output_config: Dict[str, Any] = {
            "format": {"type": "json_schema",
                       "schema": verdict_schema([i.id for i in items], self.cfg.include_rationale)}
        }
        if self.cfg.effort:
            output_config["effort"] = self.cfg.effort
        try:
            response = self.client.messages.create(
                model=self.cfg.model,
                max_tokens=self.cfg.max_tokens,
                system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content": self._render(items)}],
                output_config=output_config,
            )
        except self._anthropic.APIError as exc:
            return self._error_verdicts(items, f"judge error: {type(exc).__name__}: {exc}")

        usage = getattr(response, "usage", None)
        self.cost_usd += estimate_cost(self.cfg.model,
                                       getattr(usage, "input_tokens", 0) or 0,
                                       getattr(usage, "output_tokens", 0) or 0) or 0.0
        text = next((b.text for b in response.content if b.type == "text"), None)
        try:
            verdicts = json.loads(text or "{}").get("verdicts", [])
        except json.JSONDecodeError as exc:
            return self._error_verdicts(items, f"unparseable judge output: {exc}")
        return verdicts


def _pair_queue(preds: pd.DataFrame, existing: set) -> List[JudgeItem]:
    pairs = pairwise_table(preds)
    if pairs.empty:
        return []
    candidates = pairs[
        (~pairs["exact_agree"])
        & (pairs["label_a"] != NULL_LABEL)
        & (pairs["label_b"] != NULL_LABEL)
    ]
    items = []
    for row in candidates.itertuples():
        key = (row.doc_id, row.field, row.config_a, row.config_b)
        if key in existing:
            continue
        items.append(JudgeItem(
            id=f"p{len(items):06d}", doc_id=row.doc_id, field=row.field, field_type=row.field_type,
            kind="pair", left_config=row.config_a, right_config=row.config_b,
            left_value=row.value_a, right_value=row.value_b,
        ))
    return items


def _gold_queue(preds: pd.DataFrame, existing: set) -> List[JudgeItem]:
    candidates = preds[(~preds["correct"]) & preds["pred_populated"] & preds["gold_populated"]]
    items = []
    for row in candidates.itertuples():
        key = (row.doc_id, row.field, row.config_id, db.GOLD_SIDE)
        if key in existing:
            continue
        items.append(JudgeItem(
            id=f"g{len(items):06d}", doc_id=row.doc_id, field=row.field, field_type=row.field_type,
            kind="gold", left_config=row.config_id, right_config=db.GOLD_SIDE,
            left_value=row.value_raw, right_value=row.gold_raw,
        ))
    return items


# --------------------------------------------------------------------------
# Validating the judge itself
# --------------------------------------------------------------------------

def _reformat(value: str, field_type: str) -> Optional[str]:
    """A cosmetic rewrite that must not change the meaning."""
    if field_type == "money":
        return f"RM {value.strip()}"
    if field_type == "date":
        return value.strip().replace("/", "-")
    if field_type in ("org", "text", "address"):
        return value.strip().title() + "."
    return None


def build_controls(preds: pd.DataFrame, n_per_kind: int = 6, seed: int = 0) -> List[Tuple[JudgeItem, bool]]:
    """Control comparisons whose answer is known, for calibrating the judge.

    Three kinds:
      identical    the same string twice            -> must be equivalent
      reformatted  a cosmetic rewrite of the value  -> must be equivalent
      mismatched   another document's value for the
                   same field                       -> must not be equivalent

    A judge that fails these has no business merging values into equivalence
    clusters, and the analysis should run without it.
    """
    populated = preds[preds["pred_populated"]].drop_duplicates(subset=["field", "value_raw"])
    if populated.empty:
        return []
    rng = np.random.default_rng(seed)
    controls: List[Tuple[JudgeItem, bool]] = []

    def add(kind: str, row, left: str, right: str, expected: bool) -> None:
        controls.append((JudgeItem(
            id=f"c{len(controls):04d}", doc_id=f"control/{kind}", field=row.field,
            field_type=row.field_type, kind="pair", left_config="control",
            right_config=kind, left_value=left, right_value=right,
        ), expected))

    sample = populated.sample(n=min(n_per_kind, len(populated)), random_state=seed)
    for row in sample.itertuples():
        add("identical", row, row.value_raw, row.value_raw, True)

    for row in populated.sample(n=min(n_per_kind, len(populated)), random_state=seed + 1).itertuples():
        rewritten = _reformat(str(row.value_raw), row.field_type)
        if rewritten:
            add("reformatted", row, row.value_raw, rewritten, True)

    for field, group in populated.groupby("field"):
        values = group["value_raw"].tolist()
        if len(values) < 2:
            continue
        for _ in range(max(1, n_per_kind // max(1, populated["field"].nunique()))):
            i, j = rng.choice(len(values), size=2, replace=False)
            row = group.iloc[int(i)]
            add("mismatched", row, values[int(i)], values[int(j)], False)
    return controls


def check_judge(
    cfg: JudgeConfig,
    preds: pd.DataFrame,
    n_per_kind: int = 6,
    client: Any = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Run the controls and report how often the judge is right, by kind."""
    controls = build_controls(preds, n_per_kind=n_per_kind)
    if not controls:
        return {"n": 0}
    judge = LLMJudge(cfg, client=client)
    by_id = {item.id: (item, expected) for item, expected in controls}
    results: List[Dict[str, Any]] = []
    batches = [controls[i:i + cfg.batch_size] for i in range(0, len(controls), cfg.batch_size)]
    iterator = tqdm(batches, desc="judge check", unit="batch") if show_progress else batches
    for batch in iterator:
        items = [item for item, _ in batch]
        verdicts = match_verdicts(items, judge.judge_batch(items))
        for item in items:
            verdict = verdicts.get(item.id, {})
            expected = by_id[item.id][1]
            results.append({
                "kind": item.right_config,
                "field": item.field,
                "left": item.left_value,
                "right": item.right_value,
                "expected": expected,
                "verdict": verdict.get("equivalent"),
                "correct": verdict.get("equivalent") is expected,
            })

    frame = pd.DataFrame(results)
    summary: Dict[str, Any] = {
        "n": int(len(frame)),
        "model": cfg.model,
        "provider": cfg.provider,
        "accuracy": float(frame["correct"].mean()),
        "unusable_verdicts": int(frame["verdict"].isna().sum()),
        "by_kind": {
            kind: {"n": int(len(g)), "accuracy": float(g["correct"].mean())}
            for kind, g in frame.groupby("kind")
        },
        "examples": frame.head(12).to_dict(orient="records"),
    }
    # The dangerous error is calling different things the same: that is what
    # merges values into one cluster and manufactures agreement.
    mismatched = frame[frame["kind"] == "mismatched"]
    if len(mismatched):
        summary["false_equivalence_rate"] = float((mismatched["verdict"] == True).mean())  # noqa: E712
    return summary


def run_judging(
    conn,
    run_id: str,
    preds: pd.DataFrame,
    cfg: JudgeConfig,
    show_progress: bool = True,
    client: Any = None,
) -> Dict[str, int]:
    """Judge outstanding pair and gold comparisons; store verdicts in Postgres."""
    stats = {"pair": 0, "gold": 0, "errors": 0}
    if not cfg.enabled or preds.empty:
        return stats

    judge = LLMJudge(cfg, client=client)
    queues: List[Tuple[str, List[JudgeItem]]] = [
        ("pair", _pair_queue(preds, db.judged_keys(conn, run_id, "pair")))
    ]
    if cfg.judge_gold:
        queues.append(("gold", _gold_queue(preds, db.judged_keys(conn, run_id, "gold"))))

    for kind, items in queues:
        if not items:
            continue
        batches = [items[i:i + cfg.batch_size] for i in range(0, len(items), cfg.batch_size)]
        iterator = tqdm(batches, desc=f"judge {kind}", unit="batch") if show_progress else batches
        for batch in iterator:
            verdicts = match_verdicts(batch, judge.judge_batch(batch))
            rows = []
            for item in batch:
                verdict = verdicts.get(item.id, {})
                equivalent = verdict.get("equivalent")
                if equivalent is None:
                    stats["errors"] += 1
                rows.append({
                    "kind": item.kind,
                    "doc_id": item.doc_id,
                    "field": item.field,
                    "left_config": item.left_config,
                    "right_config": item.right_config,
                    "left_value": item.left_value,
                    "right_value": item.right_value,
                    "equivalent": equivalent,
                    "confidence": verdict.get("confidence"),
                    "rationale": verdict.get("reason"),
                    "judge_model": cfg.model,
                })
            db.insert_judgments(conn, run_id, rows)
            conn.commit()
            stats[kind] += len(rows)
    stats["cost_usd"] = round(judge.cost_usd, 4)
    return stats
