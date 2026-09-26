"""Runs every (document x config) extraction and stores it at field grain.

Model calls happen on a thread pool; database writes happen on the calling
thread (one psycopg connection, used from one thread). Completed work is
skipped on re-run, so an interrupted or partially failed run resumes cheaply.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

from .. import db
from ..config import Experiment, ExtractionConfig
from ..documents import Document
from ..normalize import normalize
from ..schemas import DatasetSchema, get_schema
from .providers import ExtractionResult, Provider, build_provider


def _field_rows(result: ExtractionResult, schema: DatasetSchema) -> List[Dict[str, Any]]:
    rows = []
    for name in schema.field_names:
        ftype = schema.field_type(name)
        raw = result.values.get(name)
        norm = normalize(raw, ftype)
        rows.append({
            "field": name,
            "field_type": ftype,
            "value_raw": raw,
            "value_norm": norm,
            # is_null is the *normalized* verdict: "N/A" and "" count as null,
            # so the populated/null split reflects meaning, not formatting.
            "is_null": norm is None,
            "evidence": (result.evidence or {}).get(name),
        })
    return rows


def _providers_for(experiment: Experiment) -> Dict[str, Provider]:
    """One provider instance per provider name, shared across configs."""
    return {name: build_provider(name) for name in {c.provider for c in experiment.configs}}


def run_extraction(
    conn,
    experiment: Experiment,
    documents: Sequence[Document],
    resume: bool = True,
    show_progress: bool = True,
) -> Dict[str, int]:
    schema = get_schema(experiment.schema_name)
    db.register_run(conn, experiment.run_id, experiment.dataset, experiment.schema_name,
                    experiment.to_dict(), experiment.notes)
    db.register_configs(conn, experiment.run_id, [c.to_row(experiment.schema_name) for c in experiment.configs])
    conn.commit()

    done = db.completed_keys(conn, experiment.run_id) if resume else set()
    tasks: List[Tuple[ExtractionConfig, Document]] = [
        (cfg, doc)
        for cfg in experiment.configs
        for doc in documents
        if (cfg.id, doc.doc_id) not in done
    ]
    stats = {"submitted": len(tasks), "ok": 0, "error": 0, "skipped": len(experiment.configs) * len(documents) - len(tasks)}
    if not tasks:
        return stats

    providers = _providers_for(experiment)

    def call(task: Tuple[ExtractionConfig, Document]) -> Tuple[ExtractionConfig, Document, ExtractionResult]:
        cfg, doc = task
        try:
            result = providers[cfg.provider].extract(doc, schema, cfg)
        except Exception as exc:  # a provider bug must not kill the whole run
            result = ExtractionResult(error=f"{type(exc).__name__}: {exc}")
        return cfg, doc, result

    with ThreadPoolExecutor(max_workers=max(1, experiment.concurrency)) as pool:
        futures = [pool.submit(call, task) for task in tasks]
        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(futures), desc=f"extract {experiment.run_id}", unit="call")
        for future in iterator:
            cfg, doc, result = future.result()
            status = "ok" if result.ok else "error"
            stats["ok" if result.ok else "error"] += 1
            db.insert_extraction(
                conn,
                run_id=experiment.run_id,
                config_id=cfg.id,
                doc_id=doc.doc_id,
                status=status,
                raw_output=result.raw_output,
                error=result.error,
                latency_ms=result.latency_ms,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cost_usd=result.cost_usd,
                field_rows=_field_rows(result, schema) if result.ok else [],
            )
            conn.commit()
    return stats
