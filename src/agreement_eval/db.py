"""Postgres access layer.

All state lives in Postgres: ingested documents, every model output at field
grain, judge verdicts, and analysis payloads. Nothing downstream reads files,
so a run can be re-analyzed (or re-judged) months later without re-calling any
model.

Connection string comes from $DATABASE_URL, falling back to the local
docker-compose service defined at the repo root.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from importlib import resources
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import pandas as pd
import psycopg
from psycopg.rows import dict_row

from .documents import Document

DEFAULT_DSN = "postgresql://agreement:agreement@localhost:5432/agreement"


def jsonable(value: Any) -> Any:
    """Make a payload safe for JSONB.

    Postgres rejects the JSON literals NaN and Infinity, and these payloads are
    full of rates that are legitimately undefined (an AUC with one class, a
    mean over zero rows). They become null.
    """
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (bool, str)) or value is None:
        return value
    if hasattr(value, "item") and not isinstance(value, (int, float)):
        value = value.item()  # numpy scalar
    if isinstance(value, float):
        return None if (value != value or value in (float("inf"), float("-inf"))) else value
    if isinstance(value, int):
        return value
    return str(value)


def dsn() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DSN)


@contextmanager
def connect(url: Optional[str] = None) -> Iterator[psycopg.Connection]:
    """Open a connection; commits on clean exit, rolls back on exception."""
    conn = psycopg.connect(url or dsn(), row_factory=dict_row)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def schema_sql() -> str:
    return resources.files("agreement_eval.sql").joinpath("schema.sql").read_text()


def init_db(conn: psycopg.Connection) -> None:
    conn.execute(schema_sql())


def reset_run(conn: psycopg.Connection, run_id: str) -> None:
    """Delete a run and everything hanging off it (cascades to extractions)."""
    conn.execute("DELETE FROM judgments WHERE run_id = %s", (run_id,))
    conn.execute("DELETE FROM analysis_results WHERE run_id = %s", (run_id,))
    conn.execute("DELETE FROM runs WHERE run_id = %s", (run_id,))


# --------------------------------------------------------------------------
# documents
# --------------------------------------------------------------------------

def upsert_documents(conn: psycopg.Connection, docs: Iterable[Document]) -> int:
    rows = [d.to_row() for d in docs]
    if not rows:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO documents (doc_id, dataset, split, gold, ocr_text, image_uri, image_path, meta)
            VALUES (%(doc_id)s, %(dataset)s, %(split)s, %(gold)s, %(ocr_text)s,
                    %(image_uri)s, %(image_path)s, %(meta)s)
            ON CONFLICT (doc_id) DO UPDATE SET
                dataset    = EXCLUDED.dataset,
                split      = EXCLUDED.split,
                gold       = EXCLUDED.gold,
                ocr_text   = EXCLUDED.ocr_text,
                image_uri  = EXCLUDED.image_uri,
                image_path = COALESCE(EXCLUDED.image_path, documents.image_path),
                meta       = EXCLUDED.meta
            """,
            rows,
        )
    return len(rows)


def set_image_path(conn: psycopg.Connection, doc_id: str, path: str) -> None:
    conn.execute("UPDATE documents SET image_path = %s WHERE doc_id = %s", (path, doc_id))


def fetch_documents(
    conn: psycopg.Connection,
    dataset: str,
    split: Optional[str] = None,
    limit: Optional[int] = None,
    doc_ids: Optional[Sequence[str]] = None,
) -> List[Document]:
    sql = "SELECT * FROM documents WHERE dataset = %s"
    params: List[Any] = [dataset]
    if split:
        sql += " AND split = %s"
        params.append(split)
    if doc_ids:
        sql += " AND doc_id = ANY(%s)"
        params.append(list(doc_ids))
    sql += " ORDER BY doc_id"
    if limit:
        sql += " LIMIT %s"
        params.append(limit)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return [Document.from_row(r) for r in cur.fetchall()]


# --------------------------------------------------------------------------
# runs & configs
# --------------------------------------------------------------------------

def register_run(
    conn: psycopg.Connection,
    run_id: str,
    dataset: str,
    schema_name: str,
    experiment: Dict[str, Any],
    notes: Optional[str] = None,
) -> None:
    conn.execute(
        """
        INSERT INTO runs (run_id, dataset, schema_name, notes, experiment)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (run_id) DO UPDATE SET
            dataset = EXCLUDED.dataset,
            schema_name = EXCLUDED.schema_name,
            notes = COALESCE(EXCLUDED.notes, runs.notes),
            experiment = EXCLUDED.experiment
        """,
        (run_id, dataset, schema_name, notes, json.dumps(jsonable(experiment))),
    )


def register_configs(conn: psycopg.Connection, run_id: str, configs: Sequence[Dict[str, Any]]) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO configs (run_id, config_id, provider, model, prompt_variant,
                                 input_mode, effort, prompt_version, params)
            VALUES (%(run_id)s, %(config_id)s, %(provider)s, %(model)s, %(prompt_variant)s,
                    %(input_mode)s, %(effort)s, %(prompt_version)s, %(params)s)
            ON CONFLICT (run_id, config_id) DO UPDATE SET
                provider = EXCLUDED.provider,
                model = EXCLUDED.model,
                prompt_variant = EXCLUDED.prompt_variant,
                input_mode = EXCLUDED.input_mode,
                effort = EXCLUDED.effort,
                prompt_version = EXCLUDED.prompt_version,
                params = EXCLUDED.params
            """,
            [{**c, "run_id": run_id, "params": json.dumps(jsonable(c.get("params", {})))} for c in configs],
        )


def run_info(conn: psycopg.Connection, run_id: str) -> Optional[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM runs WHERE run_id = %s", (run_id,))
        return cur.fetchone()


# --------------------------------------------------------------------------
# extractions
# --------------------------------------------------------------------------

def completed_keys(conn: psycopg.Connection, run_id: str, include_errors: bool = False) -> Set[Tuple[str, str]]:
    """(config_id, doc_id) pairs already stored - used to resume a run."""
    sql = "SELECT config_id, doc_id FROM extractions WHERE run_id = %s"
    if not include_errors:
        sql += " AND status = 'ok'"
    with conn.cursor() as cur:
        cur.execute(sql, (run_id,))
        return {(r["config_id"], r["doc_id"]) for r in cur.fetchall()}


def insert_extraction(
    conn: psycopg.Connection,
    *,
    run_id: str,
    config_id: str,
    doc_id: str,
    status: str,
    raw_output: Optional[Dict[str, Any]],
    error: Optional[str],
    latency_ms: Optional[int],
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    cost_usd: Optional[float],
    field_rows: Sequence[Dict[str, Any]],
) -> int:
    """Insert one model call plus its per-field rows, atomically.

    Re-running an (run, config, doc) replaces the previous attempt so a retry
    after a transient failure leaves exactly one row.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO extractions (run_id, config_id, doc_id, status, raw_output, error,
                                     latency_ms, input_tokens, output_tokens, cost_usd)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id, config_id, doc_id) DO UPDATE SET
                status = EXCLUDED.status,
                raw_output = EXCLUDED.raw_output,
                error = EXCLUDED.error,
                latency_ms = EXCLUDED.latency_ms,
                input_tokens = EXCLUDED.input_tokens,
                output_tokens = EXCLUDED.output_tokens,
                cost_usd = EXCLUDED.cost_usd,
                created_at = now()
            RETURNING id
            """,
            (run_id, config_id, doc_id, status,
             json.dumps(jsonable(raw_output)) if raw_output is not None else None,
             error, latency_ms, input_tokens, output_tokens, cost_usd),
        )
        extraction_id = cur.fetchone()["id"]
        cur.execute("DELETE FROM field_outputs WHERE extraction_id = %s", (extraction_id,))
        if field_rows:
            cur.executemany(
                """
                INSERT INTO field_outputs (extraction_id, run_id, config_id, doc_id, field,
                                           field_type, value_raw, value_norm, is_null, evidence)
                VALUES (%(extraction_id)s, %(run_id)s, %(config_id)s, %(doc_id)s, %(field)s,
                        %(field_type)s, %(value_raw)s, %(value_norm)s, %(is_null)s, %(evidence)s)
                """,
                [{**row, "extraction_id": extraction_id, "run_id": run_id,
                  "config_id": config_id, "doc_id": doc_id} for row in field_rows],
            )
    return extraction_id


def load_predictions(conn: psycopg.Connection, run_id: str) -> pd.DataFrame:
    """Long frame: one row per (doc, field, config) with the gold value attached."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT doc_id, field, field_type, config_id, value_raw, value_norm,
                   is_null, evidence, gold_raw, dataset, split
            FROM field_predictions
            WHERE run_id = %s
            ORDER BY doc_id, field, config_id
            """,
            (run_id,),
        )
        return pd.DataFrame(cur.fetchall())


def load_extraction_stats(conn: psycopg.Connection, run_id: str) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT config_id,
                   count(*)                                  AS calls,
                   count(*) FILTER (WHERE status = 'error')  AS errors,
                   avg(latency_ms)                           AS mean_latency_ms,
                   sum(input_tokens)                         AS input_tokens,
                   sum(output_tokens)                        AS output_tokens,
                   sum(cost_usd)                             AS cost_usd
            FROM extractions WHERE run_id = %s GROUP BY config_id ORDER BY config_id
            """,
            (run_id,),
        )
        return pd.DataFrame(cur.fetchall())


def load_local_timings(conn: psycopg.Connection, run_id: str) -> pd.DataFrame:
    """Per-config prefill/decode rates, for runs served by llama.cpp.

    Empty for API providers - the key simply is not in their raw_output.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT config_id,
                   count(*)                                                       AS calls,
                   avg((raw_output -> 'timings' ->> 'prompt_n')::float)           AS mean_prompt_tokens,
                   avg((raw_output -> 'timings' ->> 'prompt_per_second')::float)  AS prefill_tok_s,
                   avg((raw_output -> 'timings' ->> 'predicted_n')::float)        AS mean_output_tokens,
                   avg((raw_output -> 'timings' ->> 'predicted_per_second')::float) AS decode_tok_s,
                   avg(latency_ms) / 1000.0                                       AS mean_latency_s
            FROM extractions
            WHERE run_id = %s AND raw_output -> 'timings' IS NOT NULL
            GROUP BY config_id ORDER BY config_id
            """,
            (run_id,),
        )
        return pd.DataFrame(cur.fetchall())


# --------------------------------------------------------------------------
# judgments
# --------------------------------------------------------------------------

GOLD_SIDE = "GOLD"


def insert_judgments(conn: psycopg.Connection, run_id: str, rows: Sequence[Dict[str, Any]]) -> int:
    if not rows:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO judgments (run_id, kind, doc_id, field, left_config, right_config,
                                   left_value, right_value, equivalent, confidence,
                                   rationale, judge_model)
            VALUES (%(run_id)s, %(kind)s, %(doc_id)s, %(field)s, %(left_config)s, %(right_config)s,
                    %(left_value)s, %(right_value)s, %(equivalent)s, %(confidence)s,
                    %(rationale)s, %(judge_model)s)
            ON CONFLICT (run_id, kind, doc_id, field, left_config, right_config) DO UPDATE SET
                left_value = EXCLUDED.left_value,
                right_value = EXCLUDED.right_value,
                equivalent = EXCLUDED.equivalent,
                confidence = EXCLUDED.confidence,
                rationale = EXCLUDED.rationale,
                judge_model = EXCLUDED.judge_model,
                created_at = now()
            """,
            [{**r, "run_id": run_id} for r in rows],
        )
    return len(rows)


def load_judgments(conn: psycopg.Connection, run_id: str, kind: Optional[str] = None) -> pd.DataFrame:
    sql = "SELECT * FROM judgments WHERE run_id = %s"
    params: List[Any] = [run_id]
    if kind:
        sql += " AND kind = %s"
        params.append(kind)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return pd.DataFrame(cur.fetchall())


def judged_keys(conn: psycopg.Connection, run_id: str, kind: str) -> Set[Tuple[str, str, str, str]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT doc_id, field, left_config, right_config FROM judgments WHERE run_id = %s AND kind = %s",
            (run_id, kind),
        )
        return {(r["doc_id"], r["field"], r["left_config"], r["right_config"]) for r in cur.fetchall()}


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------

def save_analysis(conn: psycopg.Connection, run_id: str, name: str, payload: Any) -> None:
    conn.execute(
        """
        INSERT INTO analysis_results (run_id, name, payload)
        VALUES (%s, %s, %s)
        ON CONFLICT (run_id, name) DO UPDATE SET payload = EXCLUDED.payload, created_at = now()
        """,
        (run_id, name, json.dumps(jsonable(payload))),
    )


def load_analysis(conn: psycopg.Connection, run_id: str, name: str) -> Optional[Any]:
    with conn.cursor() as cur:
        cur.execute("SELECT payload FROM analysis_results WHERE run_id = %s AND name = %s", (run_id, name))
        row = cur.fetchone()
        return row["payload"] if row else None
