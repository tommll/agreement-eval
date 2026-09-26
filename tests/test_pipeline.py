"""End-to-end test against a live Postgres.

Skipped when no database is reachable, so `pytest` still runs on a laptop with
nothing started. Bring one up with `docker compose up -d`.
"""

from __future__ import annotations

import uuid

import pytest

from agreement_eval import db
from agreement_eval.analysis import analyze
from agreement_eval.config import ExtractionConfig, Experiment, JudgeConfig
from agreement_eval.extract.runner import run_extraction
from agreement_eval.ingest import load_dataset


@pytest.fixture(scope="module")
def conn():
    try:
        with db.connect() as connection:
            db.init_db(connection)
            yield connection
    except Exception as exc:  # psycopg.OperationalError and friends
        pytest.skip(f"no Postgres available ({type(exc).__name__}); `docker compose up -d` to run this")


@pytest.fixture()
def experiment():
    run_id = f"test-{uuid.uuid4().hex[:8]}"
    return Experiment(
        run_id=run_id,
        dataset="synthetic",
        schema="synthetic",
        split="demo",
        limit=30,
        concurrency=4,
        configs=[
            ExtractionConfig(id="m1", provider="mock", model="mock", prompt="detailed",
                             params={"cluster": "shared", "seed": 1}),
            ExtractionConfig(id="m2", provider="mock", model="mock", prompt="terse",
                             params={"cluster": "shared", "seed": 1}),
            ExtractionConfig(id="m3", provider="mock", model="mock", prompt="evidence",
                             params={"cluster": "solo", "seed": 1}),
        ],
        judge=JudgeConfig(enabled=False),
    )


def test_extract_then_analyze(conn, experiment):
    docs = load_dataset("synthetic", limit=30, seed=99)
    db.upsert_documents(conn, docs)
    conn.commit()
    stored = db.fetch_documents(conn, "synthetic", "demo", limit=30)
    assert len(stored) == 30

    stats = run_extraction(conn, experiment, stored, show_progress=False)
    assert stats["ok"] == 3 * 30 and stats["error"] == 0

    # Re-running skips completed work rather than re-calling the model.
    again = run_extraction(conn, experiment, stored, show_progress=False)
    assert again["submitted"] == 0 and again["skipped"] == 3 * 30

    result = analyze(conn, experiment.run_id)
    n_fields = len(stored[0].gold)
    assert len(result.tables["items"]) == 30 * n_fields
    assert set(result.tables["accuracy_per_config"]["config_id"]) == {"m1", "m2", "m3"}
    assert 0.0 <= result.headline["consensus_accuracy"] <= 1.0
    assert "dawid_skene_one_coin" in set(result.tables["aggregator_comparison"]["method"])

    # The headline is persisted, so a report can be rebuilt without re-analysis.
    assert db.load_analysis(conn, experiment.run_id, "headline") is not None

    db.reset_run(conn, experiment.run_id)
    conn.commit()
    assert db.run_info(conn, experiment.run_id) is None


def test_predictions_view_joins_gold(conn, experiment):
    docs = load_dataset("synthetic", limit=5, seed=5)
    db.upsert_documents(conn, docs)
    run_extraction(conn, experiment, docs, show_progress=False)
    preds = db.load_predictions(conn, experiment.run_id)
    assert not preds.empty
    assert set(preds.columns) >= {"doc_id", "field", "config_id", "value_norm", "gold_raw"}
    db.reset_run(conn, experiment.run_id)
    conn.commit()
