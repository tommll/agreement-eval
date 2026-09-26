-- Postgres schema for the agreement-vs-accuracy study.
--
-- Grain notes:
--   extractions   : one row per (run, config, document) model call
--   field_outputs : one row per (extraction, field)  <- everything analytical reads this
--   judgments     : one row per judged pair, or per (prediction, gold) verdict
-- Re-running `extract` is resumable because of the unique key on extractions.

CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    dataset     TEXT NOT NULL,
    split       TEXT,
    gold        JSONB NOT NULL,
    ocr_text    TEXT,
    image_uri   TEXT,
    image_path  TEXT,
    meta        JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS documents_dataset_split_idx ON documents (dataset, split);

CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    dataset      TEXT NOT NULL,
    schema_name  TEXT NOT NULL,
    notes        TEXT,
    experiment   JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS configs (
    run_id          TEXT NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
    config_id       TEXT NOT NULL,
    provider        TEXT NOT NULL,
    model           TEXT NOT NULL,
    prompt_variant  TEXT NOT NULL,
    input_mode      TEXT NOT NULL,
    effort          TEXT,
    prompt_version  TEXT NOT NULL,
    params          JSONB NOT NULL DEFAULT '{}'::JSONB,
    PRIMARY KEY (run_id, config_id)
);

CREATE TABLE IF NOT EXISTS extractions (
    id             BIGSERIAL PRIMARY KEY,
    run_id         TEXT NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
    config_id      TEXT NOT NULL,
    doc_id         TEXT NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
    status         TEXT NOT NULL CHECK (status IN ('ok', 'error')),
    raw_output     JSONB,
    error          TEXT,
    latency_ms     INTEGER,
    input_tokens   INTEGER,
    output_tokens  INTEGER,
    cost_usd       NUMERIC(12, 6),
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, config_id, doc_id)
);
CREATE INDEX IF NOT EXISTS extractions_run_idx ON extractions (run_id, config_id);

CREATE TABLE IF NOT EXISTS field_outputs (
    id             BIGSERIAL PRIMARY KEY,
    extraction_id  BIGINT NOT NULL REFERENCES extractions (id) ON DELETE CASCADE,
    run_id         TEXT NOT NULL,
    config_id      TEXT NOT NULL,
    doc_id         TEXT NOT NULL,
    field          TEXT NOT NULL,
    field_type     TEXT NOT NULL,
    value_raw      TEXT,
    value_norm     TEXT,
    is_null        BOOLEAN NOT NULL,
    evidence       TEXT,
    UNIQUE (extraction_id, field)
);
CREATE INDEX IF NOT EXISTS field_outputs_item_idx ON field_outputs (run_id, doc_id, field);
CREATE INDEX IF NOT EXISTS field_outputs_config_idx ON field_outputs (run_id, config_id);

-- config_id is only unique within a run, so these FKs are composite. Added
-- here rather than inline so databases created before them pick them up too.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'extractions_config_fk') THEN
        ALTER TABLE extractions ADD CONSTRAINT extractions_config_fk
            FOREIGN KEY (run_id, config_id) REFERENCES configs (run_id, config_id) ON DELETE CASCADE;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'field_outputs_config_fk') THEN
        ALTER TABLE field_outputs ADD CONSTRAINT field_outputs_config_fk
            FOREIGN KEY (run_id, config_id) REFERENCES configs (run_id, config_id) ON DELETE CASCADE;
    END IF;
END $$;

-- LLM-judge verdicts. kind='pair' compares two model outputs (semantic
-- agreement); kind='gold' compares one model output against the label
-- (semantic correctness). For kind='gold', right_config is the literal 'GOLD'
-- so the unique key works without NULL semantics.
CREATE TABLE IF NOT EXISTS judgments (
    id            BIGSERIAL PRIMARY KEY,
    run_id        TEXT NOT NULL,
    kind          TEXT NOT NULL CHECK (kind IN ('pair', 'gold')),
    doc_id        TEXT NOT NULL,
    field         TEXT NOT NULL,
    left_config   TEXT NOT NULL,
    right_config  TEXT NOT NULL,
    left_value    TEXT,
    right_value   TEXT,
    equivalent    BOOLEAN,
    confidence    DOUBLE PRECISION,
    rationale     TEXT,
    judge_model   TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, kind, doc_id, field, left_config, right_config)
);
CREATE INDEX IF NOT EXISTS judgments_item_idx ON judgments (run_id, doc_id, field);

-- Analysis output kept next to the data it came from (also written to CSV).
CREATE TABLE IF NOT EXISTS analysis_results (
    id          BIGSERIAL PRIMARY KEY,
    run_id      TEXT NOT NULL,
    name        TEXT NOT NULL,
    payload     JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, name)
);

CREATE OR REPLACE VIEW field_predictions AS
SELECT fo.run_id,
       fo.doc_id,
       fo.field,
       fo.field_type,
       fo.config_id,
       fo.value_raw,
       fo.value_norm,
       fo.is_null,
       fo.evidence,
       d.dataset,
       d.split,
       d.gold ->> fo.field AS gold_raw
FROM field_outputs fo
JOIN documents d USING (doc_id);
