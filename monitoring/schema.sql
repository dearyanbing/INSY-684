-- Predictions log for the Telco churn API (PostgreSQL).
--
-- Every call to /predict or /predict/batch stores one row per customer:
-- the inputs the model saw, what it predicted, and which model version made it.
-- This table feeds monitoring:
--   * data / prediction drift  -> compare these inputs with the training set (Evidently)
--   * model performance        -> fill actual_churn later, then compute recall/precision
--   * operations               -> volume of requests and offers per day (Grafana)

CREATE TABLE IF NOT EXISTS predictions (
    id                BIGSERIAL PRIMARY KEY,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    request_id        UUID,
    endpoint          TEXT NOT NULL CHECK (endpoint IN ('/predict', '/predict/batch')),
    model_version     TEXT NOT NULL,
    threshold         REAL NOT NULL,

    -- Model inputs (same 19 columns as churn.config.INPUT_COLUMNS)
    gender            TEXT NOT NULL CHECK (gender IN ('Female', 'Male')),
    senior_citizen    SMALLINT NOT NULL CHECK (senior_citizen IN (0, 1)),
    partner           TEXT NOT NULL,
    dependents        TEXT NOT NULL,
    tenure            INTEGER NOT NULL CHECK (tenure BETWEEN 0 AND 120),
    phone_service     TEXT NOT NULL,
    multiple_lines    TEXT NOT NULL,
    internet_service  TEXT NOT NULL CHECK (internet_service IN ('DSL', 'Fiber optic', 'No')),
    online_security   TEXT NOT NULL,
    online_backup     TEXT NOT NULL,
    device_protection TEXT NOT NULL,
    tech_support      TEXT NOT NULL,
    streaming_tv      TEXT NOT NULL,
    streaming_movies  TEXT NOT NULL,
    contract          TEXT NOT NULL CHECK (contract IN ('Month-to-month', 'One year', 'Two year')),
    paperless_billing TEXT NOT NULL,
    payment_method    TEXT NOT NULL,
    monthly_charges   REAL NOT NULL CHECK (monthly_charges >= 0),
    total_charges     REAL CHECK (total_charges >= 0),

    -- Model outputs
    churn_probability REAL NOT NULL CHECK (churn_probability BETWEEN 0 AND 1),
    send_offer        BOOLEAN NOT NULL,

    -- Ground truth, filled in later to measure real performance
    actual_churn      BOOLEAN,
    labeled_at        TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_predictions_created_at    ON predictions (created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_model_version ON predictions (model_version);
