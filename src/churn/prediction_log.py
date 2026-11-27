"""Save every prediction to the `predictions` table (see monitoring/schema.sql).

The API (src/churn/api.py) only needs two lines:

    from churn.prediction_log import get_engine, log_predictions

    engine = get_engine()        # once, at startup (None if DATABASE_URL is not set)
    ...
    log_predictions(engine, customers_df, scores, model_version=MODEL_URI)

Logging never breaks a prediction: if the database is down, we print a warning and
the API still returns its answer. The logged rows are what the drift report
(churn.drift_report) compares against the training data.
"""
from __future__ import annotations

import logging
import os
import uuid

import pandas as pd
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Engine,
    Float,
    Integer,
    MetaData,
    SmallInteger,
    Table,
    Text,
    Uuid,
    create_engine,
    func,
    insert,
    select,
)

from churn.config import DECISION_THRESHOLD, FEATURE_COLS

log = logging.getLogger(__name__)

# Dataset column name -> database column name (snake_case, as in schema.sql)
COLUMN_MAP = {
    "gender": "gender",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "tenure": "tenure",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
}
assert set(COLUMN_MAP) == set(FEATURE_COLS), "COLUMN_MAP must cover every model input"

# Python mirror of monitoring/schema.sql. In Postgres the table is created by
# schema.sql; this definition lets us insert/query it and create it in tests (SQLite).
metadata = MetaData()
predictions = Table(
    "predictions",
    metadata,
    Column("id", BigInteger().with_variant(Integer, "sqlite"), primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("request_id", Uuid),
    Column("endpoint", Text, nullable=False),
    Column("model_version", Text, nullable=False),
    Column("threshold", Float, nullable=False),
    *[
        Column(db_col, SmallInteger if db_col == "senior_citizen"
               else Integer if db_col == "tenure"
               else Float if db_col in ("monthly_charges", "total_charges")
               else Text,
               nullable=(db_col == "total_charges"))
        for db_col in COLUMN_MAP.values()
    ],
    Column("churn_probability", Float, nullable=False),
    Column("send_offer", Boolean, nullable=False),
    Column("actual_churn", Boolean),
    Column("labeled_at", DateTime(timezone=True)),
)


def get_engine(url: str | None = None) -> Engine | None:
    """Database connection from DATABASE_URL, e.g. the one docker-compose sets:
    postgresql://churn:churn@postgres:5432/churn
    Returns None when no URL is configured, so logging is simply switched off."""
    url = url or os.getenv("DATABASE_URL")
    if not url:
        return None
    # A plain postgresql:// URL makes SQLAlchemy look for psycopg2; we ship psycopg 3
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            url = "postgresql+psycopg://" + url[len(prefix):]
    return create_engine(url, pool_pre_ping=True)


def create_table(engine: Engine) -> None:
    """Create the table if it doesn't exist (used in tests and local runs)."""
    metadata.create_all(engine)


def to_rows(
    customers: pd.DataFrame,
    scores,
    *,
    model_version: str,
    threshold: float = DECISION_THRESHOLD,
    endpoint: str = "/predict",
) -> list[dict]:
    """Turn model inputs + scores into rows for the predictions table."""
    if len(customers) != len(scores):
        raise ValueError(f"{len(customers)} customers but {len(scores)} scores")

    df = customers[FEATURE_COLS].copy()
    # Same rule as data.clean(): blank TotalCharges means a new customer
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.rename(columns=COLUMN_MAP)
    df = df.astype(object).where(df.notna(), None)  # NaN -> NULL

    request_id = uuid.uuid4()  # one id per API call, shared by a batch
    rows = []
    for record, score in zip(df.to_dict(orient="records"), scores):
        score = float(score)
        rows.append({
            **record,
            "senior_citizen": int(record["senior_citizen"]),
            "tenure": int(record["tenure"]),
            "monthly_charges": float(record["monthly_charges"]),
            "request_id": request_id,
            "endpoint": endpoint,
            "model_version": model_version,
            "threshold": threshold,
            "churn_probability": score,
            "send_offer": score >= threshold,
        })
    return rows


def log_predictions(engine: Engine | None, customers: pd.DataFrame, scores, **kwargs) -> int:
    """Insert one row per prediction. Returns the number of rows written
    (0 if logging is off or the database is unreachable)."""
    if engine is None:
        return 0
    try:
        rows = to_rows(customers, scores, **kwargs)
        with engine.begin() as conn:
            conn.execute(insert(predictions), rows)
        return len(rows)
    except Exception as exc:  # never let monitoring break a prediction
        log.warning("Could not log %d prediction(s): %s", len(customers), exc)
        return 0


def load_logged_inputs(
    engine: Engine, since=None, model_version: str | None = None
) -> pd.DataFrame:
    """Read logged predictions back with the dataset's column names, ready to
    compare with the training data."""
    query = select(predictions)
    if since is not None:
        query = query.where(predictions.c.created_at >= pd.Timestamp(since).to_pydatetime())
    if model_version is not None:
        query = query.where(predictions.c.model_version == model_version)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df.rename(columns={v: k for k, v in COLUMN_MAP.items()})
