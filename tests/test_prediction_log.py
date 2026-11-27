"""Prediction logging, tested against a temporary SQLite database (no Postgres needed)."""
import pandas as pd
import pytest

from churn.config import DECISION_THRESHOLD, FEATURE_COLS
from churn.prediction_log import (
    create_table,
    get_engine,
    load_logged_inputs,
    log_predictions,
    to_rows,
)


@pytest.fixture
def engine(tmp_path):
    eng = get_engine(f"sqlite:///{tmp_path / 'monitoring.db'}")
    create_table(eng)
    return eng


@pytest.fixture
def customers(raw_df):
    return raw_df[FEATURE_COLS].head(5).reset_index(drop=True)


SCORES = [0.10, 0.44, 0.45, 0.80, 0.99]


def test_logs_one_row_per_prediction(engine, customers):
    n = log_predictions(engine, customers, SCORES, model_version="v1")
    assert n == 5
    logged = load_logged_inputs(engine)
    assert len(logged) == 5
    assert set(FEATURE_COLS) <= set(logged.columns)  # back to dataset column names


def test_send_offer_uses_threshold(engine, customers):
    log_predictions(engine, customers, SCORES, model_version="v1")
    logged = load_logged_inputs(engine).sort_values("id")
    expected = [s >= DECISION_THRESHOLD for s in SCORES]
    assert logged["send_offer"].astype(bool).tolist() == expected


def test_batch_shares_request_id(customers):
    rows = to_rows(customers, SCORES, model_version="v1", endpoint="/predict/batch")
    assert len({r["request_id"] for r in rows}) == 1
    assert all(r["endpoint"] == "/predict/batch" for r in rows)


def test_blank_total_charges_stored_as_null(engine, customers):
    customers.loc[0, "TotalCharges"] = " "  # new customer, not billed yet
    log_predictions(engine, customers, SCORES, model_version="v1")
    logged = load_logged_inputs(engine).sort_values("id")
    assert pd.isna(logged["TotalCharges"].iloc[0])


def test_filter_by_model_version(engine, customers):
    log_predictions(engine, customers, SCORES, model_version="v1")
    log_predictions(engine, customers.head(2), SCORES[:2], model_version="v2")
    assert len(load_logged_inputs(engine, model_version="v2")) == 2


def test_mismatched_lengths_rejected(customers):
    with pytest.raises(ValueError):
        to_rows(customers, SCORES[:3], model_version="v1")


def test_no_database_means_logging_off(customers, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert get_engine() is None
    assert log_predictions(None, customers, SCORES, model_version="v1") == 0


def test_compose_url_uses_psycopg3():
    """docker-compose sets DATABASE_URL=postgresql://...; it must use the psycopg 3 driver."""
    eng = get_engine("postgresql://churn:churn@postgres:5432/churn")
    assert eng.dialect.driver == "psycopg"


def test_database_error_does_not_raise(tmp_path, customers):
    eng = get_engine(f"sqlite:///{tmp_path / 'empty.db'}")  # table never created
    assert log_predictions(eng, customers, SCORES, model_version="v1") == 0
