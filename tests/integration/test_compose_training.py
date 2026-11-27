"""Integration test: the training pipeline against the MLflow server in docker compose.

Checks that the pieces work together, not just one by one: real data -> churn.train ->
MLflow server (runs + artifacts over HTTP) -> model registry @champion -> load and score.

    docker compose up -d --build --wait
    python scripts/download_data.py
    MLFLOW_URL=http://localhost:5001 pytest -m integration -v

Skipped in the normal `pytest` run, because it needs the compose stack.
"""

import os

import mlflow
import pytest
from mlflow import MlflowClient
from sklearn.metrics import roc_auc_score

from churn import train
from churn.config import RAW_DATA_PATH, REGISTERED_MODEL_NAME
from churn.data import clean, load_raw, split

MLFLOW_URL = os.getenv("MLFLOW_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not MLFLOW_URL, reason="set MLFLOW_URL to the compose MLflow server"),
    pytest.mark.skipif(
        not RAW_DATA_PATH.exists(), reason="run scripts/download_data.py first"
    ),
]


def test_train_registers_champion_on_mlflow_server(monkeypatch):
    # point the training script at the compose server instead of the local mlflow.db
    monkeypatch.setattr(train, "MLFLOW_TRACKING_URI", MLFLOW_URL)
    train.main()

    client = MlflowClient(tracking_uri=MLFLOW_URL)
    champion = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
    assert champion.version is not None

    mlflow.set_tracking_uri(MLFLOW_URL)
    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}@champion")
    _, X_test, _, y_test = split(clean(load_raw()))
    proba = model.predict_proba(X_test)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    assert roc_auc_score(y_test, proba) > 0.80
