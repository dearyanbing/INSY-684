import numpy as np
import pytest

from churn.data import clean, split
from churn.features import build_preprocessor
from churn.models import build_pipeline, candidate_models, cv_pr_auc, evaluate


def test_preprocessor_output_has_no_missing_values(raw_df):
    X_train, _, _, _ = split(clean(raw_df))
    X_t = build_preprocessor().fit_transform(X_train)
    X_t = X_t.toarray() if hasattr(X_t, "toarray") else X_t
    assert X_t.shape[0] == len(X_train)
    assert not np.isnan(X_t).any()


def test_preprocessor_handles_unseen_category(raw_df):
    X_train, X_test, _, _ = split(clean(raw_df))
    pre = build_preprocessor().fit(X_train)
    X_test = X_test.copy()
    X_test.iloc[0, X_test.columns.get_loc("PaymentMethod")] = "Crypto"
    pre.transform(X_test)  # should not raise


@pytest.mark.parametrize("name", list(candidate_models()))
def test_each_model_trains_and_predicts(raw_df, name):
    X_train, X_test, y_train, y_test = split(clean(raw_df))
    pipe = build_pipeline(candidate_models()[name]).fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    assert len(proba) == len(X_test)
    assert ((proba >= 0) & (proba <= 1)).all()

    metrics = evaluate(pipe, X_test, y_test)
    assert set(metrics) == {"pr_auc", "roc_auc", "accuracy", "precision", "recall", "f1"}
    assert all(0 <= v <= 1 for v in metrics.values())


def test_cv_pr_auc_runs_on_training_data(raw_df):
    X_train, _, y_train, _ = split(clean(raw_df))
    pipe = build_pipeline(candidate_models()["logistic_regression"])
    mean, std = cv_pr_auc(pipe, X_train, y_train, folds=3)
    assert 0 <= mean <= 1
    assert std >= 0

def test_evaluate_includes_score_equal_to_threshold():
    from unittest.mock import Mock

    model = Mock()
    model.predict_proba.return_value = np.array([
        [0.56, 0.44],
        [0.55, 0.45],
        [0.54, 0.46],
    ])
    actual_churn = [0, 1, 1]

    metrics = evaluate(model, X=[0, 1, 2], y=actual_churn, threshold=0.45)

    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 1.0

def test_evaluate_no_positive_predictions():
    from unittest.mock import Mock

    model = Mock()
    model.predict_proba.return_value = np.array([
        [0.90, 0.10],
        [0.80, 0.20],
        [0.70, 0.30],
    ])

    metrics = evaluate(
        model,
        X=[0, 1, 2],
        y=[0, 1, 1],
        threshold=0.45
    )

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def monitor_performance(predictions):
    # Keep only customers with known outcomes
    labeled = predictions.dropna(
        subset=["actual_churn"]
    )

    if labeled.empty:
        return {
            "status": "waiting_for_labels",
            "sample_size": 0,
        }

    # Evaluate one model version at a time
    if labeled["model_version"].nunique() != 1:
        raise ValueError(
            "Evaluate one model version at a time"
        )

    y_true = labeled["actual_churn"].astype(int)
    y_pred = labeled["send_offer"].astype(int)

    return {
        "status": "ok",
        "sample_size": len(labeled),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        ),
    }
