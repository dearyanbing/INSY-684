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
