"""Candidate models and evaluation. No MLflow here so it is easy to unit test."""
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from churn.config import CV_FOLDS, DECISION_THRESHOLD, RANDOM_STATE
from churn.features import build_preprocessor


def candidate_models() -> dict:
    """name -> estimator. Add new models here."""
    return {
        "logistic_regression": LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, max_depth=4, random_state=RANDOM_STATE
        ),
    }


def build_pipeline(estimator) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])


def cv_pr_auc(pipeline: Pipeline, X, y, folds: int = CV_FOLDS):
    """Mean and std of PR-AUC over stratified k-fold CV on the training set.
    Used for model selection so the test set stays untouched."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="average_precision")
    return scores.mean(), scores.std()


def evaluate(pipeline: Pipeline, X, y, threshold: float = DECISION_THRESHOLD) -> dict:
    proba = pipeline.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "pr_auc": average_precision_score(y, proba),
        "roc_auc": roc_auc_score(y, proba),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
    }
