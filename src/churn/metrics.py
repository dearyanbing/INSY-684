"""Prometheus metrics for the prediction API.

The API (src/churn/api.py) only needs two things:

    from prometheus_client import make_asgi_app
    from churn.metrics import record_prediction

    app.mount("/metrics", make_asgi_app())   # Prometheus scrapes this
    ...
    record_prediction(score, latency_seconds) # call once per prediction

Grafana then reads these from Prometheus (see monitoring/prometheus.yml).
"""
from prometheus_client import Counter, Histogram

from churn.config import DECISION_THRESHOLD

PREDICTIONS = Counter(
    "churn_predictions_total",
    "Number of predictions served, split by decision",
    ["decision"],  # "offer" or "no_offer"
)

SCORE = Histogram(
    "churn_prediction_score",
    "Distribution of predicted churn probability",
    buckets=[0.1, 0.2, 0.3, 0.4, DECISION_THRESHOLD, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

LATENCY = Histogram(
    "churn_prediction_latency_seconds",
    "Time to compute one prediction",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)


def decision(score: float, threshold: float = DECISION_THRESHOLD) -> str:
    return "offer" if score >= threshold else "no_offer"


def record_prediction(score: float, latency_seconds: float) -> str:
    """Update all metrics for one prediction and return the decision."""
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"score must be between 0 and 1, got {score}")
    d = decision(score)
    PREDICTIONS.labels(decision=d).inc()
    SCORE.observe(score)
    LATENCY.observe(latency_seconds)
    return d
