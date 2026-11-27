import pytest
from prometheus_client import REGISTRY

from churn.metrics import decision, record_prediction


def _value(name, labels=None):
    return REGISTRY.get_sample_value(name, labels or {}) or 0.0


def test_decision_uses_threshold():
    assert decision(0.45) == "offer"
    assert decision(0.9) == "offer"
    assert decision(0.44) == "no_offer"


def test_record_prediction_updates_counters():
    offers_before = _value("churn_predictions_total", {"decision": "offer"})
    scores_before = _value("churn_prediction_score_count")

    assert record_prediction(0.8, latency_seconds=0.01) == "offer"

    assert _value("churn_predictions_total", {"decision": "offer"}) == offers_before + 1
    assert _value("churn_prediction_score_count") == scores_before + 1


def test_invalid_score_raises():
    with pytest.raises(ValueError):
        record_prediction(1.5, latency_seconds=0.01)
