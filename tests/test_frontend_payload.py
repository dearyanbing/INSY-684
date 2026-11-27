import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontend"))

from churn.config import FEATURE_COLS  # noqa: E402
from payload import OPTIONS, build_payload  # noqa: E402


@pytest.fixture
def form():
    values = {key: options[0] for key, options in OPTIONS.items()}
    values.update(SeniorCitizen=False, tenure=2, MonthlyCharges=95.5, TotalCharges=190.0)
    return values


def test_payload_has_every_model_feature(form):
    assert set(build_payload(form)) == set(FEATURE_COLS)


def test_payload_types(form):
    payload = build_payload(form)
    assert payload["SeniorCitizen"] == 0
    assert isinstance(payload["tenure"], int)
    assert isinstance(payload["MonthlyCharges"], float)


def test_blank_total_charges_becomes_zero(form):
    form["TotalCharges"] = ""
    assert build_payload(form)["TotalCharges"] == 0.0


def test_rejects_unknown_option(form):
    form["Contract"] = "Weekly"
    with pytest.raises(ValueError, match="Contract"):
        build_payload(form)


def test_rejects_negative_charges(form):
    form["MonthlyCharges"] = -5
    with pytest.raises(ValueError, match="negative"):
        build_payload(form)
