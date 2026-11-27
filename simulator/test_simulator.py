import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulator"))

from churn.config import FEATURE_COLS  # noqa: E402
from drift import apply_drift, to_requests  # noqa: E402
from simulate import build_traffic  # noqa: E402


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_requests_have_exactly_the_model_features(raw_df):
    body = to_requests(raw_df)[0]
    assert set(body) == set(FEATURE_COLS)
    assert isinstance(body["TotalCharges"], float)


def test_price_hike_raises_charges(raw_df, rng):
    drifted = apply_drift(raw_df, "price_hike", rng)
    assert drifted["MonthlyCharges"].mean() > raw_df["MonthlyCharges"].mean() * 1.2


def test_new_customers_are_short_tenure_month_to_month(raw_df, rng):
    drifted = apply_drift(raw_df, "new_customers", rng)
    assert drifted["tenure"].max() < 6
    assert (drifted["Contract"] == "Month-to-month").all()
    assert to_requests(drifted)  # still valid API requests


def test_fiber_shift_keeps_addons_consistent(raw_df, rng):
    drifted = apply_drift(raw_df, "fiber_shift", rng)
    fiber = drifted[drifted["InternetService"] == "Fiber optic"]
    assert (fiber["OnlineSecurity"] != "No internet service").all()
    assert len(fiber) > len(raw_df) * 0.6


def test_drift_does_not_change_input(raw_df, rng):
    before = raw_df.copy()
    apply_drift(raw_df, "price_hike", rng)
    assert raw_df.equals(before)


def test_unknown_scenario_rejected(raw_df, rng):
    with pytest.raises(ValueError, match="Unknown scenario"):
        apply_drift(raw_df, "aliens", rng)


def test_traffic_switches_to_drift_after_cutoff(raw_df, rng):
    traffic = build_traffic(raw_df, n=50, drift="price_hike", drift_after=30, rng=rng)
    phases = [phase for phase, _ in traffic]
    assert phases[:30] == ["normal"] * 30
    assert phases[30:] == ["price_hike"] * 20
