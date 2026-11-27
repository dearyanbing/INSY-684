"""Drift report: no alarm on similar data, alarm on clearly shifted data."""
import numpy as np
import pandas as pd
import pytest

from churn.data import clean
from churn.drift_report import run_drift


@pytest.fixture
def reference(raw_df):
    return clean(raw_df)


def _resample(df, n=200, seed=1):
    return df.sample(n, replace=True, random_state=seed).reset_index(drop=True)


def test_similar_data_no_drift(reference):
    summary = run_drift(reference, _resample(reference))
    assert summary["drift_detected"] is False
    assert summary["share_drifted"] < 0.3


def test_price_hike_flags_monthly_charges(reference):
    current = _resample(reference)
    current["MonthlyCharges"] = current["MonthlyCharges"] * 1.5
    summary = run_drift(reference, current)
    assert "MonthlyCharges" in summary["drifted_columns"]


def test_new_customer_wave_detected(reference):
    """Marketing campaign: short tenure, month-to-month, fiber, e-check."""
    current = _resample(reference)
    rng = np.random.default_rng(0)
    current["tenure"] = rng.integers(0, 6, len(current))
    current["TotalCharges"] = current["tenure"] * current["MonthlyCharges"]
    current["Contract"] = "Month-to-month"
    current["InternetService"] = "Fiber optic"
    current["PaymentMethod"] = "Electronic check"
    current["PaperlessBilling"] = "Yes"
    current["MonthlyCharges"] = current["MonthlyCharges"] + 30
    summary = run_drift(reference, current)
    assert summary["drift_detected"] is True
    assert {"tenure", "Contract", "InternetService"} <= set(summary["drifted_columns"])


def test_too_few_rows_skips_check(reference):
    summary = run_drift(reference, reference.head(5))
    assert summary["drift_detected"] is False
    assert "note" in summary


def test_html_report_saved(reference, tmp_path):
    out = tmp_path / "drift.html"
    run_drift(reference, _resample(reference), out_html=out)
    assert out.exists() and out.stat().st_size > 0


def test_blank_total_charges_handled(reference):
    current = _resample(reference)
    current["TotalCharges"] = current["TotalCharges"].astype(object)
    current.loc[:4, "TotalCharges"] = " "
    run_drift(reference, pd.DataFrame(current))  # should not raise
