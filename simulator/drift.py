"""Ways to make simulated traffic drift away from the training data.

Pure pandas, no network calls, so it can be unit tested in CI.
Each scenario takes a batch of customers and returns a changed copy.
"""
import numpy as np
import pandas as pd

DROP_COLS = ["customerID", "Churn"]


def to_requests(df: pd.DataFrame) -> list[dict]:
    """Rows of the raw CSV -> JSON bodies for the API's /predict."""
    out = df.drop(columns=[c for c in DROP_COLS if c in df.columns]).copy()
    # Blank TotalCharges = new customer not billed yet (same rule as training)
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce").fillna(0.0)
    out["SeniorCitizen"] = out["SeniorCitizen"].astype(int)
    out["tenure"] = out["tenure"].astype(int)
    return out.to_dict(orient="records")


def price_hike(df: pd.DataFrame, rng: np.random.Generator, pct: float = 0.30) -> pd.DataFrame:
    """Company raises prices: monthly charges go up by about `pct`."""
    out = df.copy()
    factor = 1 + pct * rng.uniform(0.8, 1.2, len(out))
    out["MonthlyCharges"] = (out["MonthlyCharges"] * factor).round(2)
    return out


def new_customers(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Marketing campaign brings in many new month-to-month customers."""
    out = df.copy()
    out["tenure"] = rng.integers(0, 6, len(out))
    out["Contract"] = "Month-to-month"
    total = out["MonthlyCharges"] * out["tenure"]
    out["TotalCharges"] = total.round(2).astype(str).where(out["tenure"] > 0, " ")
    return out


def fiber_shift(df: pd.DataFrame, rng: np.random.Generator, share: float = 0.8) -> pd.DataFrame:
    """Most customers move to fiber optic (the segment that churns most)."""
    out = df.copy()
    move = rng.random(len(out)) < share
    out.loc[move, "InternetService"] = "Fiber optic"
    # Anyone with internet must have real add-on values, not "No internet service"
    has_internet = out["InternetService"] != "No"
    addons = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]
    for col in addons:
        out.loc[has_internet & (out[col] == "No internet service"), col] = "No"
    return out


SCENARIOS = {
    "price_hike": price_hike,
    "new_customers": new_customers,
    "fiber_shift": fiber_shift,
}


def apply_drift(df: pd.DataFrame, scenario: str, rng: np.random.Generator) -> pd.DataFrame:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}, pick one of {sorted(SCENARIOS)}")
    return SCENARIOS[scenario](df, rng)
