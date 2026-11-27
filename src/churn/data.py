"""Loading, cleaning and splitting the Telco churn data."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churn.config import (
    FEATURE_COLS,
    ID_COL,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_COL,
    TEST_SIZE,
)


def load_raw(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix types and encode the target. Returns a new DataFrame."""
    missing = set(FEATURE_COLS + [TARGET_COL]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    out = df.copy()

    # TotalCharges is read as text because new customers (tenure 0) have a blank value.
    # They have not been billed yet, so 0 is the correct value, not the median.
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce").fillna(0.0)

    out[TARGET_COL] = out[TARGET_COL].map({"Yes": 1, "No": 0})
    if out[TARGET_COL].isna().any():
        raise ValueError("Churn column should only contain 'Yes' or 'No'")
    out[TARGET_COL] = out[TARGET_COL].astype(int)

    if ID_COL in out.columns:
        out = out.drop(columns=[ID_COL])
    return out


def split(df: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """Stratified train/test split so both sets keep the same churn rate."""
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
