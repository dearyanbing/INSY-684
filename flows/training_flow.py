"""Prefect flow that runs the training pipeline end to end.

    download data -> load + clean -> validate -> train, log to MLflow, register @champion

Usage:
    python flows/training_flow.py          # run once, locally
    prefect server start                   # optional: see runs at http://127.0.0.1:4200

Still to do (by Dec 4):
    - validate_data: real checks (schema, ranges, missing values, churn rate)
    - retrain_on_drift: a second flow that retrains when the Evidently drift share
      passes a threshold (needs feature/monitoring)
"""
import sys
from pathlib import Path

import pandas as pd
from prefect import flow, get_run_logger, task

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churn import train  # noqa: E402
from churn.config import RAW_DATA_PATH  # noqa: E402
from churn.data import clean, load_raw  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import download_data  # noqa: E402


@task(retries=2, retry_delay_seconds=10)
def get_data() -> Path:
    """Download the CSV if we don't have it yet (retries cover network errors)."""
    download_data.main()
    return RAW_DATA_PATH


@task
def load_and_clean(path: Path) -> pd.DataFrame:
    return clean(load_raw(path))


@task
def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Placeholder: basic sanity checks. Full validation comes in the next PR."""
    logger = get_run_logger()
    if df.empty:
        raise ValueError("Dataset is empty")
    logger.info("%d rows, churn rate %.1f%%", len(df), 100 * df["Churn"].mean())
    # TODO: schema / range / missing-value checks
    return df


@task
def train_and_register() -> None:
    """Trains the candidate models, logs them to MLflow and sets @champion.
    Reuses churn.train so the flow and the plain script always do the same thing."""
    train.main()


@flow(name="telco-churn-training")
def training_flow() -> None:
    path = get_data()
    df = load_and_clean(path)
    validate_data(df)
    train_and_register()


if __name__ == "__main__":
    training_flow()
