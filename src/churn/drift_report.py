"""Evidently data-drift report: are the customers we score now different from the
customers the model was trained on?

    reference = training set (what the model learned from)
    current   = inputs logged in the `predictions` table (what the API sees now)

Evidently runs a statistical test per column (K-S for numbers, chi-square/Z-test for
categories; distance metrics on large samples) and flags the columns that changed.
If too many columns drift, `drift_detected` is True, which is the signal for the
Prefect flow to retrain.

Usage:
    DATABASE_URL=postgresql://churn:churn@localhost:5433/churn \
        python -m churn.drift_report --since 2026-12-01
    python -m churn.drift_report --current-csv some_new_customers.csv
    add --fail-on-drift to exit with code 1 when drift is detected
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from churn.config import FEATURE_COLS, PROJECT_ROOT
from churn.data import clean, load_raw, split

# Retrain if at least this share of input columns drifted (Evidently's default is 0.5;
# we are stricter because a few key columns like Contract or tenure matter a lot)
DRIFT_SHARE = 0.3
MIN_ROWS = 30  # below this the statistical tests are not meaningful
REPORTS_DIR = PROJECT_ROOT / "reports"


def reference_data() -> pd.DataFrame:
    """The training split, exactly as the model saw it."""
    X_train, _, _, _ = split(clean(load_raw()))
    return X_train


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df[FEATURE_COLS].copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce").fillna(0.0)
    return out.reset_index(drop=True)


def _is_drifted(value: float, method: str, threshold: float) -> bool:
    # p-value tests: drift when p < threshold; distance metrics: drift when distance > threshold
    return value < threshold if "p_value" in method else value > threshold


def run_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    out_html: str | Path | None = None,
    drift_share: float = DRIFT_SHARE,
) -> dict:
    """Compare current vs reference. Returns a small summary; optionally saves the
    full Evidently HTML report."""
    summary = {
        "n_reference": len(reference),
        "n_current": len(current),
        "drifted_columns": [],
        "share_drifted": 0.0,
        "drift_detected": False,
    }
    if len(current) < MIN_ROWS:
        summary["note"] = f"only {len(current)} rows; need at least {MIN_ROWS}"
        return summary

    from evidently import Report  # heavy import, only when needed
    from evidently.presets import DataDriftPreset

    snapshot = Report([DataDriftPreset()]).run(
        current_data=_prepare(current), reference_data=_prepare(reference)
    )
    if out_html:
        Path(out_html).parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(out_html))
        summary["report"] = str(out_html)

    for m in snapshot.dict()["metrics"]:
        cfg = m.get("config", {})
        if m["metric_name"].startswith("ValueDrift") and "column" in cfg:
            if _is_drifted(m["value"], cfg.get("method", ""), cfg["threshold"]):
                summary["drifted_columns"].append(cfg["column"])

    summary["share_drifted"] = round(len(summary["drifted_columns"]) / len(FEATURE_COLS), 3)
    summary["drift_detected"] = summary["share_drifted"] >= drift_share
    return summary


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--since", help="only predictions logged on/after this date (DB mode)")
    src.add_argument("--current-csv", help="compare a CSV of customers instead of the DB")
    ap.add_argument("--model-version", help="only predictions from this model version")
    ap.add_argument("--out", default=str(REPORTS_DIR / "drift_report.html"))
    ap.add_argument("--fail-on-drift", action="store_true", help="exit 1 if drift detected")
    args = ap.parse_args(argv)

    if args.current_csv:
        current = pd.read_csv(args.current_csv)
    else:
        from churn.prediction_log import get_engine, load_logged_inputs

        engine = get_engine()
        if engine is None:
            ap.error("set DATABASE_URL or use --current-csv")
        current = load_logged_inputs(engine, since=args.since, model_version=args.model_version)

    summary = run_drift(reference_data(), current, out_html=args.out)
    print(json.dumps(summary, indent=2))
    if args.fail_on_drift and summary["drift_detected"]:
        sys.exit(1)
    return summary


if __name__ == "__main__":
    main()
