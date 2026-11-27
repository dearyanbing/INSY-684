"""Traffic simulator: send customers from the Telco data to the model API.

Normal traffic is sampled from the real data. After --drift-after requests the
chosen drift scenario kicks in, so the Evidently report and Grafana dashboard
should show the change.

Examples:
    python simulator/simulate.py --n 200                         # normal traffic only
    python simulator/simulate.py --n 400 --drift price_hike --drift-after 200
    python simulator/simulate.py --n 20 --dry-run                # print, don't send
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drift import SCENARIOS, apply_drift, to_requests  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "Telco-Customer-Churn.csv"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--api-url", default=os.getenv("API_URL", "http://localhost:8000"))
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--n", type=int, default=200, help="total requests to send")
    p.add_argument("--rate", type=float, default=5.0, help="requests per second")
    p.add_argument("--drift", choices=sorted(SCENARIOS), help="drift scenario to inject")
    p.add_argument("--drift-after", type=int, default=0, help="normal requests before drift starts")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="print requests instead of sending")
    return p.parse_args()


def build_traffic(df: pd.DataFrame, n: int, drift: str | None, drift_after: int,
                  rng: np.random.Generator) -> list[tuple[str, dict]]:
    """List of (phase, request body). Phase is 'normal' or the drift scenario name."""
    sample = df.sample(n=n, replace=True, random_state=int(rng.integers(1_000_000)))
    sample = sample.reset_index(drop=True)
    if not drift:
        return [("normal", r) for r in to_requests(sample)]

    cut = min(drift_after, n)
    normal = to_requests(sample.iloc[:cut])
    drifted = to_requests(apply_drift(sample.iloc[cut:], drift, rng))
    return [("normal", r) for r in normal] + [(drift, r) for r in drifted]


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(args.data)
    traffic = build_traffic(df, args.n, args.drift, args.drift_after, rng)

    if args.dry_run:
        for phase, body in traffic:
            print(phase, json.dumps(body))
        return

    import requests  # only needed when actually sending

    sent = failed = offers = 0
    for i, (phase, body) in enumerate(traffic, 1):
        try:
            resp = requests.post(f"{args.api_url}/predict", json=body, timeout=10)
            resp.raise_for_status()
            offers += bool(resp.json().get("send_offer"))
            sent += 1
        except requests.RequestException as err:
            failed += 1
            print(f"[{i}] {phase}: request failed: {err}", file=sys.stderr)
        if i % 50 == 0:
            print(f"{i}/{len(traffic)} sent ({phase}), offers so far: {offers}")
        time.sleep(1 / args.rate)

    print(f"Done: {sent} ok, {failed} failed, {offers} offers")


if __name__ == "__main__":
    main()
