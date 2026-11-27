# Monitoring: prediction log + drift report

Owner: Fatima (`feature/monitoring`)

```
API /predict ──► predictions table (Postgres) ──► drift_report.py (Evidently) ──► HTML report
                                                                              └─► drift_detected → Prefect retrain
```

| File | What it does |
|---|---|
| `monitoring/schema.sql` | Creates the `predictions` table: model inputs, churn probability, `send_offer`, model version, time, and `actual_churn` (filled later) |
| `src/churn/prediction_log.py` | `log_predictions()` saves one row per prediction; `load_logged_inputs()` reads them back |
| `src/churn/drift_report.py` | Compares logged inputs with the training set using Evidently; saves an HTML report and returns `drift_detected` |
| `tests/test_prediction_log.py`, `tests/test_drift_report.py` | Unit tests (SQLite in a temp folder, so CI needs no Postgres) |

## Plugging into the API (when `src/churn/api.py` lands)

```python
from churn.prediction_log import get_engine, log_predictions

engine = get_engine()   # reads DATABASE_URL; None = logging off

# inside /predict and /predict/batch, after computing `scores` for DataFrame `df`:
log_predictions(engine, df, scores, model_version=MODEL_URI, endpoint="/predict")
```

If the database is down, the API still answers; the failure is only logged as a warning.

## Postgres in docker-compose

Already set up in `docker-compose.yml` (Esteban): the `postgres` service runs `schema.sql`
on first start, and the `api` service gets `DATABASE_URL=postgresql://churn:churn@postgres:5432/churn`.
From your laptop the database is at `localhost:5433`. After changing `schema.sql`, reset with
`docker compose down -v`.

## Running the drift report

```bash
# against logged predictions
DATABASE_URL=postgresql://churn:churn@localhost:5433/churn \
  python -m churn.drift_report --since 2026-12-01

# against any CSV of customers (e.g. the traffic simulator's output)
python -m churn.drift_report --current-csv new_customers.csv

# for the Prefect retrain flow: exit code 1 when drift is detected
python -m churn.drift_report --since 2026-12-01 --fail-on-drift
```

Output is a JSON summary plus `reports/drift_report.html`.

**Retrain rule:** `drift_detected` is True when at least 30% of the 19 input columns drift
(`DRIFT_SHARE` in `drift_report.py`). Below 30 logged rows the check is skipped.

**Checked on the real data:** a normal month of customers (test set) → no columns drift.
A 20% price increase → `MonthlyCharges` flagged (1 of 19 columns, so a warning, not a retrain).
