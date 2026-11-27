import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_df():
    """Small fake dataset with the same columns as the real Telco CSV."""
    rng = np.random.default_rng(0)
    n = 200
    yes_no = ["Yes", "No"]
    internet_opts = ["Yes", "No", "No internet service"]
    tenure = rng.integers(0, 72, n)
    monthly = rng.uniform(18, 120, n).round(2)
    total = (tenure * monthly).round(2).astype(str)
    total[tenure == 0] = " "  # same quirk as the real file

    return pd.DataFrame(
        {
            "customerID": [f"{i:04d}-ABCDE" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], n),
            "SeniorCitizen": rng.integers(0, 2, n),
            "Partner": rng.choice(yes_no, n),
            "Dependents": rng.choice(yes_no, n),
            "tenure": tenure,
            "PhoneService": rng.choice(yes_no, n),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
            "OnlineSecurity": rng.choice(internet_opts, n),
            "OnlineBackup": rng.choice(internet_opts, n),
            "DeviceProtection": rng.choice(internet_opts, n),
            "TechSupport": rng.choice(internet_opts, n),
            "StreamingTV": rng.choice(internet_opts, n),
            "StreamingMovies": rng.choice(internet_opts, n),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "PaperlessBilling": rng.choice(yes_no, n),
            "PaymentMethod": rng.choice(
                ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                 "Credit card (automatic)"], n
            ),
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Churn": rng.choice(yes_no, n, p=[0.27, 0.73]),
        }
    )
