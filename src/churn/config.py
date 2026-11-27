"""Project-wide settings: paths, column lists and MLflow names."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "Telco-Customer-Churn.csv"

# Public copy of the IBM Telco dataset, so nobody needs a Kaggle account
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

ID_COL = "customerID"
TARGET_COL = "Churn"

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# From the course 1 project: send a retention offer if score >= 0.45
# (threshold chosen by minimizing cost on out-of-fold scores)
DECISION_THRESHOLD = 0.45

MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
MLFLOW_EXPERIMENT = "telco-churn"
REGISTERED_MODEL_NAME = "telco-churn-classifier"
