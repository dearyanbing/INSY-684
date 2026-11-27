"""Form options and request building for the frontend.

Kept separate from app.py (no Streamlit import) so it can be unit tested in CI.
"""

YES_NO = ["Yes", "No"]
INTERNET_ADDON = ["No", "Yes", "No internet service"]

OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["Fiber optic", "DSL", "No"],
    "OnlineSecurity": INTERNET_ADDON,
    "OnlineBackup": INTERNET_ADDON,
    "DeviceProtection": INTERNET_ADDON,
    "TechSupport": INTERNET_ADDON,
    "StreamingTV": INTERNET_ADDON,
    "StreamingMovies": INTERNET_ADDON,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": YES_NO,
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}


def build_payload(form: dict) -> dict:
    """Turn raw form values into the JSON body the API's /predict expects."""
    payload = {key: form[key] for key in OPTIONS}
    for key, value in payload.items():
        if value not in OPTIONS[key]:
            raise ValueError(f"{key}={value!r} is not one of {OPTIONS[key]}")

    payload["SeniorCitizen"] = int(bool(form["SeniorCitizen"]))
    payload["tenure"] = int(form["tenure"])
    payload["MonthlyCharges"] = float(form["MonthlyCharges"])
    total = form.get("TotalCharges")
    # Same rule as the training data: new customers have not been billed yet
    payload["TotalCharges"] = 0.0 if total in (None, "") else float(total)

    if payload["tenure"] < 0 or payload["MonthlyCharges"] < 0 or payload["TotalCharges"] < 0:
        raise ValueError("tenure and charges cannot be negative")
    return payload
