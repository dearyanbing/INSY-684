"""Streamlit front end for the churn prediction API.

Collects one customer's details, sends them to the API's /predict endpoint
and shows the churn probability and whether the retention offer applies.
"""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://api:8000").rstrip("/")
DECISION_THRESHOLD = 0.45  # same value as src/churn/config.py

YES_NO = ["Yes", "No"]

st.set_page_config(page_title="Telco churn predictor", page_icon="📉")
st.title("Telco customer churn predictor")
st.write("Enter a customer's details to estimate how likely they are to cancel.")

with st.form("customer_form"):
    st.subheader("Customer")
    col1, col2, col3 = st.columns(3)
    gender = col1.selectbox("Gender", ["Female", "Male"])
    senior = col2.selectbox("Senior citizen", ["No", "Yes"])
    partner = col3.selectbox("Partner", YES_NO, index=1)
    dependents = col1.selectbox("Dependents", YES_NO, index=1)
    tenure = col2.number_input("Tenure (months)", min_value=0, max_value=100, value=12)

    st.subheader("Services")
    col4, col5, col6 = st.columns(3)
    phone = col4.selectbox("Phone service", YES_NO)
    multiple_lines = col5.selectbox(
        "Multiple lines", ["No", "Yes", "No phone service"]
    )
    internet = col6.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    online_security = col4.selectbox(
        "Online security", ["No", "Yes", "No internet service"]
    )
    online_backup = col5.selectbox(
        "Online backup", ["No", "Yes", "No internet service"]
    )
    device_protection = col6.selectbox(
        "Device protection", ["No", "Yes", "No internet service"]
    )
    tech_support = col4.selectbox("Tech support", ["No", "Yes", "No internet service"])
    streaming_tv = col5.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = col6.selectbox(
        "Streaming movies", ["No", "Yes", "No internet service"]
    )

    st.subheader("Billing")
    col7, col8, col9 = st.columns(3)
    contract = col7.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = col8.selectbox("Paperless billing", YES_NO)
    payment = col9.selectbox(
        "Payment method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    monthly = col7.number_input(
        "Monthly charges ($)", min_value=0.0, max_value=500.0, value=70.0, step=0.5
    )
    total = col8.number_input(
        "Total charges ($)", min_value=0.0, max_value=20000.0, value=840.0, step=10.0
    )

    submitted = st.form_submit_button("Predict churn")

if submitted:
    payload = {
        "tenure": int(tenure),
        "MonthlyCharges": float(monthly),
        "TotalCharges": float(total),
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "gender": gender,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        # Assumed response shape: {"churn_probability": float}. Update if the API differs.
        probability = float(response.json()["churn_probability"])
    except requests.RequestException as exc:
        st.error(f"Could not reach the prediction API at {API_URL}: {exc}")
    except (KeyError, ValueError) as exc:
        st.error(f"Unexpected response from the prediction API: {exc}")
    else:
        st.metric("Churn probability", f"{probability:.1%}")
        if probability >= DECISION_THRESHOLD:
            st.warning("High risk: send this customer a retention offer.")
        else:
            st.success("Low risk: no retention offer needed.")
