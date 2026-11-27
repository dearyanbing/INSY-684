"""Streamlit frontend: enter a customer, get a churn score from the model API.

Run locally:
    pip install -r frontend/requirements.txt
    API_URL=http://localhost:8000 streamlit run frontend/app.py
"""
import os

import requests
import streamlit as st

from payload import OPTIONS, build_payload

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Telco Churn", page_icon="📉")
st.title("Telco customer churn")
st.caption("Score a customer and decide whether to send a retention offer.")

with st.sidebar:
    st.subheader("Model API")
    st.code(API_URL)
    try:
        ok = requests.get(f"{API_URL}/health", timeout=3).ok
    except requests.RequestException:
        ok = False
    st.success("API is up") if ok else st.error("API not reachable")

with st.form("customer"):
    st.subheader("Account")
    c1, c2, c3 = st.columns(3)
    tenure = c1.number_input("Tenure (months)", 0, 120, 2)
    monthly = c2.number_input("Monthly charges ($)", 0.0, 500.0, 95.5)
    total = c3.number_input("Total charges ($)", 0.0, 20000.0, 190.0)
    contract = c1.selectbox("Contract", OPTIONS["Contract"])
    payment = c2.selectbox("Payment method", OPTIONS["PaymentMethod"])
    paperless = c3.selectbox("Paperless billing", OPTIONS["PaperlessBilling"])

    st.subheader("Customer")
    c1, c2, c3, c4 = st.columns(4)
    gender = c1.selectbox("Gender", OPTIONS["gender"])
    senior = c2.checkbox("Senior citizen")
    partner = c3.selectbox("Partner", OPTIONS["Partner"])
    dependents = c4.selectbox("Dependents", OPTIONS["Dependents"])

    st.subheader("Services")
    c1, c2, c3 = st.columns(3)
    phone = c1.selectbox("Phone service", OPTIONS["PhoneService"])
    lines = c2.selectbox("Multiple lines", OPTIONS["MultipleLines"])
    internet = c3.selectbox("Internet service", OPTIONS["InternetService"])
    addons = {}
    for i, key in enumerate(
        ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
         "TechSupport", "StreamingTV", "StreamingMovies"]
    ):
        addons[key] = (c1, c2, c3)[i % 3].selectbox(key, OPTIONS[key])

    submitted = st.form_submit_button("Score customer", type="primary")

if submitted:
    form = {
        "gender": gender, "SeniorCitizen": senior, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
        "MultipleLines": lines, "InternetService": internet, "Contract": contract,
        "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total, **addons,
    }
    try:
        resp = requests.post(f"{API_URL}/predict", json=build_payload(form), timeout=10)
        resp.raise_for_status()
        result = resp.json()
    except (requests.RequestException, ValueError) as err:
        st.error(f"Could not score this customer: {err}")
    else:
        prob = result["churn_probability"]
        st.metric("Churn probability", f"{prob:.1%}")
        if result.get("send_offer"):
            st.warning("High risk: send a retention offer.")
        else:
            st.success("Low risk: no offer needed.")
