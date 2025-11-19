# app.py — Streamlit UI for CLTV API (updated to show server-side segment)
import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="CLTV Demo", layout="centered")
st.title("CLTV Prediction — Demo")
st.markdown("This UI calls your deployed CLTV API and shows predictions (server-side segmentation).")

# ---- CONFIG ----
DEFAULT_API = st.secrets.get("API_BASE_URL", "http://127.0.0.1:8000")
api_base = st.text_input("API base URL", value=DEFAULT_API)
predict_url = api_base.rstrip("/") + "/predict"

st.header("Single customer prediction")
with st.form("single_form"):
    cust_id = st.text_input("Customer ID", value="C101")
    frequency = st.number_input("Frequency", value=4, min_value=0)
    total_spend = st.number_input("Total spend", value=200.0, min_value=0.0, format="%.2f")
    aov = st.number_input("AOV (avg order value)", value=50.0, min_value=0.0, format="%.2f")
    recency_days = st.number_input("Recency days", value=20, min_value=0)
    T_days = st.number_input("T (customer age days)", value=300, min_value=0)
    avg_interpurchase_days = st.number_input("Avg interpurchase days", value=100, min_value=0)
    active_months = st.number_input("Active months", value=3, min_value=0)
    purchase_days_std = st.number_input("Purchase days std", value=12.0, min_value=0.0, format="%.2f")
    category_diversity = st.number_input("Category diversity", value=2, min_value=0)
    avg_order_value = st.number_input("avg_order_value", value=float(aov), min_value=0.0, format="%.2f")
    unique_days = st.number_input("unique_days", value=max(1, int(frequency)), min_value=0)
    submitted = st.form_submit_button("Get prediction")

if submitted:
    payload = {"customers": [{
        "customer_id": str(cust_id),
        "frequency": frequency,
        "total_spend": total_spend,
        "aov": aov,
        "recency_days": recency_days,
        "T_days": T_days,
        "avg_interpurchase_days": avg_interpurchase_days,
        "active_months": active_months,
        "purchase_days_std": purchase_days_std,
        "category_diversity": category_diversity,
        "avg_order_value": avg_order_value,
        "unique_days": unique_days
    }]}
    st.write("Request payload:")
    st.json(payload)

    try:
        resp = requests.post(predict_url, json=payload, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            st.success("Prediction received")
            st.json(data)
            predicted = data[0].get("predicted_LTV")
            segment = data[0].get("segment", "Unknown")
            st.metric("Predicted LTV", f"{predicted:.2f}")
            st.info(f"Server-side Segment: **{segment}**")
        else:
            st.error(f"API error {resp.status_code}: {resp.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")

st.markdown("---")
st.header("Batch predictions (CSV)")
st.write("Upload CSV with required columns (customer_id, frequency, total_spend, aov, ...). avg_order_value and unique_days optional.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Preview (first 5 rows):")
    st.dataframe(df.head())
    if st.button("Predict batch"):
        payload = {"customers": df.to_dict(orient="records")}
        try:
            resp = requests.post(predict_url, json=payload, timeout=120)
            if resp.status_code == 200:
                out = pd.DataFrame(resp.json())
                st.success("Batch predictions complete")
                st.dataframe(out.head(200))
                csv = out.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions CSV", csv, "predictions.csv")
            else:
                st.error(f"API error {resp.status_code}: {resp.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")
