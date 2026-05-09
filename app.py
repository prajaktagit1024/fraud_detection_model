import streamlit as st
import pandas as pd
import pickle
import numpy as np


# --- 1. Load the Model and Encoder ---
@st.cache_resource  # This keeps the model in memory so it doesn't reload every click
def load_assets():
    model = pickle.load(open("fraud_model.pkl", "rb"))
    encoder = pickle.load(open("label_encoder.pkl", "rb"))
    return model, encoder


try:
    model, le = load_assets()
except FileNotFoundError:
    st.error("Model files not found! Please run your training script first.")
    st.stop()

# --- 2. Page Configuration ---
st.set_page_config(page_title="FraudGuard AI", page_icon="🛡️")

st.title("🛡️ FraudGuard Detection System")
st.markdown("""
Predict whether a financial transaction is **Legitimate** or **Fraudulent** using Machine Learning.
---
""")

# --- 3. Sidebar Inputs ---
st.sidebar.header("Transaction Details")

# Transaction Type (Matches your LabelEncoder classes)
tx_type = st.sidebar.selectbox("Transaction Type", le.classes_)

# Numerical Inputs
amount = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, value=1000.0)
oldbalanceOrg = st.sidebar.number_input("Origin Account: Initial Balance", min_value=0.0, value=5000.0)
newbalanceOrig = st.sidebar.number_input("Origin Account: New Balance", min_value=0.0, value=4000.0)
oldbalanceDest = st.sidebar.number_input("Destination: Initial Balance", min_value=0.0, value=0.0)
newbalanceDest = st.sidebar.number_input("Destination: New Balance", min_value=0.0, value=1000.0)

# Optional flags (Common in the Paysim/Kaggle dataset)
is_flagged = st.sidebar.toggle("System Flagged (isFlaggedFraud)", value=False)

# --- 4. Prediction Logic ---
if st.button("Analyze Transaction"):
    # Pre-process inputs
    type_encoded = le.transform([tx_type])[0]

    # Create feature array matching the training columns:
    # [step, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, isFlaggedFraud]
    # Note: Using 1 for 'step' as a placeholder
    features = np.array([[1, type_encoded, amount, oldbalanceOrg,
                          newbalanceOrig, oldbalanceDest, newbalanceDest, int(is_flagged)]])

    # Predict
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]  # Probability of fraud

    # --- 5. Display Results ---
    st.subheader("Analysis Results")

    if prediction[0] == 1:
        st.error(f"⚠️ **ALERT: High Risk of Fraud Detected!**")
        st.metric("Fraud Probability", f"{probability * 100:.2f}%")
        st.warning("This transaction matches patterns of known fraudulent activity.")
    else:
        st.success(f"✅ **Transaction Appears Safe**")
        st.metric("Fraud Probability", f"{probability * 100:.2f}%")
        st.info("No suspicious indicators were found for this transaction.")

# --- Footer ---
st.divider()
st.caption("Internal Security Tool • Powered by Random Forest Classifier")