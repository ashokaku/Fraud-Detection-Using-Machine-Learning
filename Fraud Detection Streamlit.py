import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load trained model & scaler
model = joblib.load("xgb_fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit App
st.title("💳 Fraud Detection App")
st.write("Enter transaction details to check if it's Fraud or Not Fraud")

# Input fields
step = st.number_input("Step (Transaction Time in Hours)", min_value=0, max_value=1000, step=1)
type_options = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT","CASH_IN"]
type_choice = st.selectbox("Transaction Type", type_options)

amount = st.number_input("Transaction Amount", min_value=0.0, step=0.01)
oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0, step=0.01)
newbalanceOrig = st.number_input("New Balance (Origin)", min_value=0.0, step=0.01)
oldbalanceDest = st.number_input("Old Balance (Destination)", min_value=0.0, step=0.01)
newbalanceDest = st.number_input("New Balance (Destination)", min_value=0.0, step=0.01)

# Encode 'type' same way as during training
le = LabelEncoder()
le.fit(["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT","CASH_IN"])
type_encoded = le.transform([type_choice])[0]

# Create dataframe with same columns as model training
input_data = pd.DataFrame([[
    step, type_encoded, amount, oldbalanceOrg, newbalanceOrig,
    oldbalanceDest, newbalanceDest
]], columns=["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"])

# Scale data
input_scaled = scaler.transform(input_data)

# Predict
if st.button("🔍 Predict"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected!")
    else:
        st.success(f"✅ Legitimate Transaction ")

# Run Command:
# streamlit run app.py
