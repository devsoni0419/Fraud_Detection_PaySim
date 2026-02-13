import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="PaySim Fraud Detection", layout="centered")

bundle = joblib.load("models/model.pkl")
model = bundle["model"]
preprocessor = bundle["preprocessor"]

st.title("💳 Fraud Detection System (PaySim)")
st.write("Mobile money fraud detection using ML")

amount = st.number_input("Transaction Amount", min_value=0.0)
tx_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT"])
old_org = st.number_input("Old Balance (Sender)", min_value=0.0)
new_org = st.number_input("New Balance (Sender)", min_value=0.0)
old_dest = st.number_input("Old Balance (Receiver)", min_value=0.0)
new_dest = st.number_input("New Balance (Receiver)", min_value=0.0)

threshold = st.slider("Fraud Threshold", 0.05, 0.9, 0.2)

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "step": 1,
        "type": tx_type,
        "amount": amount,
        "oldbalanceOrg": old_org,
        "newbalanceOrig": new_org,
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest
    }])

    X = preprocessor.transform(input_df)
    prob = model.predict_proba(X)[0][1]
    pred = 1 if prob >= threshold else 0

    if pred == 1:
        st.error(f"🚨 Fraud Detected (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Transaction Safe (Fraud Probability: {prob:.2f})")
