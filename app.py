import streamlit as st
import numpy as np
import pickle

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Fraud Detection System",
    layout="centered"
)

st.title("💳 Transaction Fraud Detection System")

st.write(
    "Enter transaction details to predict whether the transaction is Fraudulent or Legitimate."
)

# -----------------------------------
# LOAD TRAINED MODEL
# -----------------------------------
#model = pickle.load(open("best_fraud_model_tuned.pkl", "rb"))
saved_objects = pickle.load(open("best_fraud_model_tuned.pkl", "rb"))

model = saved_objects["model"]



# -----------------------------------
# USER INPUT SECTION
# -----------------------------------

transaction_type = st.selectbox(
    "Transaction Type",
    ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
)

amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    format="%.2f"
)

oldbalanceOrg = st.number_input(
    "Sender Old Balance",
    min_value=0.0
)

newbalanceOrig = st.number_input(
    "Sender New Balance",
    min_value=0.0
)

oldbalanceDest = st.number_input(
    "Receiver Old Balance",
    min_value=0.0
)

newbalanceDest = st.number_input(
    "Receiver New Balance",
    min_value=0.0
)

hour = st.slider(
    "Transaction Hour (0–23)",
    0, 23
)

# -----------------------------------
# FEATURE ENGINEERING
# -----------------------------------

type_dict = {
    "PAYMENT": 0,
    "TRANSFER": 1,
    "CASH_OUT": 2,
    "DEBIT": 3,
    "CASH_IN": 4
}

type_enc = type_dict[transaction_type]

# engineered features
log_amount = np.log1p(amount)

is_high_amount = 1 if amount > 200000 else 0

is_night = 1 if 0 <= hour <= 6 else 0

balance_diff_orig = oldbalanceOrg - newbalanceOrig
balance_diff_dest = newbalanceDest - oldbalanceDest

# step approximation
step = hour

# -----------------------------------
# MODEL INPUT (ORDER MATTERS)
# -----------------------------------

input_data = np.array([[
    step,
    amount,
    log_amount,
    is_high_amount,
    hour,
    is_night,
    balance_diff_orig,
    balance_diff_dest,
    type_enc
]])

# -----------------------------------
# PREDICTION
# -----------------------------------

if st.button("Check Transaction"):

    probability = model.predict_proba(input_data)[0][1]

    # higher fraud threshold (better for imbalanced data)
    threshold = 0.90
    prediction = 1 if probability > threshold else 0

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Fraudulent Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")

    st.write(f"Fraud Probability: {probability:.2%}")

    st.progress(float(probability))