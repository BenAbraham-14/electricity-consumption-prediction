import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load('electricity_bill_model.pkl')

st.title("⚡ Household Electricity Bill Estimator")

# --- Model Evaluation Metrics Display ---
st.sidebar.header("📊 Model Performance Metrics")
st.sidebar.metric(label="R² Score (Accuracy)", value="0.9999")
st.sidebar.metric(label="Mean Absolute Error (MAE)", value="₹1.32")
st.sidebar.metric(label="Root Mean Squared Error (RMSE)", value="₹5.09")
st.sidebar.caption("Evaluated on 20% unseen test data using Random Forest Regressor.")

# --- User Input Form / Controls ---
st.subheader("Enter Household Appliance Usage")

fan = st.number_input("Number of Fans", min_value=0, max_value=20, value=4)
refrigerator = st.number_input("Refrigerator Hours/Day", min_value=0.0, max_value=24.0, value=24.0)
air_conditioner = st.number_input("Air Conditioners Count", min_value=0, max_value=10, value=3)
television = st.number_input("Television Hours/Day", min_value=0.0, max_value=24.0, value=6.0)
monitor = st.number_input("Monitors Count", min_value=0, max_value=10, value=0)
tariff_rate = st.slider("Tariff Rate (₹/kWh)", min_value=1.0, max_value=20.0, value=8.60)

month_mapping = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}
selected_month = st.selectbox("Billing Month", list(month_mapping.keys()), index=7)
month = month_mapping[selected_month]

# Monthly hours calculation (approx 30 days)
monthly_hours = 720

# Prediction
if st.button("Predict Bill"):
    input_data = pd.DataFrame([[
        fan, refrigerator, air_conditioner, television, 
        monitor, month, monthly_hours, tariff_rate
    ]], columns=['Fan', 'Refrigerator', 'AirConditioner', 'Television', 'Monitor', 'Month', 'MonthlyHours', 'TariffRate'])
    
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Electricity Bill for {selected_month}: ₹{prediction:,.2f}")