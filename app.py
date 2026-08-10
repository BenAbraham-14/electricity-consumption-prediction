# Create app.py content
app_code = """import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained pipeline
model = joblib.load('electricity_bill_model.pkl')

st.title("⚡ Household Electricity Bill Estimator")
st.write("Enter appliance counts and location details to predict your monthly bill.")

# Inputs
fan = st.number_input("Number of Fans", min_value=0, max_value=30, value=10)
fridge = st.number_input("Refrigerator Hours/Day", min_value=0.0, max_value=24.0, value=20.0)
ac = st.number_input("Air Conditioners Count", min_value=0, max_value=10, value=2)
tv = st.number_input("Television Hours/Day", min_value=0.0, max_value=24.0, value=5.0)
monitor = st.number_input("Monitors Count", min_value=0, max_value=10, value=1)
tariff = st.slider("Tariff Rate (₹/kWh)", min_value=7.0, max_value=10.0, value=8.5, step=0.1)
month = st.selectbox("Month", list(range(1, 13)))

if st.button("Predict Bill"):
    monthly_hours = (fan*6 + fridge*15 + ac*8 + tv*5 + monitor*4)
    input_data = pd.DataFrame([{
        'Fan': fan,
        'Refrigerator': fridge,
        'AirConditioner': ac,
        'Television': tv,
        'Monitor': monitor,
        'Month': month,
        'MonthlyHours': monthly_hours,
        'TariffRate': tariff
    }])
    
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Monthly Electricity Bill: **₹{prediction:,.2f}**")
"""

with open("app.py", "w") as f:
    f.write(app_code)

# Create requirements.txt content
req_code = """streamlit
pandas
numpy
scikit-learn
joblib
"""

with open("requirements.txt", "w") as f:
    f.write(req_code)

print("app.py and requirements.txt written successfully.")