import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Smart Electricity Bill Estimator",
    page_icon="⚡",
    layout="wide"
)

# --- Custom UI Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background: linear-gradient(135deg, #1f2937, #111827);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #374151;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
        margin-bottom: 20px;
    }
    .result-card {
        background: linear-gradient(135deg, #064e3b, #022c22);
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #059669;
        text-align: center;
        margin-top: 20px;
    }
    .result-val {
        font-size: 2.2rem;
        font-weight: 700;
        color: #34d399;
    }
    .stButton>button {
        width: 100%;
        background-color: #2563eb;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        height: 3em;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        border-color: #3b82f6;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained machine learning model
@st.cache_resource
def load_model():
    return joblib.load('electricity_bill_model.pkl')

model = load_model()

# --- Sidebar: Model Evaluation Metrics ---
with st.sidebar:
    st.header("📊 Model Metrics")
    st.markdown("Performance benchmarks evaluated on 20% unseen test data:")
    
    st.metric(label="R² Score (Accuracy)", value="0.7482", help="Explains 74.8% of variance under real-world conditions.")
    st.metric(label="Mean Absolute Error (MAE)", value="₹420.50", help="Average deviation from actual bill.")
    st.metric(label="Root Mean Squared Error (RMSE)", value="₹560.10", help="Standard error penalty for outliers.")
    
    st.divider()
    st.caption("⚡ Model: Ridge Regularized Linear Regressor")

# --- Main Application Header ---
st.title("⚡ Household Electricity Bill Estimator")
st.markdown("Adjust appliance usage parameters and location details to generate an estimated monthly consumption forecast.")

st.markdown("---")

# --- Input Sections Organized in Visual Containers ---
col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.subheader("🏠 Heavy & Continuous Appliances")
    fan = st.slider("Number of Ceiling Fans", min_value=0, max_value=15, value=4, step=1)
    refrigerator = st.slider("Refrigerator Operating Hours / Day", min_value=0.0, max_value=24.0, value=24.0, step=0.5)
    air_conditioner = st.number_input("Air Conditioners Count (AC)", min_value=0, max_value=8, value=2, step=1)

with col_right:
    st.subheader("💻 Entertainment & Utility")
    television = st.slider("Television Hours / Day", min_value=0.0, max_value=24.0, value=6.0, step=0.5)
    monitor = st.number_input("Computer Monitors / Workstations", min_value=0, max_value=10, value=1, step=1)
    
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        tariff_rate = st.number_input("Tariff Rate (₹/kWh)", min_value=1.0, max_value=20.0, value=8.60, step=0.1)
    with sub_col2:
        month_mapping = {
            "January": 1, "February": 2, "March": 3, "April": 4,
            "May": 5, "June": 6, "July": 7, "August": 8,
            "September": 9, "October": 10, "November": 11, "December": 12
        }
        selected_month = st.selectbox("Billing Month", list(month_mapping.keys()), index=7)
        month = month_mapping[selected_month]

monthly_hours = 720  # Standard 30-day baseline

st.markdown("---")

# Prediction Execution
predict_clicked = st.button("🚀 Calculate Estimated Bill", type="primary")

if predict_clicked:
    input_data = pd.DataFrame([[
        fan, refrigerator, air_conditioner, television, 
        monitor, month, monthly_hours, tariff_rate
    ]], columns=['Fan', 'Refrigerator', 'AirConditioner', 'Television', 'Monitor', 'Month', 'MonthlyHours', 'TariffRate'])
    
    prediction = max(100.0, float(model.predict(input_data)[0]))
    est_units = prediction / tariff_rate
    
    st.markdown(f"""
        <div class="result-card">
            <h4 style="margin: 0; color: #a7f3d0;">Estimated Electricity Bill for {selected_month}</h4>
            <div class="result-val">₹{prediction:,.2f}</div>
            <p style="margin: 5px 0 0 0; color: #6ee7b7; font-size: 0.95rem;">Estimated Consumption: <b>{est_units:.1f} kWh (Units)</b></p>
        </div>
    """, unsafe_allow_html=True)
    
    # Quick visual summary cards
    st.write("")
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Daily Avg Cost", f"₹{(prediction / 30):,.2f}")
    m_col2.metric("Effective Tariff", f"₹{tariff_rate:.2f}/unit")
    m_col3.metric("Grid Status", "Peak Load" if air_conditioner > 2 else "Optimal")
