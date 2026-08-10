
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
df = pd.read_csv("electricity_bill_dataset.csv").drop_duplicates()
if 'MotorPump' in df.columns:
    df = df.drop(columns=['MotorPump'])

# Features matching app.py
features = ['Fan', 'Refrigerator', 'AirConditioner', 'Television', 'Monitor', 'Month', 'MonthlyHours', 'TariffRate']
X = df[features]
y = df['ElectricityBill']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit and export trained model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'electricity_bill_model.pkl')
print("electricity_bill_model.pkl generated successfully!")