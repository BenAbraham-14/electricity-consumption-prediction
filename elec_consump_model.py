import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# 1. Load dataset
df = pd.read_csv("electricity_bill_dataset.csv").drop_duplicates()
if 'MotorPump' in df.columns:
    df = df.drop(columns=['MotorPump'])

# 2. Features matching app.py
features = ['Fan', 'Refrigerator', 'AirConditioner', 'Television', 'Monitor', 'Month', 'MonthlyHours', 'TariffRate']
X = df[features]
y = df['ElectricityBill']

# 3. Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Random Forest Model
model = RandomForestRegressor(n_estimators=20, max_depth=15, random_state=42)
model.fit(X_train, y_train)

# 5. Generate Predictions on Test Data
y_pred = model.predict(X_test)

# 6. Calculate Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# 7. Print Model Performance Report
print("=" * 45)
print("       MODEL EVALUATION METRICS REPORT       ")
print("=" * 45)
print(f"Mean Absolute Error (MAE)     : ₹{mae:.2f}")
print(f"Mean Squared Error (MSE)      : {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): ₹{rmse:.2f}")
print(f"R-squared Score (R²)          : {r2:.6f}")
print("=" * 45)

# 8. Save model file
joblib.dump(model, 'electricity_bill_model.pkl', compress=3)
print("Model saved as electricity_bill_model.pkl successfully!")
