import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset
df = pd.read_csv("medical_insurance_dataset.csv")
print(df.head())

# Encode categorical variables if present
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# Check if the dataset has a 'PremiumPrice' column
if 'PremiumPrice' not in df.columns:
    raise ValueError("Dataset must contain a 'PremiumPrice' column as the target variable.")

# Define features and target
X = df.drop(columns=['PremiumPrice'])
y = df['PremiumPrice']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Train Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Train XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# Evaluate models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, mse, r2

models = {"Linear Regression": lr, "Random Forest": rf, "XGBoost": xgb_model}
for name, model in models.items():
    mae, mse, r2 = evaluate_model(model, X_test, y_test)
    print(f"{name} - MAE: {mae}, MSE: {mse}, R2 Score: {r2}")

# Save models
with open("linear_regression.pkl", "wb") as f:
    pickle.dump(lr, f)
with open("random_forest.pkl", "wb") as f:
    pickle.dump(rf, f)
with open("xgboost.pkl", "wb") as f:
    pickle.dump(xgb_model, f)

print("Models trained and saved successfully.")