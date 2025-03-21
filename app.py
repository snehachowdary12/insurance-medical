import streamlit as st
import pickle
import numpy as np

# Load trained models


# Load trained models with correct paths
with open("linear_regression.pkl", "rb") as f:
    lr_model = pickle.load(f)

with open("random_forest.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("xgboost.pkl", "rb") as f:
    xgb_model = pickle.load(f)


# Streamlit UI
st.title("Medical Insurance Premium Prediction")
st.sidebar.header("Enter Patient Details")

# User inputs
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
diabetes = st.sidebar.selectbox("Diabetes", [0, 1])
blood_pressure = st.sidebar.selectbox("Blood Pressure Problems", [0, 1])
transplants = st.sidebar.selectbox("Any Transplants", [0, 1])
chronic_diseases = st.sidebar.selectbox("Any Chronic Diseases", [0, 1])
height = st.sidebar.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
weight = st.sidebar.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
allergies = st.sidebar.selectbox("Known Allergies", [0, 1])
history_cancer = st.sidebar.selectbox("History of Cancer in Family", [0, 1])
major_surgeries = st.sidebar.number_input("Number of Major Surgeries", min_value=0, max_value=10, value=0)

# Prepare input data
input_data = np.array([[age, diabetes, blood_pressure, transplants, chronic_diseases, height, weight, allergies, history_cancer, major_surgeries]])

# Predictions
if st.sidebar.button("Predict"):
    lr_pred = lr_model.predict(input_data)[0]
    rf_pred = rf_model.predict(input_data)[0]
    xgb_pred = xgb_model.predict(input_data)[0]
    
    # Display results
    st.write("### Predicted Premium Prices")
    st.write(f"Linear Regression: **₹{lr_pred:.2f}**")
    st.write(f"Random Forest: **₹{rf_pred:.2f}**")
    st.write(f"XGBoost: **₹{xgb_pred:.2f}**")