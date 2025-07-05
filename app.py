import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("water_quality_model.pkl")

# App title
st.title("💧 Water Quality Prediction App")
st.write("Enter the chemical values to check if the water is **Safe** or **Unsafe**.")

# Input fields
O2 = st.number_input("Oxygen (O2)", min_value=0.0)
Suspended = st.number_input("Suspended Solids", min_value=0.0)
NH4 = st.number_input("Ammonium (NH4)", min_value=0.0)
NO3 = st.number_input("Nitrate (NO3)", min_value=0.0)
NO2 = st.number_input("Nitrite (NO2)", min_value=0.0)
SO4 = st.number_input("Sulfate (SO4)", min_value=0.0)
PO4 = st.number_input("Phosphate (PO4)", min_value=0.0)
CL = st.number_input("Chloride (CL)", min_value=0.0)
BSK5 = st.number_input("Biochemical Oxygen Demand (BSK5)", min_value=0.0)

# Predict button
if st.button("Predict Water Quality"):
    features = np.array([[O2, Suspended, NH4, NO3, NO2, SO4, PO4, CL, BSK5]])
    result = model.predict(features)
    if result[0] == 1:
        st.success(" The water is SAFE to drink.")
    else:
        st.error(" The water is UNSAFE to drink.")
