import streamlit as st
import numpy as np
import pickle

# Load trained model
model = pickle.load(open("diabetes_rf_model.pkl", "rb"))

st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")
st.title("💊 Diabetes Risk Predictor")
st.markdown("Enter your health details to check your diabetes risk.")

# Input fields
preg = st.number_input("Pregnancies", 0, 20, step=1)
glucose = st.number_input("Glucose Level", 0, 200, step=1)
bp = st.number_input("Blood Pressure", 0, 140, step=1)
skin = st.number_input("Skin Thickness", 0, 100, step=1)
insulin = st.number_input("Insulin Level", 0, 900, step=1)
bmi = st.number_input("BMI", 0.0, 70.0, step=0.1)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, step=0.01)
age = st.number_input("Age", 10, 100, step=1)

if st.button("🔍 Predict"):
    user_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(user_data)[0]
    if prediction == 1:
        st.error("🔴 High Risk: You may have diabetes. Please consult a doctor.")
    else:
        st.success("🟢 Low Risk: You are unlikely to have diabetes.")

st.caption("📊 Powered by a Random Forest Classifier trained on the PIMA Indian Diabetes Dataset.")
