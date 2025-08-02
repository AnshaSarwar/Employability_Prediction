# Import Necessary Libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import os

# ─────────────────────────────────────────────────────────────
# 📋 PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="🎓 Student Employability Predictor", layout="centered")

# ─────────────────────────────────────────────────────────────
# 📋 STYLING
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp {
    background-color: #e6f2ff;
}
html, body, [class*="css"] {
    font-size: 14px;
}
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 📋 LOAD MODEL & SCALER
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    model_path = "employability_predictor.pkl"
    scaler_path = "scaler.pkl"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_resources()

if model is None or scaler is None:
    st.error("⚠️ Model or scaler file not found. Please ensure 'employability_predictor.pkl' and 'scaler.pkl' exist.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# 📋 HEADER IMAGE
# ─────────────────────────────────────────────────────────────
try:
    image = Image.open("group-business-people-silhouette-businesspeople-abstract-background_656098-461.avif")
    st.image(image, use_container_width=True)
except FileNotFoundError:
    st.warning("Header image not found. Skipping image display.")

# ─────────────────────────────────────────────────────────────
# 📋 TITLE
# ─────────────────────────────────────────────────────────────
st.markdown("<h2 style='text-align: center;'>🎓 Student Employability Predictor — SVM Model</h2>", unsafe_allow_html=True)
st.markdown("Fill in the input features to predict employability.")

# ─────────────────────────────────────────────────────────────
# 📋 FEATURE INPUTS
# ─────────────────────────────────────────────────────────────
feature_columns = [
    'GENDER', 'GENERAL_APPEARANCE', 'GENERAL_POINT_AVERAGE',
    'MANNER_OF_SPEAKING', 'PHYSICAL_CONDITION', 'MENTAL_ALERTNESS',
    'SELF-CONFIDENCE', 'ABILITY_TO_PRESENT_IDEAS', 'COMMUNICATION_SKILLS',
    'STUDENT_PERFORMANCE_RATING', 'NO_SKILLS', 'Year_of_Graduate'
]

def get_user_inputs():
    col1, col2, col3 = st.columns(3)
    inputs = {}

    with col1:
        inputs['GENDER'] = st.radio("Gender", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female", index=1)
        inputs['GENERAL_APPEARANCE'] = st.slider("Appearance (1-5)", 1, 5, 3)
        inputs['GENERAL_POINT_AVERAGE'] = st.number_input("GPA (0.0 - 4.0)", 0.0, 4.0, 3.0, 0.01)
        inputs['MANNER_OF_SPEAKING'] = st.slider("Speaking (1-5)", 1, 5, 3)

    with col2:
        inputs['PHYSICAL_CONDITION'] = st.slider("Physical (1-5)", 1, 5, 3)
        inputs['MENTAL_ALERTNESS'] = st.slider("Alertness (1-5)", 1, 5, 3)
        inputs['SELF-CONFIDENCE'] = st.slider("Confidence (1-5)", 1, 5, 3)
        inputs['ABILITY_TO_PRESENT_IDEAS'] = st.slider("Ideas (1-5)", 1, 5, 3)

    with col3:
        inputs['COMMUNICATION_SKILLS'] = st.slider("Communication (1-5)", 1, 5, 3)
        inputs['STUDENT_PERFORMANCE_RATING'] = st.slider("Performance (1-5)", 1, 5, 3)
        inputs['NO_SKILLS'] = st.radio("Has No Skills", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0)
        inputs['Year_of_Graduate'] = st.number_input("Graduation Year", 2019, 2022, 2022)

    return pd.DataFrame([inputs])[feature_columns]

def predict_employability(model, scaler, input_df):
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    proba = model.predict_proba(scaled_input)[0]
    return prediction, proba

# ─────────────────────────────────────────────────────────────
# 📋 PREDICT BUTTON
# ─────────────────────────────────────────────────────────────
input_df = get_user_inputs()

if st.button("Predict"):
    prediction, probability = predict_employability(model, scaler, input_df)

    if prediction == 1:
        st.success("🎉 The student is predicted to be **Employable**!")
        st.balloons()
    else:
        st.warning("⚠️ The student is predicted to be **Less Employable**.")

    st.info(f"Probability of being Employable: {probability[1] * 100:.2f}%")
    st.info(f"Probability of being Less Employable: {probability[0] * 100:.2f}%")

# ─────────────────────────────────────────────────────────────
# 📋 FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("""
Disclaimer: This prediction model is for research and informational purposes only.  
Version 1.0, © 2025 CHOONG MUH IN — Last updated: August 2025.  
Developed by Ms. CHOONG MUH IN (TP068331)
""")
