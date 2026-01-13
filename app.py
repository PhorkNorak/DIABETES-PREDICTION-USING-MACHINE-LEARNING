import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

FEATURES = [
    "Pregnancies",
    "PlasmaGlucose",
    "DiastolicBloodPressure",
    "TricepsThickness",
    "SerumInsulin",
    "BMI",
    "DiabetesPedigree",
    "Age",
]
MODEL_PATH = Path("artifacts/best_diabetes_model.joblib")

def load_model():
    if not MODEL_PATH.exists():
        st.error("Model file not found. Run the training cells to create artifacts/best_diabetes_model.joblib")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

st.set_page_config(page_title="Taipei Diabetes Predictor", page_icon="stethoscope", layout="centered")
st.title("Taipei Diabetes Predictor")
st.write(
    "Enter patient data to estimate the probability of diabetes based on the Taipei medical center dataset."
)

with st.form("predict-form"):
    inputs = {}
    inputs["Pregnancies"] = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
    inputs["PlasmaGlucose"] = st.number_input("Plasma Glucose", min_value=0.0, max_value=300.0, value=120.0)
    inputs["DiastolicBloodPressure"] = st.number_input("Diastolic Blood Pressure (mm Hg)", min_value=0.0, max_value=200.0, value=70.0)
    inputs["TricepsThickness"] = st.number_input("Triceps Thickness (mm)", min_value=0.0, max_value=100.0, value=25.0)
    inputs["SerumInsulin"] = st.number_input("Serum Insulin (mu U/ml)", min_value=0.0, max_value=500.0, value=80.0)
    inputs["BMI"] = st.number_input("BMI", min_value=0.0, max_value=70.0, value=26.0)
    inputs["DiabetesPedigree"] = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=5.0, value=0.5)
    inputs["Age"] = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([inputs])
    proba = model.predict_proba(input_df)[0, 1]
    pred = int(proba >= 0.5)
    label = "Diabetic" if pred == 1 else "Not diabetic"

    st.metric("Estimated probability", f"{proba:.2%}")
    st.success(f"Prediction: {label}")
# streamlit run app.py