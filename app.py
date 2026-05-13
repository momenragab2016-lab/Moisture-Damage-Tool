import streamlit as st
import pandas as pd
import joblib

# =========================
# Load models and preprocessors
# =========================
detection_model = joblib.load("detection_model.pkl")
severity_model = joblib.load("severity_model.pkl")

preprocessor_detection = joblib.load("preprocessor_detection.pkl")
preprocessor_severity = joblib.load("preprocessor_severity.pkl")

# =========================
# Title
# =========================
st.title("Moisture Damage Prediction Tool")

st.subheader("Enter Pavement Information")

# =========================
# INPUTS
# =========================

age_class = st.selectbox("Age class", ["0-2", "3-5", "6-8", "9-12", ">=13"])
mr_class = st.text_input("M&R Class")
base_type = st.text_input("Base type Code")
subbase_type = st.text_input("Subbase type Code")
subgrade = st.text_input("Subgrade Code")
drainage_location = st.text_input("Drainage Location")
drainage_type = st.text_input("Drainage Type")

aadtt = st.number_input("AADTT", value=1000)
temp = st.number_input("Annual Temperature", value=25.0)
prec = st.number_input("Annual Precipitation", value=10.0)
humidity = st.number_input("Annual Humidity", value=60.0)
ac_thickness = st.number_input("AC Thickness", value=5.0)
base_thickness = st.number_input("Base Thickness", value=10.0)
subbase_thickness = st.number_input("Subbase Thickness", value=15.0)
pipe_diameter = st.number_input("Drainpipe Diameter", value=5.0)

# =========================
# PREDICTION
# =========================

if st.button("Predict"):

    input_data = pd.DataFrame([{
        "Age class": age_class,
        "M&R Class": mr_class,
        "Base type Code": base_type,
        "Subbase type code": subbase_type,
        "Subgrade Code": subgrade,
        "DRAINAGE_LOCATION_EXP": drainage_location,
        "DRAINAGE_TYPE_EXP": drainage_type,
        "AADTT": aadtt,
        "Annual Temp": temp,
        "Annual Prec.": prec,
        "Annual Humidity": humidity,
        "AC thickness": ac_thickness,
        "Base thickness": base_thickness,
        "Subbase Thickness": subbase_thickness,
        "DRAINPIPE_DIAMETER": pipe_diameter
    }])

    categorical_cols = [
        "Age class",
        "M&R Class",
        "Base type Code",
        "Subbase type code",
        "Subgrade Code",
        "DRAINAGE_LOCATION_EXP",
        "DRAINAGE_TYPE_EXP"
    ]

    input_data[categorical_cols] = input_data[categorical_cols].astype(str)

    # =========================
    # DETECTION MODEL
    # =========================
    input_processed_d = preprocessor_detection.transform(input_data)
    prediction_d = detection_model.predict(input_processed_d)[0]

    if prediction_d == 0:
        st.success("✅ No Stripping Detected")

    else:
        st.warning("⚠️ Stripping Detected")

        # =========================
        # SEVERITY MODEL
        # =========================
        input_processed_s = preprocessor_severity.transform(input_data)
        prediction_s = severity_model.predict(input_processed_s)[0]

        severity_labels = ["Low", "Moderate", "High"]

        st.info(f"Severity Level: {severity_labels[prediction_s]}")
