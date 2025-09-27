import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# Load model using absolute path
model = joblib.load(r"C:\Users\kayal\Desktop\Heart_Disease_Project\models\final_model.pkl")

print("✅ Model loaded successfully!")


st.set_page_config(page_title="❤️ Heart Disease Prediction", layout="wide")

# Title
st.title("❤️ Heart Disease Prediction Web App")

st.write("Enter patient details to predict the likelihood of heart disease.")

# Sidebar for inputs
st.sidebar.header("Patient Data Input")

# Define user inputs (based on your dataset features)
def user_input():
    age = st.sidebar.number_input("Age", 20, 100, 50)
    sex = st.sidebar.selectbox("Sex", ("Male", "Female"))
    cp = st.sidebar.selectbox("Chest Pain Type (cp)", (0, 1, 2, 3))
    trestbps = st.sidebar.number_input("Resting Blood Pressure (trestbps)", 80, 200, 120)
    chol = st.sidebar.number_input("Serum Cholestoral (chol)", 100, 600, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", (0, 1))
    restecg = st.sidebar.selectbox("Resting ECG (restecg)", (0, 1, 2))
    thalach = st.sidebar.number_input("Max Heart Rate Achieved (thalach)", 60, 220, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina (exang)", (0, 1))
    oldpeak = st.sidebar.number_input("ST Depression (oldpeak)", 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox("Slope", (0, 1, 2))
    ca = st.sidebar.number_input("Number of Major Vessels (ca)", 0, 4, 0)
    thal = st.sidebar.selectbox("Thal", (0, 1, 2, 3))

    # Convert categorical inputs
    sex = 1 if sex == "Male" else 0

    data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }
    return pd.DataFrame([data])

# Get user input
input_df = user_input()

# Prediction
if st.sidebar.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ The model predicts a HIGH risk of Heart Disease (Probability: {probability:.2f})")
    else:
        st.success(f"✅ The model predicts a LOW risk of Heart Disease (Probability: {probability:.2f})")

# 📊 Extra: Data Visualization (Example: Age Distribution)
st.subheader("📊 Heart Disease Data Insights")
uploaded_file = st.file_uploader("Upload heart_disease.csv to explore data", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["age"], bins=20, kde=True, ax=ax)
    plt.title("Age Distribution of Patients")
    st.pyplot(fig)