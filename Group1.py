import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from Load_Trained_Model import predict_diabetes

min_values = {
    "Pregnancies": 0,
    "Glucose": 50,
    "BloodPressure": 50,
    "SkinThickness": 0,
    "Insulin": 0,
    "BMI": 10.0,
    "DiabetesPedigreeFunction": 0.08,
    "Age": 18
}

max_values = {
    "Pregnancies": 17,
    "Glucose": 250,
    "BloodPressure": 180,
    "SkinThickness": 99,
    "Insulin": 900,
    "BMI": 67.0,
    "DiabetesPedigreeFunction": 2.5,
    "Age": 120
}

# Load and preprocess the dataset
diabetes_df = pd.read_csv("diabetes.csv")

 
# Streamlit UI
st.title("Diabetes Prediction App")
st.image("https://media4.giphy.com/media/9NSeIavwr6qt1E0grf/giphy.gif", caption="-", width=400)
st.sidebar.header("User Input Features")

# User input fields in the sidebar
pregnancies = st.sidebar.number_input("Pregnancies", min_value=min_values["Pregnancies"], max_value=max_values["Pregnancies"], value=1)
glucose = st.sidebar.number_input("Glucose Level", min_value=min_values["Glucose"], max_value=max_values["Glucose"], value=100)
blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=min_values["BloodPressure"], max_value=max_values["BloodPressure"], value=80)
skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=min_values["SkinThickness"], max_value=max_values["SkinThickness"], value=20)
insulin = st.sidebar.number_input("Insulin Level", min_value=min_values["Insulin"], max_value=max_values["Insulin"], value=30)
bmi = st.sidebar.number_input("BMI", min_value=min_values["BMI"], max_value=max_values["BMI"], value=25.0)
diabetes_pedigree = st.sidebar.number_input("Diabetes Pedigree Function", min_value=min_values["DiabetesPedigreeFunction"], max_value=max_values["DiabetesPedigreeFunction"], value=0.5)
age = st.sidebar.number_input("Age", min_value=min_values["Age"], max_value=max_values["Age"], value=30)


# Prediction
if st.sidebar.button("Predict Diabetes"):
    patient_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]
    result, probabilities = predict_diabetes(patient_data)
    
    st.subheader("Prediction Result")
    
    if result == "Diabetic":
        st.error("Oops! You might have diabetes. But don't worry, here’s some helpful information: [Diabetes Info](https://www.diabetes.org/)")
    else:
        st.success("Great! You are not diabetic. Keep maintaining a healthy lifestyle!")
        

    warning_message = st.empty()
    warning_message.warning("Note: This is a predictive model and should not be used as a substitute for professional medical advice.")
    time.sleep(5)
    warning_message.empty()

    # Main section for predictions and data exploration
    st.title("Explore the Data")
    with st.expander("Click to view the dataset"):
        st.dataframe(diabetes_df)

