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

# Define min and max values
min_values = {
    "Pregnancies": 0,
    "Glucose": 70,
    "BloodPressure": 70,
    "SkinThickness": 10,
    "Insulin": 2.6,
    "BMI": 15.0,
    "DiabetesPedigreeFunction": 0.1,
    "Age": 0
}

max_values = {
    "Pregnancies": 10,
    "Glucose": 200,
    "BloodPressure": 180,
    "SkinThickness": 60,
    "Insulin": 300.0,
    "BMI": 65.0,
    "DiabetesPedigreeFunction": 2.0,
    "Age": 100
}

# Load and preprocess the dataset
diabetes_df = pd.read_csv("diabetes.csv")

 
# Streamlit UI
st.title("Diabetes Prediction App :stethoscope: :lollipop:")
st.image("https://images.unsplash.com/photo-1633613287441-3f72304088ad?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D", caption="-", width=1200, use_container_width=True)
st.markdown(
    """
    **Diabetes is a growing health concern worldwide, affecting millions of people.**
    
    Early detection is key to managing and preventing complications.  
    This app analyzes key health indicators—such as glucose levels, BMI, blood pressure, and more—to assess the likelihood of diabetes.  

    :warning: *This is not a medical diagnosis. Please consult a healthcare professional for proper evaluation.*
    """)


st.sidebar.header("User Input Features")

# User input fields in the sidebar
pregnancies = st.sidebar.number_input("Pregnancies", min_value=min_values["Pregnancies"], max_value=max_values["Pregnancies"], value=1)
glucose = st.sidebar.number_input("Glucose Level", min_value=min_values["Glucose"], max_value=max_values["Glucose"], value=100)
blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=min_values["BloodPressure"], max_value=max_values["BloodPressure"], value=80)
skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=min_values["SkinThickness"], max_value=max_values["SkinThickness"], value=20)
insulin = st.sidebar.number_input("Insulin Level", min_value=min_values["Insulin"], max_value=max_values["Insulin"], value=30.0)
bmi = st.sidebar.number_input("BMI", min_value=min_values["BMI"], max_value=max_values["BMI"], value=25.0)
diabetes_pedigree = st.sidebar.number_input("Diabetes Pedigree Function", min_value=min_values["DiabetesPedigreeFunction"], max_value=max_values["DiabetesPedigreeFunction"], value=0.5)
age = st.sidebar.number_input("Age", min_value=min_values["Age"], max_value=max_values["Age"], value=30)


# Prediction
if st.sidebar.button("Predict Diabetes"):
    patient_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]
    result, probabilities = predict_diabetes(patient_data)
    
    st.subheader("Prediction Result")
    
    if result == 1:
        st.error("Oops! You might have diabetes. But don't worry, here’s some helpful information: [Diabetes Info](https://www.diabetes.org/)")
        st.image("https://media4.giphy.com/media/9NSeIavwr6qt1E0grf/giphy.gif", caption="-", width=350)
    else:
        st.success("Great! You are not diabetic. Keep maintaining a healthy lifestyle!")
        st.image("https://img.freepik.com/free-vector/healthy-people-carrying-different-icons_53876-43069.jpg", caption="-", width=350)
        

    warning_message = st.empty()
    warning_message.warning("Note: This is a predictive model and should not be used as a substitute for professional medical advice.")
    time.sleep(5)
    warning_message.empty()

    # Advice to avoid diabetes 
    st.header("💡 Tips for Preventing Diabetes & Staying Healthy")

    st.markdown(
        """
        :white_check_mark: **Eat a balanced diet** – Focus on whole foods, fiber-rich vegetables, lean proteins, and healthy fats.  
        :woman-running: **Stay active** – Aim for at least 30 minutes of exercise most days. Walking, cycling, or dancing can help!  
        ⚖️ **Maintain a healthy weight** – Managing weight reduces the risk of diabetes and other diseases.  
        :droplet: **Drink plenty of water** – Staying hydrated helps regulate blood sugar levels.  
        :bed: **Get enough sleep** – Poor sleep can affect blood sugar control. Aim for 7–9 hours per night.  
        :stethoscope: **Monitor your health** – Regular check-ups with your doctor can help detect early signs of diabetes.  
        """
    )

# Final note
    st.markdown("🚀 *Making small, consistent changes can help you lead a healthier, happier life!*")

    # Main section for predictions and data exploration
    st.title("Explore the Data")
    with st.expander("Click to view the dataset"):
        st.dataframe(diabetes_df)

