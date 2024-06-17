import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Load the trained model
with open('tree_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the label encoders
with open('label_encoders.pkl', 'rb') as file:
    label_encoders = pickle.load(file)

# Title of the app
st.title("Heart Disease Prediction App")

# Input fields
age = st.number_input('Age', min_value=1, max_value=120, value=25)
sex = st.selectbox('Sex', ['Male', 'Female'])
cp = st.selectbox('Chest Pain Type', ['typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'])
trestbps = st.number_input('Resting Blood Pressure', min_value=50, max_value=200, value=120)
chol = st.number_input('Cholesterol', min_value=100, max_value=600, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [True, False])
restecg = st.selectbox('Resting ECG', ['normal', 'lv hypertrophy', 'st-t abnormality'])
thalach = st.number_input('Maximum Heart Rate Achieved', min_value=50, max_value=250, value=150)
exang = st.selectbox('Exercise Induced Angina', [True, False])
oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox('Slope of Peak Exercise ST Segment', ['upsloping', 'flat', 'downsloping'])
ca = st.number_input('Number of Major Vessels Colored by Fluoroscopy', min_value=0, max_value=4, value=0)
thal = st.selectbox('Thalassemia', ['normal', 'fixed defect', 'reversable defect'])

# Encode categorical variables
sex = label_encoders['sex'].transform([sex])[0]
cp = label_encoders['cp'].transform([cp])[0]
restecg = label_encoders['restecg'].transform([restecg])[0]
exang = label_encoders['exang'].transform([exang])[0]
slope = label_encoders['slope'].transform([slope])[0]
thal = label_encoders['thal'].transform([thal])[0]

# Prepare the feature vector for prediction
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

# Scale the input data
scaler = StandardScaler()
input_data_scaled = scaler.fit_transform(input_data)

# Predict button
if st.button('Predict'):
    prediction = model.predict(input_data_scaled)

    if prediction == 1:
        st.write('The model predicts that you are at risk of heart disease.')
    else:
        st.write('The model predicts that you are not at risk of heart disease.')
