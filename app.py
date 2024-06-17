import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pickle
from PIL import Image

# Load the trained model
with open('tree_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load label encoders
with open('label_encoders.pkl', 'rb') as file:
    label_encoders = pickle.load(file)

# Title of the app with an image
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")
st.title("Heart Disease Risk Assessment")
st.image("assets/heart_image.jpg", use_column_width=True)


# Sidebar
st.sidebar.header('User Input Parameters')
st.sidebar.markdown("Please enter the following details to predict the risk of heart disease.")

# Input fields in the sidebar
age = st.sidebar.number_input('Age', min_value=1, max_value=120, value=25)
sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
cp = st.sidebar.selectbox('Chest Pain Type', ['typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'])
trestbps = st.sidebar.number_input('Resting Blood Pressure', min_value=50, max_value=200, value=120)
chol = st.sidebar.number_input('Cholesterol', min_value=100, max_value=600, value=200)
fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', [True, False])
restecg = st.sidebar.selectbox('Resting ECG', ['normal', 'lv hypertrophy', 'st-t abnormality'])
thalach = st.sidebar.number_input('Maximum Heart Rate Achieved', min_value=50, max_value=250, value=150)
exang = st.sidebar.selectbox('Exercise Induced Angina', [True, False])
oldpeak = st.sidebar.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=1.0)
slope = st.sidebar.selectbox('Slope of Peak Exercise ST Segment', ['upsloping', 'flat', 'downsloping'])
ca = st.sidebar.number_input('Number of Major Vessels Colored by Fluoroscopy', min_value=0, max_value=4, value=0)
thal = st.sidebar.selectbox('Thalassemia', ['normal', 'fixed defect', 'reversable defect'])

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
if st.sidebar.button('Predict'):
    prediction = model.predict(input_data_scaled)
        
    if prediction == 1:
        st.markdown(
            """
            <div style="margin: 0 auto; width: 500px; padding: 20px; border-radius: 5px; background-color: #BFFF66; color: black; text-align: center; font-size: 20px; font-weight: bold;">                    
                <h2 style="color: black;">Prediction Result</h2>
                <p>The model predicts that you are <b style="color: red;">at risk</b> of heart disease.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            """
            <div style="margin: 0 auto; width: 500px; padding: 20px; border-radius: 5px; background-color: #BFFF66; color: black; text-align: center; font-size: 20px; font-weight: bold;">  
                <h2 style="color: black;">Prediction Result</h2>
                <p>The model predicts that you are <b style="color: red;">not at risk</b> of heart disease.</p>
            </div>
            """,
            unsafe_allow_html=True
        )



# Additional informative content
st.markdown("""
    ## What You Should Know About Heart Disease
    Heart disease, also known as cardiovascular disease, is a broad term encompassing various conditions that affect the heart and blood vessels. It is the leading cause of death worldwide, with risk factors including high blood pressure, high cholesterol, smoking, obesity, diabetes, and a sedentary lifestyle. Symptoms can vary depending on the specific condition but often include chest pain, shortness of breath, fatigue, and irregular heartbeat. Preventative measures include a healthy diet, regular exercise, maintaining a healthy weight, avoiding tobacco, and controlling blood pressure and cholesterol levels. Early detection and management are crucial to improving outcomes and reducing the risk of severe complications such as heart attacks and strokes.
    """)
st.image("assets/info.jpeg", use_column_width=True)
st.markdown("""
    ### How to Reduce Your Risk
    - **Eat a healthy diet**: Include plenty of fruits, vegetables, and whole grains.
    - **Exercise regularly**: Aim for at least 30 minutes of moderate exercise most days of the week.
    - **Maintain a healthy weight**: Lose weight if you are overweight or obese.
    - **Quit smoking**: If you smoke, quitting is the best thing you can do for your heart health.
    - **Limit alcohol**: Drink alcohol in moderation.
    - **Manage stress**: Reduce stress as much as possible.
    """)
