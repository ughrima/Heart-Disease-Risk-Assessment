
# Heart Disease Risk Assessment

This project is a web application that allows users to assess their risk of heart disease by entering various health metrics. The app uses a machine learning model to predict whether a user is at risk of heart disease based on the input data.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Input Parameters](#input-parameters)
- [Output](#output)
- [Machine Learning Model](#machine-learning-model)

## Overview
Heart disease describes a range of conditions that affect your heart. These diseases include blood vessel diseases, such as coronary artery disease; heart rhythm problems (arrhythmias); and heart defects you're born with (congenital heart defects). This application helps in predicting the risk of heart disease based on user input, leveraging a trained machine learning model.

## Features
- User-friendly interface for inputting health metrics.
- Machine learning model to predict heart disease risk.
- Informative sections about heart disease and how to reduce the risk.
- Responsive design and visually appealing UI.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/heart-disease-risk-assessment.git
    ```
2. Navigate to the project directory:
    ```bash
    cd heart-disease-risk-assessment
    ```
3. Create a virtual environment:
    ```bash
    python -m venv venv
    ```
4. Activate the virtual environment:
    - On Windows:
        ```bash
        venv\Scripts\activate
        ```
    - On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```
5. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Start the Streamlit app:
    ```bash
    streamlit run app.py
    ```
2. Open your web browser and navigate to the provided local URL (usually http://localhost:8501).

## Input Parameters
The following input parameters are required to predict the risk of heart disease:
- Age
- Sex (Male/Female)
- Chest Pain Type (typical angina/atypical angina/non-anginal/asymptomatic)
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar > 120 mg/dl (True/False)
- Resting ECG (normal/lv hypertrophy/st-t abnormality)
- Maximum Heart Rate Achieved
- Exercise Induced Angina (True/False)
- ST Depression Induced by Exercise
- Slope of Peak Exercise ST Segment (upsloping/flat/downsloping)
- Number of Major Vessels Colored by Fluoroscopy
- Thalassemia (normal/fixed defect/reversable defect)

## Output
Based on the input parameters, the application will predict whether the user is at risk of heart disease. The result will be displayed in a highlighted box along with relevant images:
- **At Risk**: The prediction box will have a message indicating the risk along with a relevant image.
- **Not at Risk**: The prediction box will have a message indicating no risk along with a relevant image.

## Machine Learning Model
The machine learning model was trained using the UCI Heart Disease dataset. Various algorithms were tested, including Logistic Regression, Decision Tree, Random Forest, and Support Vector Machine (SVM). 

### Data Preprocessing
- The data was cleaned, missing values were filled, and categorical variables were encoded.
- The dataset was split into training and testing sets.
- SMOTE was applied to balance the dataset.
- The data was scaled using StandardScaler.

### Model Training and Evaluation
Several models were trained and evaluated:
- **Logistic Regression** achieved an accuracy of 77.17%.
- **Decision Tree** was chosen for deployment due to its high accuracy of 97%.
- **Random Forest** achieved an accuracy of 80%.
- **Support Vector Machine (SVM)** achieved an accuracy of 75%.

The Decision Tree model was ultimately selected for deployment due to its superior performance.

