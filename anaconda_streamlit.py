import numpy as np
import streamlit as st
import pickle
import os

# Ensure correct path
MODEL_PATH = os.path.join("trained_diabetes_model.sav")

# Load model + scaler
loaded_model = pickle.load(open(MODEL_PATH, 'rb'))
model = loaded_model["model"]
scaler = loaded_model["scaler"]

def diabetes_prediction(input_data):
    # Convert inputs to numpy array and reshape
    input_array = np.asarray(input_data, dtype=float).reshape(1, -1)

    # Apply scaler
    std_input = scaler.transform(input_array)

    # Predict
    prediction = model.predict(std_input)

    if prediction[0] == 0:
        return "The person is NOT diabetic"
    else:
        return "The person is diabetic"

def main():
    st.title('Diabetes Prediction Web App')

    # Input fields
    Pregnancies = st.number_input('Number of Pregnancies', min_value=0, step=1)
    Glucose = st.number_input('Glucose Level', min_value=0)
    BloodPressure = st.number_input('Blood Pressure value', min_value=0)
    SkinThickness = st.number_input('Skin Thickness value', min_value=0)
    Insulin = st.number_input('Insulin Level', min_value=0)
    BMI = st.number_input('BMI value', min_value=0.0, format="%.1f")
    DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value', min_value=0.0, format="%.2f")
    Age = st.number_input('Age of the Person', min_value=0, step=1)

    # Prediction button
    diagnosis = ''
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
    
    st.success(diagnosis)

if __name__ == '__main__':
    main()
