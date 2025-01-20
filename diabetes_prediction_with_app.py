import streamlit as st
import pickle
import numpy as np
import requests

# Function to load the model from GitHub
def load_model_from_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        with open('trained_model.sav', 'wb') as f:
            f.write(response.content)
        return pickle.load(open('trained_model.sav', 'rb'))
    else:
        st.error("Error loading model from GitHub. Please check the URL.")
        return None

# Replace with your actual raw URL of the trained model
model_url = 'https://github.com/Oduobuk/Diabetes-prediction-app/raw/refs/heads/main/trained_model.sav'
loaded_model = load_model_from_github(model_url)

def diabetes_prediction(input_data):
    # Convert input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)

    # Reshape array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

def main():
    # Title of the app
    st.title('Diabetes Prediction Web App')

    # Getting the input data from the user
    try:
        Pregnancies = float(st.text_input('Number of Pregnancies', '0'))
        Glucose = float(st.text_input('Glucose Level', '0'))
        BloodPressure = float(st.text_input('Blood Pressure value', '0'))
        SkinThickness = float(st.text_input('Skin Thickness value', '0'))
        Insulin = float(st.text_input('Insulin Level', '0'))
        BMI = float(st.text_input('BMI value', '0'))
        DiabetesPedigreeFunction = float(st.text_input('Diabetes Pedigree Function value', '0'))
        Age = float(st.text_input('Age of the Person', '0'))
    except ValueError:
        st.error("Please enter valid numeric values.")
        return

    # Code for prediction
    diagnosis = ''

    # Creating a button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([
            Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
        ])
        st.success(diagnosis)

if __name__ == '__main__':
    main()
