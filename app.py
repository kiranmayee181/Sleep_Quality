import streamlit as st
import numpy as np
import pickle
import os
import base64

# Load the trained model
pickle_file_path = 'rf.pkl'
with open(pickle_file_path, 'rb') as file:
    model = pickle.load(file)

st.title('Sleep Health Prediction')

st.write('Input the feature values:')

# Create input fields for features with validation
age = st.number_input('Age', min_value=0, max_value=120, value=0)
sleep_duration = st.number_input('Sleep Duration (hours)', min_value=0.0, max_value=24.0, value=0.0)
physical_activity_level = st.number_input('Physical Activity Level (min/day)', min_value=0, value=0)
stress_level = st.selectbox('Stress Level (0-10)', options=list(range(0, 11)))

if st.button('Predict'):
    try:
        # Make a prediction
        input_data = np.array([[age, sleep_duration, physical_activity_level, stress_level]])
        prediction = model.predict(input_data)
        rounded_prediction = int(round(prediction[0]))
        st.write(f'Sleep Quality: {rounded_prediction}')
    except Exception as e:
        st.error(f'Error making prediction: {e}')
