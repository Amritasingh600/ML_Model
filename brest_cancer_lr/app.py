import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load('model.joblib')

st.title("Logistic Regression Predictor")

# Input fields for features (adjust number & names to your dataset)
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.number_input("Feature 3")
feature4 = st.number_input("Feature 4")

if st.button("Predict"):
    input_data = np.array([[feature1, feature2, feature3, feature4]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Class: {prediction[0]}")
