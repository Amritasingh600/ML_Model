import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from streamlit_lottie import st_lottie
import time
model = joblib.load("aircraft.pkl")
st.set_page_config(page_title="Aircraft Fuel App", layout="centered")
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #e0f7fa, #0000ff);
        font-family: 'Segoe UI', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_plane = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_zrqthn6o.json")
st_lottie(lottie_plane, speed=1, height=250, key="airplane")
st.markdown("<h2 style='text-align: center; color: #003366;'>Aircraft Fuel Consumption Predictor</h2>", unsafe_allow_html=True)
st.markdown("### 📥 Enter Flight Details")
col1, col2 = st.columns(2)
with col1:
    Flight_Distance = st.number_input("📏 Distance (km)", min_value=0.0)
    Number_of_Passenger = st.number_input("🧍‍♂️ Number of Passengers", min_value=1)

with col2:
    Flight_Duration = st.number_input("⏱️ Flight Duration (hrs)", min_value=0.0)
    Aircraft_Type = st.selectbox("🛩️ Aircraft Type", ["T1", "T2", "T3"])
input_data = pd.DataFrame({
    "Flight_Distance": [Flight_Distance],
    "Number_of_Passengers": [Number_of_Passenger],
    "Flight_Duration": [Flight_Duration],
    "Aircraft_Type_Type1": [1 if Aircraft_Type == "T1" else 0],
    "Aircraft_Type_Type2": [1 if Aircraft_Type == "T2" else 0],
    "Aircraft_Type_Type3": [1 if Aircraft_Type == "T3" else 0]
})
st.markdown("---")
if st.button("🚀 Predict Fuel Consumption"):
    with st.spinner("Calculating fuel requirement..."):
        time.sleep(2)
        prediction = model.predict(input_data)
        fuel_result = float(prediction[0])
        st.success(f"🛢️ Estimated Fuel Consumption: **{fuel_result:.2f} liters**")
        st.markdown("### ⛽ Fuel Gauge:")

st.markdown("---")
st.markdown("<div style='text-align: center;'>Made with ❤️ by <b>Amrita Singh</b></div>", unsafe_allow_html=True)
