# backend.py

import joblib  # or pickle
import numpy as np

# Load your trained model (only once)
model = joblib.load('model.pkl')

def predict_tumor(data_list):
    # Make sure input is in correct shape: (1, n_features)
    data_array = np.array(data_list).reshape(1, -1)
    prediction = model.predict(data_array)
    return prediction[0]  # Return 'Malignant' or 'Benign'
