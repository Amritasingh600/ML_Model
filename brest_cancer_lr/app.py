import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# ---------------------------
# Load your model
# ---------------------------
model = load_model("best_model.keras")  # path to your saved model

# ---------------------------
# Config
# ---------------------------
img_size = (224, 224)
class_names = ['covid', 'normal', 'pneumonia']
confidence_threshold = 0.8  # minimum confidence to trust prediction
use_augmentation = True     # average prediction over flipped/rotated images

# ---------------------------
# Image preprocessing
# ---------------------------
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image / 255.0
    image = np.stack((image,) * 3, axis=-1)
    image = cv2.resize(image, img_size)
    return image

def predict_with_augmentations(image):
    images = [image]
    if use_augmentation:
        images.append(cv2.flip(image, 1))  # horizontal flip
        images.append(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))  # rotate 90

    preds = [model.predict(np.expand_dims(img, axis=0))[0] for img in images]
    avg_pred = np.mean(preds, axis=0)
    return avg_pred

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("COVID-19 X-ray Classifier")

uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file:
    # Read image
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(img, channels="BGR", caption="Uploaded X-ray")

    # Preprocess
    processed_img = preprocess_image(img)

    # Predict
    pred_probs = predict_with_augmentations(processed_img)
    pred_class = class_names[np.argmax(pred_probs)]
    confidence = np.max(pred_probs)

    # Show result
    if confidence < confidence_threshold:
        st.warning(f"Prediction not confident ({confidence*100:.2f}%). Please review manually.")
    else:
        st.success(f"Predicted Class: {pred_class} ({confidence*100:.2f}% confidence)")
