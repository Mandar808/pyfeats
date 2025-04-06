import streamlit as st
import cv2
import numpy as np
from PIL import Image
from joblib import load
import os
import sys

# Add pyfeats path for local import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from pyfeats import zernikes_moments

# Load model and label encoder
model = load("zernike_shape_svm_model.joblib")
label_encoder = load("label_encoder.joblib")

st.set_page_config(page_title="Shape Classifier", layout="centered")
st.title("🔵 Shape Classifier using Zernike Moments")
st.write("Upload an image of a shape (Circle, Square, Triangle) and get the predicted class!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image_np = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    image_resized = cv2.resize(image_np, (128, 128))
    _, binary = cv2.threshold(image_resized, 127, 255, cv2.THRESH_BINARY)

    try:
        feats, _ = zernikes_moments(binary, radius=64)
    except Exception as e:
        st.error(f"Error extracting Zernike moments: {e}")
        st.stop()

    # Predict
    pred_proba = model.predict_proba([feats])[0]
    pred_class_idx = np.argmax(pred_proba)
    pred_class = label_encoder.inverse_transform([pred_class_idx])[0]
    confidence = pred_proba[pred_class_idx] * 100

    st.success(f"✅ Predicted Shape: **{pred_class.capitalize()}**")
    st.info(f"📊 Confidence: **{confidence:.2f}%**")

    # Show probability chart
    st.subheader("Class Probabilities")
    st.bar_chart({label_encoder.inverse_transform([i])[0]: prob for i, prob in enumerate(pred_proba)})
