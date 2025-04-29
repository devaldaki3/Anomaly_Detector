import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import os
from model_utils import load_model, predict_image
from image_utils import preprocess_image

# Page configuration
st.set_page_config(
    page_title="MicroCheck - Anomaly Detection",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4e8df5;
        text-align: center;
    }
    .output-container {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    .output-header {
        color: #00FF9C;
        font-size: 1.2rem;
        margin-bottom: 20px;
    }
    .result-label {
        color: #ffffff;
        font-size: 1rem;
        margin-bottom: 5px;
    }
    .probability-bar {
        height: 6px;
        background-color: #2D7FF9;
        border-radius: 3px;
        margin: 10px 0;
        transition: width 0.5s ease-in-out;
    }
    .normal-dot {
        color: #00FF9C;
        font-size: 1.2rem;
    }
    .anomaly-dot {
        color: #FF4B4B;
        font-size: 1.2rem;
    }
    .percentage {
        font-size: 2rem;
        font-weight: bold;
        margin: 5px 0;
    }
    .alert-box {
        background-color: rgba(255, 75, 75, 0.1);
        border-left: 4px solid #FF4B4B;
        padding: 10px 15px;
        margin-top: 10px;
        border-radius: 0 5px 5px 0;
    }
    .alert-text {
        color: #FF4B4B;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .footer {
        text-align: center;
        color: #888888;
        font-size: 0.8rem;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>MicroCheck</h1>", unsafe_allow_html=True)
    st.image("assets/overview_dataset.jpg", width=None)
    st.markdown("### Anomaly Detection Tool")
    st.markdown("---")
    st.markdown("### How to use:")
    st.markdown("1. Choose input method (Upload or Camera)")
    st.markdown("2. Capture or select an image")
    st.markdown("3. View the detection results")
    st.markdown("---")
    st.markdown("### Model Info:")
    
    model_path = "model/keras_model.h5"
    labels_path = "model/labels.txt"
    
    if os.path.exists(model_path) and os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            class_names = [line.strip().split(' ', 1)[1] for line in f.readlines()]
        st.success(f"Model loaded with {len(class_names)} classes")
    else:
        st.error("Model files not found. Please add keras_model.h5 and labels.txt to the model folder.")
        class_names = ["Class not available"]

# Main content
st.markdown("<h1 class='main-header'>MicroCheck Anomaly Detection System</h1>", unsafe_allow_html=True)

# Input method selection
input_method = st.radio("Choose input method:", ["File Upload", "Camera Input"])

image_bytes = None
image = None

if input_method == "File Upload":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))
elif input_method == "Camera Input":
    camera_file = st.camera_input("Take a picture")
    if camera_file is not None:
        image_bytes = camera_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))

# Process image and show results
if image_bytes is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Input Image", use_column_width=True)
    
    with col2:
        if os.path.exists(model_path) and os.path.exists(labels_path):
            try:
                model = load_model(model_path)
                if model is None:
                    st.error("Failed to load model. Please check the model file.")
                    st.stop()
                
                processed_image = preprocess_image(image_bytes)
                if processed_image is None:
                    st.error("Failed to process image. Please try a different image.")
                    st.stop()
                
                prediction, confidence_scores = predict_image(model, processed_image, class_names)
                
                # New UI Output
                st.markdown("<div class='output-container'>", unsafe_allow_html=True)
                st.markdown("<div class='output-header'>Detection Results:</div>", unsafe_allow_html=True)
                
                # Display results for both classes
                normal_score = confidence_scores[0]
                anomaly_score = confidence_scores[1]
                
                # Normal probability
                st.markdown(f"<span class='normal-dot'>●</span> Normal", unsafe_allow_html=True)
                st.markdown(f"<div class='percentage'>{normal_score:.2f}%</div>", unsafe_allow_html=True)
                
                # Anomaly probability
                st.markdown(f"<span class='anomaly-dot'>●</span> Anomaly", unsafe_allow_html=True)
                st.markdown(f"<div class='percentage'>{anomaly_score:.2f}%</div>", unsafe_allow_html=True)
                
                # Probability bar
                st.markdown("<div class='result-label'>Anomaly Probability</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='probability-bar' style='width: {anomaly_score}%;'></div>", unsafe_allow_html=True)
                
                # Alert box if anomaly detected
                if anomaly_score > 50:
                    st.markdown("""
                        <div class='alert-box'>
                            <div class='alert-text'>
                                ⚠️ Anomaly Detected!
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("Please try again with a different image or check if the model is properly loaded.")
        else:
            st.error("Model not found. Please upload the model files first.")

# Footer
st.markdown("<div class='footer'>MicroCheck | Powered by TensorFlow and Streamlit | Created for educational purposes</div>", unsafe_allow_html=True)