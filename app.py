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
    .subheader {
        font-size: 1.5rem;
        color: #4e8df5;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .confidence-meter {
        height: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
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
    st.image("assets/overview_dataset.jpg", use_column_width=True)
    st.markdown("### Anomaly Detection Tool")
    st.markdown("---")
    st.markdown("### How to use:")
    st.markdown("1. Upload an image using the file uploader")
    st.markdown("2. View the model's prediction")
    st.markdown("3. See confidence scores for each class")
    st.markdown("---")
    st.markdown("### Model Info:")
    
    model_path = "model/keras_model.h5"
    labels_path = "model/labels.txt"
    
    if os.path.exists(model_path) and os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            class_names = [line.strip().split(' ', 1)[1] for line in f.readlines()]
        st.success(f"Model loaded with {len(class_names)} classes")
        st.markdown(f"**Classes**: {', '.join(class_names)}")
    else:
        st.error("Model files not found. Please add keras_model.h5 and labels.txt to the model folder.")
        class_names = ["Class not available"]

# Main content
st.markdown("<h1 class='main-header'>MicroCheck Anomaly Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image to detect anomalies using AI</p>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Process uploaded image
if uploaded_file is not None:
    # Convert the file to an image
    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes))
    
    # Display original image
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<p class='subheader'>Original Image</p>", unsafe_allow_html=True)
        st.image(image, use_column_width=True)
    
    # Model prediction section
    with col2:
        st.markdown("<p class='subheader'>Detection Results</p>", unsafe_allow_html=True)
        
        if os.path.exists(model_path) and os.path.exists(labels_path):
            # Load model
            model = load_model(model_path)
            
            # Preprocess image for model
            processed_image = preprocess_image(image_bytes)
            
            # Predict
            prediction, confidence_scores = predict_image(model, processed_image, class_names)
            
            # Display prediction result
            st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
            st.markdown(f"### Prediction: {prediction}")
            st.markdown("### Confidence Scores:")
            
            # Display confidence bars for all classes
            for i, (class_name, score) in enumerate(zip(class_names, confidence_scores)):
                score_percentage = float(score * 100)
                color = "#4CAF50" if score == max(confidence_scores) else "#9E9E9E"
                st.markdown(f"**{class_name}**: {score_percentage:.2f}%")
                st.markdown(
                    f"""<div class="confidence-meter" style="width: {score_percentage}%; background-color: {color};"></div>""",
                    unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error("Model not found. Please upload the model files first.")

    # Additional analysis
    st.markdown("### Analysis Details")
    with st.expander("See technical details"):
        st.write(f"Filename: {uploaded_file.name}")
        st.write(f"Image size: {image.size}")
        st.write(f"Image format: {image.format}")
        st.write("Note: The model is trained to detect specific anomalies based on the training data provided in Teachable Machine.")

# Footer
st.markdown("<div class='footer'>MicroCheck | Powered by TensorFlow and Streamlit | Created for educational purposes</div>", unsafe_allow_html=True)