import tensorflow as tf
import numpy as np
import os

def load_model(model_path):
    """
    Load the Keras model from the given path.
    
    Args:
        model_path (str): Path to the keras_model.h5 file
        
    Returns:
        model: TensorFlow model
    """
    # Load the model with proper configuration to avoid compatibility issues
    try:
        model = tf.keras.models.load_model(
            model_path,
            compile=False,  # Don't compile to avoid issues with custom losses
            custom_objects=None  # No custom objects for Teachable Machine models
        )
        
        # Ensure the model is compiled with standard settings
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Warmup inference (first prediction can be slow)
        warmup_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
        _ = model.predict(warmup_input)
        
        return model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_image(model, processed_image, class_names):
    """
    Predict the class of the processed image using the model.
    
    Args:
        model: TensorFlow model
        processed_image (numpy.ndarray): Preprocessed image array
        class_names (list): List of class names
        
    Returns:
        tuple: (predicted_class_name, confidence_scores)
    """
    # Make prediction
    prediction = model.predict(processed_image)
    confidence_scores = prediction[0]
    
    # Get the index of the class with highest probability
    predicted_class_index = np.argmax(confidence_scores)
    
    # Get the name of the predicted class
    if predicted_class_index < len(class_names):
        predicted_class_name = class_names[predicted_class_index]
    else:
        predicted_class_name = f"Unknown Class {predicted_class_index}"
    
    return predicted_class_name, confidence_scores