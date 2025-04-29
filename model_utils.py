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
            compile=False  # Don't compile to avoid issues with custom losses
        )
        
        # Ensure the model is compiled with standard settings
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',  # Changed to binary_crossentropy for 2-class problem
            metrics=['accuracy']
        )
        
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
    try:
        # Ensure the input is in float32 format
        processed_image = processed_image.astype('float32')
        
        # Make prediction with error handling
        prediction = model.predict(processed_image, verbose=0)
        
        # Handle different prediction shapes
        if len(prediction.shape) > 1:
            confidence_scores = prediction[0]
        else:
            confidence_scores = prediction

        # Ensure we have the right number of confidence scores
        if len(confidence_scores) != len(class_names):
            # If model output doesn't match class names, adjust the scores
            confidence_scores = np.array([1 - confidence_scores[0], confidence_scores[0]])
        
        # Get the index of the class with highest probability
        predicted_class_index = np.argmax(confidence_scores)
        
        # Get the name of the predicted class
        if predicted_class_index < len(class_names):
            predicted_class_name = class_names[predicted_class_index]
        else:
            predicted_class_name = f"Unknown Class {predicted_class_index}"
        
        return predicted_class_name, confidence_scores
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        # Return safe default values
        return "Error in prediction", np.array([0.0] * len(class_names))