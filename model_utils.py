import tensorflow as tf
import numpy as np
import os
import json

def custom_objects():
    """
    Create custom objects dictionary to handle model loading across different TF versions
    """
    try:
        # Custom DepthwiseConv2D layer that ignores 'groups' parameter
        class CompatibleDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
            def __init__(self, **kwargs):
                # Remove 'groups' from kwargs if present
                kwargs.pop('groups', None)
                super().__init__(**kwargs)
        
        return {
            'DepthwiseConv2D': CompatibleDepthwiseConv2D
        }
    except Exception as e:
        print(f"Error creating custom objects: {str(e)}")
        return {}

def load_model(model_path):
    """
    Load the Keras model from the given path.
    
    Args:
        model_path (str): Path to the keras_model.h5 file
        
    Returns:
        model: TensorFlow model
    """
    try:
        print(f"Loading model from: {model_path}")
        print(f"TensorFlow version: {tf.__version__}")
        
        # Load the model with custom objects
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects=custom_objects()
        )
        
        # Print model summary for debugging
        print("Model loaded successfully. Model summary:")
        model.summary()
        
        # Configure model for inference
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Verify model is loaded correctly
        if model is None:
            raise ValueError("Model failed to load")
            
        return model
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise e

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
        print("Starting prediction...")
        print(f"Input shape: {processed_image.shape}")
        print(f"Input dtype: {processed_image.dtype}")
        print(f"Number of classes: {len(class_names)}")
        
        # Ensure model is not None
        if model is None:
            raise ValueError("Model is None")
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=1)
        print(f"Raw prediction shape: {prediction.shape}")
        print(f"Raw prediction values: {prediction}")
        
        # For Teachable Machine models, prediction is usually a single array
        confidence_scores = prediction[0] if len(prediction.shape) > 1 else prediction
        print(f"Confidence scores: {confidence_scores}")
        
        # Get the predicted class
        predicted_class_index = np.argmax(confidence_scores)
        predicted_class_name = class_names[predicted_class_index]
        
        # Format confidence scores as percentages
        confidence_scores = confidence_scores * 100
        
        print(f"Predicted class: {predicted_class_name}")
        print(f"Confidence scores (%): {confidence_scores}")
        
        return predicted_class_name, confidence_scores
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        print("Stack trace:", exc_info=True)
        # Return error state
        return "Error in prediction", np.zeros(len(class_names))