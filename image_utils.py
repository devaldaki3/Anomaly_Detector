import cv2
import numpy as np
import io
from PIL import Image

def preprocess_image(image_bytes, target_size=(224, 224)):
    """
    Preprocess image bytes to match the format expected by Teachable Machine models.
    
    Args:
        image_bytes (bytes): Raw image bytes
        target_size (tuple): Target size for the model input (default: 224x224 for Teachable Machine)
        
    Returns:
        numpy.ndarray: Preprocessed image in the format ready for model inference
    """
    try:
        # Convert to numpy array from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Resize image to target size
        image = Image.fromarray(image_array)
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        image_array = np.array(image)
        
        # Normalize pixel values to [0, 1]
        normalized_image = image_array.astype(np.float32) / 255.0
        
        # Ensure shape is (1, height, width, channels)
        if len(normalized_image.shape) == 3:
            normalized_image = np.expand_dims(normalized_image, axis=0)
        
        return normalized_image
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        # Return a blank image in case of error
        return np.zeros((1, target_size[0], target_size[1], 3), dtype=np.float32)