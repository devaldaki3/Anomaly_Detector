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
    # Convert to numpy array from bytes
    image_array = np.array(Image.open(io.BytesIO(image_bytes)))
    
    # Convert to BGR if needed (OpenCV format)
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    # Resize image to target size
    resized_image = cv2.resize(image_array, target_size)
    
    # Convert back to RGB for model input
    if len(resized_image.shape) == 3 and resized_image.shape[2] == 3:
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values to [0, 1]
    normalized_image = resized_image.astype(np.float32) / 255.0
    
    # Expand dimensions to create batch of size 1
    batched_image = np.expand_dims(normalized_image, axis=0)
    
    return batched_image