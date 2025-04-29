import cv2
import numpy as np
import io
from PIL import Image
import sys

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
        print("Starting image preprocessing...")
        print(f"Target size: {target_size}")
        
        # Convert to PIL Image first
        image = Image.open(io.BytesIO(image_bytes))
        print(f"Original image size: {image.size}")
        print(f"Original image mode: {image.mode}")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            print(f"Converting image from {image.mode} to RGB")
            image = image.convert('RGB')
        
        # Resize image to target size
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        print(f"Resized image size: {image.size}")
        
        # Convert to numpy array
        image_array = np.array(image)
        print(f"Numpy array shape: {image_array.shape}")
        
        # Normalize pixel values to [0, 1]
        normalized_image = image_array.astype(np.float32) / 255.0
        print(f"Value range: [{normalized_image.min():.3f}, {normalized_image.max():.3f}]")
        
        # Expand dimensions to create batch of size 1
        batched_image = np.expand_dims(normalized_image, axis=0)
        print(f"Final shape: {batched_image.shape}")
        print(f"Final dtype: {batched_image.dtype}")
        
        return batched_image
        
    except Exception as e:
        print(f"Error in image preprocessing: {str(e)}")
        print("Stack trace:", sys.exc_info())
        # Return None instead of zeros to trigger proper error handling
        return None