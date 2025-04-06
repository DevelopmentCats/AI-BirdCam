"""
Bird Classification Module for Raspberry Pi

This module provides functionality to:
1. Load a TensorFlow Lite bird classification model
2. Process images from the camera module
3. Classify bird species with confidence scores
4. Handle model downloading if not present locally

Compatible with Google AIY Vision Classifier (Birds V1) and similar TensorFlow Lite models.
"""

import os
import numpy as np
import time
import urllib.request
import logging
from typing import List, Tuple, Dict, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global flag to track TensorFlow availability
TENSORFLOW_AVAILABLE = False

# Try importing TensorFlow Lite
try:
    import tflite_runtime.interpreter as tflite
    logger.info("Using TensorFlow Lite Runtime for inference")
    TENSORFLOW_AVAILABLE = True
except ImportError:
    try:
        import tensorflow as tf
        logger.info("Using TensorFlow for inference")
        tflite = tf.lite
        TENSORFLOW_AVAILABLE = True
    except ImportError:
        logger.error("Failed to import TensorFlow or TensorFlow Lite. Some functionality will be limited.")
        # Define a placeholder for the tflite module to avoid errors
        class PlaceholderTFLite:
            class Interpreter:
                def __init__(self, model_path=None):
                    raise ImportError("TensorFlow or TensorFlow Lite is required for model inference")
        tflite = PlaceholderTFLite()

class BirdClassifier:
    """Bird species classifier using TensorFlow Lite models."""
    
    # Default model URLs and paths
    DEFAULT_MODEL_URL = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/image_classification/android/mobilenet_v1_1.0_224_quantized_1_metadata_1.tflite"
    DEFAULT_LABELS_URL = "https://raw.githubusercontent.com/tensorflow/examples/master/lite/examples/image_classification/android/app/src/main/assets/labels_mobilenet_quant_v1_224.txt"
    
    def __init__(self, 
                 model_path: str = "bird_model.tflite", 
                 labels_path: str = "bird_labels.txt",
                 model_url: str = None,
                 labels_url: str = None,
                 top_k: int = 5,
                 confidence_threshold: float = 0.1):
        """
        Initialize the bird classifier.
        
        Args:
            model_path: Path to the TensorFlow Lite model file
            labels_path: Path to the labels file
            model_url: URL to download the model if not found locally
            labels_url: URL to download the labels if not found locally
            top_k: Number of top predictions to return
            confidence_threshold: Minimum confidence score to consider a valid prediction
        """
        self.model_path = model_path
        self.labels_path = labels_path
        self.model_url = model_url if model_url else self.DEFAULT_MODEL_URL
        self.labels_url = labels_url if labels_url else self.DEFAULT_LABELS_URL
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold
        
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.labels = None
        self.input_shape = None
        
        # Initialize the model if TensorFlow is available
        if TENSORFLOW_AVAILABLE:
            try:
                self._ensure_model_available()
                self._load_model()
                self._load_labels()
            except Exception as e:
                logger.error(f"Error during initialization: {e}")
        else:
            logger.warning("TensorFlow not available. Model loading and inference will not work.")
            # Set default input shape for testing purposes
            self.input_shape = [1, 224, 224, 3]
        
    def _ensure_model_available(self) -> None:
        """Ensure the model and labels files are available, downloading if necessary."""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Cannot ensure model availability.")
            return
            
        # Check for model file
        if not os.path.exists(self.model_path):
            logger.info(f"Model file not found at {self.model_path}. Downloading...")
            try:
                urllib.request.urlretrieve(self.model_url, self.model_path)
                logger.info(f"Model downloaded successfully to {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to download model: {e}")
                raise
        
        # Check for labels file
        if not os.path.exists(self.labels_path):
            logger.info(f"Labels file not found at {self.labels_path}. Downloading...")
            try:
                urllib.request.urlretrieve(self.labels_url, self.labels_path)
                logger.info(f"Labels downloaded successfully to {self.labels_path}")
            except Exception as e:
                logger.error(f"Failed to download labels: {e}")
                raise
    
    def _load_model(self) -> None:
        """Load the TensorFlow Lite model."""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Cannot load model.")
            return
            
        try:
            # Load the TensorFlow Lite model
            self.interpreter = tflite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Get input shape
            self.input_shape = self.input_details[0]['shape']
            logger.info(f"Model loaded successfully. Input shape: {self.input_shape}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_labels(self) -> None:
        """Load the labels file."""
        try:
            if os.path.exists(self.labels_path):
                with open(self.labels_path, 'r') as f:
                    self.labels = [line.strip() for line in f.readlines()]
                logger.info(f"Loaded {len(self.labels)} labels")
            else:
                logger.warning(f"Labels file not found at {self.labels_path}")
                self.labels = [f"Class {i}" for i in range(1000)]  # Default placeholder labels
        except Exception as e:
            logger.error(f"Failed to load labels: {e}")
            self.labels = [f"Class {i}" for i in range(1000)]  # Default placeholder labels
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image for model input.
        
        Args:
            image: RGB image as numpy array (height, width, 3)
            
        Returns:
            Preprocessed image ready for model input
        """
        # Ensure image is in RGB format
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be RGB with shape (height, width, 3)")
        
        # Get target dimensions from input shape or use default
        if self.input_shape is not None:
            target_height, target_width = self.input_shape[1], self.input_shape[2]
        else:
            target_height, target_width = 224, 224  # Default for most models
        
        # Check if resizing is needed
        if image.shape[0] != target_height or image.shape[1] != target_width:
            try:
                import cv2
                image = cv2.resize(image, (target_width, target_height))
            except ImportError:
                from PIL import Image
                image = np.array(Image.fromarray(image).resize((target_width, target_height)))
        
        # Normalize pixel values based on model requirements
        # If we don't have model details, default to [0, 1] normalization
        if not TENSORFLOW_AVAILABLE or self.input_details is None:
            image = image.astype(np.float32) / 255.0
        else:
            # Check the quantization parameter to determine normalization method
            if self.input_details[0]['dtype'] == np.uint8:
                # Quantized model (uint8)
                image = image.astype(np.uint8)
            else:
                # Float model, normalize to [0, 1]
                image = image.astype(np.float32) / 255.0
                
                # Check if the model expects [-1, 1] range
                if self.input_details[0]['quantization'][0] == 0 and self.input_details[0]['quantization'][1] == 0:
                    image = image * 2.0 - 1.0
        
        # Add batch dimension if needed
        if len(self.input_shape) == 4:
            image = np.expand_dims(image, axis=0)
            
        return image
    
    def classify_image(self, image: np.ndarray) -> List[Dict[str, Union[str, float]]]:
        """
        Classify an image and return top predictions with confidence scores.
        
        Args:
            image: RGB image as numpy array (height, width, 3)
            
        Returns:
            List of dictionaries with species name and confidence score
        """
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow not available. Cannot perform classification.")
            return [{"species": "TensorFlow not available", "confidence": 0.0, "rank": 1}]
            
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(image)
            
            # Set the input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], processed_image)
            
            # Run inference
            start_time = time.time()
            self.interpreter.invoke()
            inference_time = time.time() - start_time
            
            # Get the output tensor
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Process the results
            results = self._process_output(output_data)
            
            logger.info(f"Inference completed in {inference_time*1000:.2f}ms")
            return results
        
        except Exception as e:
            logger.error(f"Error during classification: {e}")
            return []
    
    def _process_output(self, output_data: np.ndarray) -> List[Dict[str, Union[str, float]]]:
        """
        Process the model output to get top-k predictions.
        
        Args:
            output_data: Raw output from the model
            
        Returns:
            List of dictionaries with species name and confidence score
        """
        # Get the output shape
        output = np.squeeze(output_data)
        
        # Check if output is a probability distribution
        if output.sum() > 0 and abs(output.sum() - 1.0) > 0.1:
            # Apply softmax if needed
            exp_output = np.exp(output - np.max(output))
            output = exp_output / exp_output.sum()
        
        # Get top-k indices and scores
        top_indices = output.argsort()[-self.top_k:][::-1]
        top_scores = output[top_indices]
        
        # Create results list
        results = []
        for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
            if score >= self.confidence_threshold:
                species_name = self.labels[idx] if self.labels and idx < len(self.labels) else f"Unknown ({idx})"
                results.append({
                    "species": species_name,
                    "confidence": float(score),
                    "rank": i + 1
                })
        
        return results
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_path": self.model_path,
            "labels_path": self.labels_path,
            "input_shape": self.input_shape,
            "num_classes": len(self.labels) if self.labels else 0,
            "top_k": self.top_k,
            "confidence_threshold": self.confidence_threshold,
            "tensorflow_available": TENSORFLOW_AVAILABLE
        }

# Utility functions
def download_model(url: str, save_path: str) -> bool:
    """
    Download a model from a URL.
    
    Args:
        url: URL to download from
        save_path: Path to save the downloaded file
        
    Returns:
        True if download was successful, False otherwise
    """
    try:
        logger.info(f"Downloading model from {url} to {save_path}")
        urllib.request.urlretrieve(url, save_path)
        return True
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return False

def get_available_models() -> List[Dict[str, str]]:
    """
    Get a list of available bird classification models.
    
    Returns:
        List of dictionaries with model information
    """
    # This is a placeholder for a more comprehensive model registry
    return [
        {
            "name": "Google AIY Vision Classifier (Birds V1)",
            "url": "https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/image_classification/android/mobilenet_v1_1.0_224_quantized_1_metadata_1.tflite",
            "labels_url": "https://raw.githubusercontent.com/tensorflow/examples/master/lite/examples/image_classification/android/app/src/main/assets/labels_mobilenet_quant_v1_224.txt",
            "description": "Bird classification model based on MobileNet V1"
        },
        {
            "name": "MobileNet V2 Bird Classification",
            "url": "https://tfhub.dev/google/lite-model/aiy/vision/classifier/birds_V1/3",
            "labels_url": "https://www.gstatic.com/aihub/tfhub/labelmaps/aiy_birds_V1_labelmap.csv",
            "description": "Bird classification model based on MobileNet V2"
        }
    ]