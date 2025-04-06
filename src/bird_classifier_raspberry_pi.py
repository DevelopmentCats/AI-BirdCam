#!/usr/bin/env python3
"""
Bird Classification on Raspberry Pi using TensorFlow Lite
--------------------------------------------------------
This script demonstrates how to use a TensorFlow Lite bird classification model
on a Raspberry Pi. It includes code for image preprocessing, model loading,
and inference.

Requirements:
- TensorFlow Lite Runtime
- OpenCV
- NumPy
- Pillow (PIL)

Install dependencies:
pip3 install tflite-runtime opencv-python numpy pillow
"""

import os
import time
import numpy as np
import cv2
from PIL import Image

# Try to import TensorFlow Lite Interpreter
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    print("TFLite runtime not found. Trying to use full TensorFlow.")
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

def load_labels(label_path):
    """Load labels from file"""
    with open(label_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def preprocess_image(image_path, input_size=(224, 224)):
    """Preprocess the image to meet the model's requirements"""
    # Load image with PIL to handle various image formats
    image = Image.open(image_path).convert('RGB')
    
    # Resize the image
    image = image.resize(input_size)
    
    # Convert to numpy array
    image = np.array(image)
    
    # Normalize pixel values
    image = image.astype(np.float32) / 255.0
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image

def classify_image(interpreter, image, top_k=5):
    """Run inference and return top k results"""
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], image)
    
    # Run inference
    start_time = time.time()
    interpreter.invoke()
    inference_time = time.time() - start_time
    
    # Get output tensor
    output = interpreter.get_tensor(output_details[0]['index'])
    
    # Get top k results
    results = np.squeeze(output)
    top_indices = results.argsort()[-top_k:][::-1]
    
    return top_indices, results[top_indices], inference_time

def main():
    # Model and label paths - update these to your actual paths
    model_path = "bird_model.tflite"  # Path to your TFLite model
    label_path = "bird_labels.txt"    # Path to your labels file
    image_path = "test_bird.jpg"      # Path to test image
    
    # Check if model exists
    if not os.path.isfile(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        print("Please download a bird classification model using one of these options:")
        print("1. Google AIY Vision Classifier (Birds V1): https://www.kaggle.com/models/google/aiy/tensorFlow1/vision-classifier-birds-v1/1")
        print("2. GitHub repository: https://github.com/astrocoding/model-ai-bird-classification-tflite")
        return
    
    # Check if labels exist
    if not os.path.isfile(label_path):
        print(f"Error: Labels file '{label_path}' not found.")
        return
    
    # Check if test image exists
    if not os.path.isfile(image_path):
        print(f"Error: Test image '{image_path}' not found.")
        return
    
    # Load labels
    print("Loading labels...")
    labels = load_labels(label_path)
    
    # Load TFLite model
    print("Loading model...")
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Get input shape
    input_shape = input_details[0]['shape']
    input_size = (input_shape[2], input_shape[1])  # Width, Height
    
    print(f"Model loaded. Input size: {input_size}")
    
    # Preprocess image
    print(f"Processing image: {image_path}")
    image = preprocess_image(image_path, input_size)
    
    # Classify image
    print("Running inference...")
    top_indices, scores, inference_time = classify_image(interpreter, image)
    
    # Print results
    print(f"\nInference time: {inference_time*1000:.1f}ms")
    print("\nTop predictions:")
    for i, (idx, score) in enumerate(zip(top_indices, scores)):
        if i < len(labels):
            print(f"{i+1}. {labels[idx]}: {score*100:.1f}%")
        else:
            print(f"{i+1}. Unknown (index {idx}): {score*100:.1f}%")
    
    # Display image with prediction (if running with display)
    try:
        # Load image for display
        display_img = cv2.imread(image_path)
        # Add prediction text
        if len(labels) > top_indices[0]:
            label = f"{labels[top_indices[0]]}: {scores[0]*100:.1f}%"
        else:
            label = f"Unknown (index {top_indices[0]}): {scores[0]*100:.1f}%"
        
        cv2.putText(display_img, label, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Show image
        cv2.imshow("Bird Classification", display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Note: Could not display image. Running in headless mode or missing OpenCV display support.")
        print(f"Error details: {e}")

if __name__ == "__main__":
    main()