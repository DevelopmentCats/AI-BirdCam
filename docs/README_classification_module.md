# Bird Classification Module for Raspberry Pi

This module provides functionality for bird species classification using TensorFlow Lite models on Raspberry Pi.

## Features

- Loads TensorFlow Lite bird classification models
- Processes images from the camera module
- Classifies bird species with confidence scores
- Downloads models if not present locally
- Handles preprocessing specific to the selected model
- Provides a clean API for classification
- Includes functions to interpret model outputs
- Handles errors gracefully

## Requirements

- Raspberry Pi (3B+ or 4 recommended)
- Python 3.7+
- TensorFlow Lite or TensorFlow
- NumPy
- OpenCV (optional, for image processing)
- PIL/Pillow (as fallback for image processing)

## Installation

1. Ensure you have the required dependencies:

```bash
sudo apt-get update
sudo apt-get install -y python3-pip

# Install TensorFlow Lite (recommended for Raspberry Pi)
pip3 install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0-cp37-cp37m-linux_armv7l.whl

# Or install TensorFlow (if you have a 64-bit OS and more resources)
# pip3 install tensorflow

# Install other dependencies
pip3 install numpy pillow matplotlib
```

2. Copy the `classification_module.py` file to your project directory

## Usage

### Basic Usage

```python
from classification_module import BirdClassifier
import numpy as np

# Initialize the classifier
classifier = BirdClassifier(
    model_path="bird_model.tflite",
    labels_path="bird_labels.txt"
)

# Load an image (replace with your image loading code)
image = np.array(...)  # RGB image as numpy array

# Classify the image
results = classifier.classify_image(image)

# Process the results
for result in results:
    print(f"{result['species']}: {result['confidence']*100:.2f}%")
```

### Integration with Camera Module

```python
from camera_module import CameraModule
from classification_module import BirdClassifier

# Initialize camera
camera = CameraModule()
camera.initialize()

# Initialize classifier
classifier = BirdClassifier()

try:
    # Capture and process a frame
    frame = camera.capture_and_process()
    
    # Classify the frame
    results = classifier.classify_image(frame)
    
    # Process the results
    for result in results:
        print(f"{result['species']}: {result['confidence']*100:.2f}%")
        
finally:
    # Release camera resources
    camera.release()
```

## API Reference

### BirdClassifier Class

#### Constructor

```python
BirdClassifier(
    model_path="bird_model.tflite",
    labels_path="bird_labels.txt",
    model_url=None,
    labels_url=None,
    top_k=5,
    confidence_threshold=0.1
)
```

Parameters:
- `model_path`: Path to the TensorFlow Lite model file
- `labels_path`: Path to the labels file
- `model_url`: URL to download the model if not found locally
- `labels_url`: URL to download the labels if not found locally
- `top_k`: Number of top predictions to return
- `confidence_threshold`: Minimum confidence score to consider a valid prediction

#### Methods

- `classify_image(image)`: Classify an image and return top predictions with confidence scores
- `preprocess_image(image)`: Preprocess the image for model input
- `get_model_info()`: Get information about the loaded model

### Utility Functions

- `download_model(url, save_path)`: Download a model from a URL
- `get_available_models()`: Get a list of available bird classification models

## Example Scripts

### Test Classification Module

The `test_classification_module.py` script demonstrates how to use the BirdClassifier class with a sample image.

```bash
python3 test_classification_module.py --image bird.jpg
```

Options:
- `--image`: Path to a bird image for testing
- `--model`: Path to the TensorFlow Lite model
- `--labels`: Path to the labels file
- `--top_k`: Number of top predictions to return
- `--threshold`: Confidence threshold for predictions

### Bird Classification Example

The `bird_classification_example.py` script demonstrates how to integrate the camera module with the bird classification module.

```bash
python3 bird_classification_example.py --continuous --interval 2.0 --save_images
```

Options:
- `--camera`: Camera index
- `--resolution`: Camera resolution (WxH)
- `--fps`: Camera frame rate
- `--model`: Path to the model file
- `--labels`: Path to the labels file
- `--top_k`: Number of top predictions to return
- `--threshold`: Confidence threshold
- `--continuous`: Run in continuous mode
- `--interval`: Interval between captures in continuous mode (seconds)
- `--output_dir`: Directory to save captured images
- `--save_images`: Save captured images

## Troubleshooting

### Model Loading Issues

If the model fails to load:
1. Check if the model file exists at the specified path
2. Verify that the model is a valid TensorFlow Lite model
3. Ensure you have the correct version of TensorFlow Lite or TensorFlow installed

### Classification Issues

If classification results are poor:
1. Check if the image is properly preprocessed
2. Verify that the labels file matches the model
3. Try adjusting the confidence threshold
4. Ensure adequate lighting for the camera

### Memory Issues

If you encounter memory issues on Raspberry Pi:
1. Use TensorFlow Lite instead of full TensorFlow
2. Lower the camera resolution
3. Process frames at a lower frequency
4. Close other memory-intensive applications

## References

- [TensorFlow Lite Documentation](https://www.tensorflow.org/lite)
- [Google AIY Vision Models](https://aiyprojects.withgoogle.com/vision/)
- [MobileNet Models](https://www.tensorflow.org/lite/models/image_classification/overview)