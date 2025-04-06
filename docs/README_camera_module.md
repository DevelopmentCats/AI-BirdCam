# Raspberry Pi Camera Module for Bird Classification

This module provides a camera interface for capturing and processing images on Raspberry Pi, specifically optimized for bird classification models.

## Features

- Camera initialization and configuration
- Frame capture from webcams (with emphasis on Logitech 1080p Webcam)
- Image processing for bird classification models
  - Resizing to 224x224 pixels
  - RGB color format
  - Pixel normalization (0-1 range)
- Error handling for common camera issues
- Image saving with proper formatting

## Requirements

- Raspberry Pi (3B+ or 4 recommended)
- Python 3.6+
- OpenCV
- NumPy
- PIL (Pillow)
- Compatible webcam (Logitech 1080p Webcam recommended)

## Installation

1. Ensure you have the required dependencies:

```bash
sudo apt-get update
sudo apt-get install -y python3-opencv python3-pip
pip3 install numpy pillow matplotlib
```

2. Connect your webcam to the Raspberry Pi

3. Copy the `camera_module.py` file to your project directory

## Usage

### Basic Usage

```python
from camera_module import CameraModule

# Initialize camera
camera = CameraModule()
camera.initialize()

try:
    # Capture and process a frame
    processed_frame = camera.capture_and_process()
    
    # Save the processed image
    camera.save_image(processed_frame, "bird_image.jpg")
    
finally:
    # Always release camera resources
    camera.release()
```

### Configuration Options

The `CameraModule` class accepts several parameters:

- `camera_index` (int): Index of the camera device (default: 0)
- `resolution` (tuple): Desired camera resolution (width, height) (default: 1920x1080)
- `frame_rate` (int): Desired frame rate (default: 30)
- `target_size` (tuple): Target size for processed images (default: 224x224)
- `auto_focus` (bool): Enable/disable auto-focus if supported (default: True)

Example with custom settings:

```python
camera = CameraModule(
    camera_index=0,
    resolution=(1280, 720),
    frame_rate=30,
    target_size=(224, 224),
    auto_focus=True
)
```

### Testing

A test script is provided to verify the camera module functionality:

```bash
python3 test_camera_module.py
```

This will:
1. Initialize the camera
2. Capture a frame
3. Process the frame for model input
4. Save both original and processed images
5. Create a visual comparison

## API Reference

### CameraModule Class

#### Methods

- `initialize()`: Initialize the camera with specified parameters
- `capture_frame()`: Capture a single frame from the camera
- `process_image(image, normalize=True)`: Process image for bird classification models
- `capture_and_process()`: Capture a frame and process it for model input
- `save_image(image, file_path, original_format=False)`: Save image to disk
- `get_camera_info()`: Get information about the camera
- `release()`: Release camera resources

## Troubleshooting

### Camera Not Found

If the camera initialization fails:

1. Check if the camera is properly connected
2. Verify the camera is recognized by the system:
   ```bash
   ls -l /dev/video*
   ```
3. Try a different USB port
4. Ensure you have permission to access the camera:
   ```bash
   sudo usermod -a -G video $USER
   ```

### Image Quality Issues

If captured images have poor quality:

1. Ensure adequate lighting
2. Try different resolution settings
3. Check if auto-focus is working properly
4. Clean the camera lens

### Resource Issues

If you encounter resource or memory issues:

1. Release the camera when not in use
2. Lower the resolution if necessary
3. Ensure you're not creating multiple camera instances

## Integration with Bird Classification

This camera module is designed to work with the bird classification models specified in the project. The processed images (224x224 pixels, RGB format, normalized to 0-1 range) are ready to be fed directly into the classification models.