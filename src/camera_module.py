#!/usr/bin/env python3
"""
Camera Module for Raspberry Pi
------------------------------
This module handles webcam initialization, frame capture, and basic image processing
for bird classification on Raspberry Pi.

Compatible with various webcams, with emphasis on Logitech 1080p Webcam.
"""

import cv2
import numpy as np
import os
import time
from PIL import Image
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('camera_module')

class CameraModule:
    """
    Camera interface for capturing and processing images from webcams on Raspberry Pi.
    Optimized for bird classification models requiring 224x224 RGB images.
    """
    
    def __init__(self, 
                 camera_index=0, 
                 resolution=(1920, 1080), 
                 frame_rate=30,
                 target_size=(224, 224),
                 auto_focus=True):
        """
        Initialize the camera module with specified parameters.
        
        Args:
            camera_index (int): Index of the camera device (default: 0)
            resolution (tuple): Desired camera resolution (width, height) (default: 1920x1080)
            frame_rate (int): Desired frame rate (default: 30)
            target_size (tuple): Target size for processed images (default: 224x224)
            auto_focus (bool): Enable/disable auto-focus if supported (default: True)
        """
        self.camera_index = camera_index
        self.resolution = resolution
        self.frame_rate = frame_rate
        self.target_size = target_size
        self.auto_focus = auto_focus
        self.camera = None
        self.is_initialized = False
        
    def initialize(self):
        """
        Initialize the camera with specified parameters.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            logger.info(f"Initializing camera (index: {self.camera_index})")
            self.camera = cv2.VideoCapture(self.camera_index)
            
            if not self.camera.isOpened():
                logger.error("Failed to open camera")
                return False
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.camera.set(cv2.CAP_PROP_FPS, self.frame_rate)
            
            # Enable auto-focus if supported and requested
            if self.auto_focus:
                self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            
            # Verify settings were applied
            actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Camera initialized with resolution: {actual_width}x{actual_height}, FPS: {actual_fps}")
            
            # Warm up the camera
            for _ in range(5):
                ret, _ = self.camera.read()
                if not ret:
                    logger.warning("Camera warm-up frame capture failed")
                time.sleep(0.1)
                
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Camera initialization error: {str(e)}")
            self.release()
            return False
    
    def capture_frame(self):
        """
        Capture a single frame from the camera.
        
        Returns:
            numpy.ndarray: Captured frame or None if capture failed
        """
        if not self.is_initialized:
            logger.error("Camera not initialized. Call initialize() first")
            return None
        
        try:
            ret, frame = self.camera.read()
            if not ret:
                logger.error("Failed to capture frame")
                return None
            
            # Convert from BGR to RGB (OpenCV uses BGR by default)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame_rgb
            
        except Exception as e:
            logger.error(f"Frame capture error: {str(e)}")
            return None
    
    def process_image(self, image, normalize=True):
        """
        Process image for bird classification models.
        
        Args:
            image (numpy.ndarray): Input image in RGB format
            normalize (bool): Whether to normalize pixel values to [0,1]
            
        Returns:
            numpy.ndarray: Processed image ready for model input
        """
        try:
            # Resize to target size
            resized_image = cv2.resize(image, self.target_size)
            
            # Normalize pixel values if requested
            if normalize:
                processed_image = resized_image.astype(np.float32) / 255.0
            else:
                processed_image = resized_image
                
            return processed_image
            
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}")
            return None
    
    def capture_and_process(self):
        """
        Capture a frame and process it for model input.
        
        Returns:
            numpy.ndarray: Processed image ready for model input or None if failed
        """
        frame = self.capture_frame()
        if frame is not None:
            return self.process_image(frame)
        return None
    
    def save_image(self, image, file_path, original_format=False):
        """
        Save image to disk.
        
        Args:
            image (numpy.ndarray): Image to save
            file_path (str): Path to save the image
            original_format (bool): If True, save the image as-is; 
                                   if False, convert to model input format first
                                   
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            if not original_format:
                # Process image before saving
                processed_image = self.process_image(image, normalize=False)
                
                # Convert to PIL Image and save
                pil_image = Image.fromarray(processed_image.astype('uint8'))
                pil_image.save(file_path)
            else:
                # For OpenCV, convert RGB back to BGR before saving
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(file_path, image_bgr)
                else:
                    cv2.imwrite(file_path, image)
                    
            logger.info(f"Image saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving image: {str(e)}")
            return False
    
    def get_camera_info(self):
        """
        Get information about the camera.
        
        Returns:
            dict: Camera information
        """
        if not self.is_initialized:
            return {"status": "Not initialized"}
        
        info = {
            "index": self.camera_index,
            "resolution": {
                "width": self.camera.get(cv2.CAP_PROP_FRAME_WIDTH),
                "height": self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            },
            "fps": self.camera.get(cv2.CAP_PROP_FPS),
            "target_size": self.target_size
        }
        return info
    
    def release(self):
        """
        Release camera resources.
        """
        if self.camera is not None:
            self.camera.release()
            self.camera = None
            self.is_initialized = False
            logger.info("Camera resources released")


# Example usage
if __name__ == "__main__":
    # Initialize camera
    camera = CameraModule(camera_index=0, resolution=(1280, 720))
    
    if camera.initialize():
        try:
            # Capture and process a frame
            processed_frame = camera.capture_and_process()
            
            if processed_frame is not None:
                print(f"Captured frame shape: {processed_frame.shape}")
                print(f"Pixel value range: {processed_frame.min()} to {processed_frame.max()}")
                
                # Save the processed image
                camera.save_image(processed_frame, "captured_image.jpg")
                
                # Print camera info
                print("Camera info:", camera.get_camera_info())
            else:
                print("Failed to capture and process frame")
                
        finally:
            # Always release camera resources
            camera.release()
    else:
        print("Failed to initialize camera")