#!/usr/bin/env python3
"""
Test script for camera_module.py
--------------------------------
This script tests the functionality of the CameraModule class
for capturing and processing images on Raspberry Pi.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from camera_module import CameraModule

def test_camera_initialization():
    """Test camera initialization and properties"""
    print("\n=== Testing Camera Initialization ===")
    camera = CameraModule(resolution=(1280, 720), frame_rate=30)
    
    success = camera.initialize()
    if success:
        print("✓ Camera initialized successfully")
        info = camera.get_camera_info()
        print(f"Camera info: {info}")
    else:
        print("✗ Camera initialization failed")
    
    camera.release()
    return success

def test_frame_capture(camera):
    """Test frame capture functionality"""
    print("\n=== Testing Frame Capture ===")
    
    # Capture a frame
    frame = camera.capture_frame()
    
    if frame is not None:
        print(f"✓ Frame captured successfully with shape {frame.shape}")
        return frame
    else:
        print("✗ Frame capture failed")
        return None

def test_image_processing(camera, frame):
    """Test image processing functionality"""
    print("\n=== Testing Image Processing ===")
    
    if frame is None:
        print("✗ No frame to process")
        return None
    
    # Process the frame
    processed = camera.process_image(frame)
    
    if processed is not None:
        print(f"✓ Image processed successfully")
        print(f"  - Shape: {processed.shape}")
        print(f"  - Data type: {processed.dtype}")
        print(f"  - Value range: [{processed.min():.4f}, {processed.max():.4f}]")
        return processed
    else:
        print("✗ Image processing failed")
        return None

def test_image_saving(camera, frame, processed_frame):
    """Test image saving functionality"""
    print("\n=== Testing Image Saving ===")
    
    # Create output directory
    os.makedirs("test_output", exist_ok=True)
    
    # Save original frame
    if frame is not None:
        success = camera.save_image(
            frame, 
            "test_output/original_frame.jpg", 
            original_format=True
        )
        print(f"✓ Original frame saved: {success}")
    
    # Save processed frame
    if processed_frame is not None:
        success = camera.save_image(
            processed_frame, 
            "test_output/processed_frame.jpg", 
            original_format=False
        )
        print(f"✓ Processed frame saved: {success}")

def visualize_results(frame, processed_frame):
    """Visualize original and processed frames"""
    print("\n=== Visualizing Results ===")
    
    if frame is None or processed_frame is None:
        print("✗ Cannot visualize: missing frames")
        return
    
    try:
        plt.figure(figsize=(10, 5))
        
        # Original frame
        plt.subplot(1, 2, 1)
        plt.imshow(frame)
        plt.title("Original Frame")
        plt.axis('off')
        
        # Processed frame
        plt.subplot(1, 2, 2)
        plt.imshow(processed_frame)
        plt.title("Processed Frame (224x224)")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig("test_output/comparison.png")
        print("✓ Visualization saved to test_output/comparison.png")
        
    except Exception as e:
        print(f"✗ Visualization failed: {str(e)}")

def run_full_test():
    """Run a complete test of the camera module"""
    print("=== CAMERA MODULE TEST ===")
    
    # Test initialization
    camera = CameraModule(resolution=(1280, 720), frame_rate=30)
    if not camera.initialize():
        print("Camera initialization failed. Exiting tests.")
        return False
    
    try:
        # Test frame capture
        frame = test_frame_capture(camera)
        
        # Test image processing
        processed_frame = test_image_processing(camera, frame)
        
        # Test image saving
        test_image_saving(camera, frame, processed_frame)
        
        # Visualize results
        visualize_results(frame, processed_frame)
        
        print("\n=== Test Summary ===")
        print("All tests completed. Check the test_output directory for results.")
        return True
        
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        return False
        
    finally:
        # Always release camera resources
        camera.release()
        print("Camera resources released")

if __name__ == "__main__":
    run_full_test()