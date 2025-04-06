#!/usr/bin/env python3
"""
Bird Classification System - Main Application

This script integrates camera capture, bird classification, and notification components
to create a complete bird classification and notification system.

The application captures images at regular intervals or based on motion detection,
classifies birds in the captured images, and sends notifications when birds are
detected with sufficient confidence.
"""

import os
import time
import logging
import argparse
import signal
import sys
from datetime import datetime
import cv2
import numpy as np

# Import modules
from camera_module import CameraModule
from classification_module import BirdClassifier
from notification_module import BirdNotifier

# Import configuration
from config import CAMERA_CONFIG, CLASSIFICATION_CONFIG, NOTIFICATION_CONFIG, APP_CONFIG

# Global flag for graceful shutdown
running = True

class BirdClassificationApp:
    """Main application class for the Bird Classification System."""
    
    def __init__(self):
        """Initialize the application components."""
        # Set up logging
        self._setup_logging()
        
        # Create required directories
        self._setup_directories()
        
        # Initialize components
        self.logger.info("Initializing Bird Classification System")
        self._init_camera()
        self._init_classifier()
        self._init_notifier()
        
        # Initialize motion detection variables
        self.prev_frame = None
        self.motion_detected = False
        
        # Statistics
        self.stats = {
            'images_captured': 0,
            'birds_detected': 0,
            'notifications_sent': 0,
            'start_time': datetime.now()
        }
        
        self.logger.info("Bird Classification System initialized successfully")
    
    def _setup_logging(self):
        """Set up logging configuration."""
        log_level = getattr(logging, APP_CONFIG['log_level'])
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(APP_CONFIG['log_file']),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('BirdClassifier')
        self.logger.info("Logging initialized")
    
    def _setup_directories(self):
        """Create required directories if they don't exist."""
        # Create directory for saving captured images
        os.makedirs(APP_CONFIG['image_save_dir'], exist_ok=True)
        
        # Create directory for models if it doesn't exist
        model_dir = os.path.dirname(CLASSIFICATION_CONFIG['model_path'])
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
    
    def _init_camera(self):
        """Initialize the camera module."""
        try:
            self.logger.info("Initializing camera")
            self.camera = CameraModule(
                camera_index=CAMERA_CONFIG['camera_index'],
                resolution=CAMERA_CONFIG['resolution'],
                frame_rate=CAMERA_CONFIG['frame_rate'],
                auto_focus=CAMERA_CONFIG['auto_focus']
            )
            self.logger.info("Camera initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {str(e)}")
            raise
    
    def _init_classifier(self):
        """Initialize the bird classifier."""
        try:
            self.logger.info("Initializing bird classifier")
            self.classifier = BirdClassifier(
                model_path=CLASSIFICATION_CONFIG['model_path'],
                labels_path=CLASSIFICATION_CONFIG['labels_path'],
                model_url=CLASSIFICATION_CONFIG['model_url'],
                labels_url=CLASSIFICATION_CONFIG['labels_url'],
                top_k=CLASSIFICATION_CONFIG['top_k'],
                confidence_threshold=CLASSIFICATION_CONFIG['confidence_threshold']
            )
            self.logger.info("Bird classifier initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize bird classifier: {str(e)}")
            raise
    
    def _init_notifier(self):
        """Initialize the notification module."""
        try:
            self.logger.info("Initializing notification system")
            self.notifier = BirdNotifier(NOTIFICATION_CONFIG)
            self.logger.info("Notification system initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize notification system: {str(e)}")
            self.logger.warning("Continuing without notification capability")
            self.notifier = None
    
    def _detect_motion(self, frame):
        """
        Detect motion in the frame.
        
        Args:
            frame: The current frame from the camera
            
        Returns:
            bool: True if motion is detected, False otherwise
        """
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Initialize previous frame if not set
        if self.prev_frame is None:
            self.prev_frame = gray
            return False
        
        # Calculate absolute difference between current and previous frame
        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        
        # Dilate the thresholded image to fill in holes
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours on thresholded image
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check if any contour is large enough to be considered motion
        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > APP_CONFIG['motion_threshold']:
                motion_detected = True
                break
        
        # Update previous frame
        self.prev_frame = gray
        
        return motion_detected
    
    def _save_image(self, image, bird_data=None):
        """
        Save the captured image.
        
        Args:
            image: The image to save
            bird_data: Optional bird classification data
            
        Returns:
            str: Path to the saved image
        """
        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create filename
        if bird_data:
            species = bird_data['species'].replace(' ', '_')
            confidence = int(bird_data['confidence'])
            filename = f"{timestamp}_{species}_{confidence}.jpg"
        else:
            filename = f"{timestamp}.jpg"
        
        # Full path to save the image
        save_path = os.path.join(APP_CONFIG['image_save_dir'], filename)
        
        # Save the image
        cv2.imwrite(save_path, image)
        
        return save_path
    
    def _manage_storage(self):
        """
        Manage storage by removing old images if needed.
        """
        # Get list of images in the directory
        image_dir = APP_CONFIG['image_save_dir']
        if not os.path.exists(image_dir):
            return
            
        images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                 if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # If we have more images than the maximum allowed, remove the oldest ones
        if len(images) > APP_CONFIG['max_stored_images']:
            # Sort images by modification time (oldest first)
            images.sort(key=lambda x: os.path.getmtime(x))
            
            # Calculate how many images to remove
            num_to_remove = len(images) - APP_CONFIG['max_stored_images']
            
            # Remove oldest images
            for i in range(num_to_remove):
                try:
                    os.remove(images[i])
                    self.logger.debug(f"Removed old image: {images[i]}")
                except Exception as e:
                    self.logger.error(f"Failed to remove old image {images[i]}: {str(e)}")
    
    def capture_and_classify(self):
        """
        Capture an image and classify any birds in it.
        
        Returns:
            tuple: (bird_data, image_path) if a bird is detected, (None, None) otherwise
        """
        try:
            # Capture frame from camera
            frame = self.camera.capture_frame()
            if frame is None:
                self.logger.warning("Failed to capture frame")
                return None, None
            
            self.stats['images_captured'] += 1
            
            # Check for motion if enabled
            if APP_CONFIG['motion_detection']:
                motion_detected = self._detect_motion(frame)
                if not motion_detected:
                    return None, None
                self.logger.debug("Motion detected")
            
            # Process the image for classification
            processed_image = self.camera.process_image(
                frame, 
                target_size=CAMERA_CONFIG['target_size'],
                normalize=CAMERA_CONFIG['normalize']
            )
            
            # Classify the image
            predictions = self.classifier.classify_image(processed_image)
            
            # Check if any bird was detected with sufficient confidence
            if predictions and predictions[0]['confidence'] >= CLASSIFICATION_CONFIG['confidence_threshold']:
                # Get the top prediction
                top_prediction = predictions[0]
                species = top_prediction['species']
                confidence = top_prediction['confidence']
                
                self.logger.info(f"Bird detected: {species} with {confidence:.2f}% confidence")
                self.stats['birds_detected'] += 1
                
                # Save the image
                image_path = self._save_image(frame, {
                    'species': species,
                    'confidence': confidence
                })
                
                # Prepare bird data for notification
                bird_data = {
                    'species': species,
                    'confidence': confidence,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                return bird_data, image_path
            else:
                # Save the image if configured to save all images
                if APP_CONFIG['save_all_images']:
                    self._save_image(frame)
                
                return None, None
                
        except Exception as e:
            self.logger.error(f"Error during capture and classification: {str(e)}")
            return None, None
    
    def send_notification(self, bird_data, image_path):
        """
        Send notification about the detected bird.
        
        Args:
            bird_data: Dictionary containing bird species, confidence, and timestamp
            image_path: Path to the captured image
            
        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        if not self.notifier:
            self.logger.warning("Notification system not available")
            return False
            
        try:
            # Send notification
            results = self.notifier.notify(bird_data, image_path)
            
            # Log results
            if NOTIFICATION_CONFIG['discord_enabled']:
                status = "succeeded" if results['discord'] else "failed"
                self.logger.info(f"Discord notification {status}")
                
            if NOTIFICATION_CONFIG['mobile_enabled']:
                status = "succeeded" if results['mobile'] else "failed"
                self.logger.info(f"Mobile notification {status}")
            
            # Update statistics if any notification was sent
            if (NOTIFICATION_CONFIG['discord_enabled'] and results['discord']) or \
               (NOTIFICATION_CONFIG['mobile_enabled'] and results['mobile']):
                self.stats['notifications_sent'] += 1
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error sending notification: {str(e)}")
            return False
    
    def print_status(self):
        """Print current status information."""
        runtime = datetime.now() - self.stats['start_time']
        hours, remainder = divmod(runtime.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print("\n--- Bird Classification System Status ---")
        print(f"Runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print(f"Images captured: {self.stats['images_captured']}")
        print(f"Birds detected: {self.stats['birds_detected']}")
        print(f"Notifications sent: {self.stats['notifications_sent']}")
        print("---------------------------------------\n")
    
    def run(self):
        """Run the main application loop."""
        self.logger.info("Starting Bird Classification System")
        
        try:
            while running:
                # Capture and classify
                bird_data, image_path = self.capture_and_classify()
                
                # Send notification if bird was detected
                if bird_data and image_path:
                    self.send_notification(bird_data, image_path)
                
                # Manage storage
                self._manage_storage()
                
                # Wait for the next capture
                time.sleep(APP_CONFIG['capture_interval'])
                
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        except Exception as e:
            self.logger.error(f"Error in main loop: {str(e)}")
        finally:
            self.cleanup()
    
    def test_camera(self):
        """Test the camera component."""
        print("Testing camera...")
        frame = self.camera.capture_frame()
        if frame is not None:
            print("Camera test: SUCCESS")
            test_image_path = os.path.join(APP_CONFIG['image_save_dir'], "camera_test.jpg")
            cv2.imwrite(test_image_path, frame)
            print(f"Test image saved to: {test_image_path}")
        else:
            print("Camera test: FAILED")
    
    def test_classifier(self):
        """Test the classifier component."""
        print("Testing classifier...")
        frame = self.camera.capture_frame()
        if frame is None:
            print("Classifier test: FAILED (could not capture frame)")
            return
            
        processed_image = self.camera.process_image(
            frame, 
            target_size=CAMERA_CONFIG['target_size'],
            normalize=CAMERA_CONFIG['normalize']
        )
        
        predictions = self.classifier.classify_image(processed_image)
        if predictions:
            print("Classifier test: SUCCESS")
            print("Top predictions:")
            for pred in predictions:
                print(f"  - {pred['species']}: {pred['confidence']:.2f}%")
        else:
            print("Classifier test: FAILED (no predictions returned)")
    
    def test_notifier(self):
        """Test the notification component."""
        print("Testing notifier...")
        if not self.notifier:
            print("Notifier test: FAILED (notifier not initialized)")
            return
            
        # Create test data
        test_data = {
            'species': 'Test Bird',
            'confidence': 99.9,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Capture a test image
        frame = self.camera.capture_frame()
        if frame is None:
            print("Notifier test: FAILED (could not capture frame)")
            return
            
        test_image_path = os.path.join(APP_CONFIG['image_save_dir'], "notification_test.jpg")
        cv2.imwrite(test_image_path, frame)
        
        # Send test notification
        results = self.notifier.notify(test_data, test_image_path)
        
        # Check results
        if NOTIFICATION_CONFIG['discord_enabled']:
            status = "succeeded" if results['discord'] else "failed"
            print(f"Discord notification test: {status}")
            
        if NOTIFICATION_CONFIG['mobile_enabled']:
            status = "succeeded" if results['mobile'] else "failed"
            print(f"Mobile notification test: {status}")
            
        if not NOTIFICATION_CONFIG['discord_enabled'] and not NOTIFICATION_CONFIG['mobile_enabled']:
            print("Notifier test: SKIPPED (no notification methods enabled)")
    
    def cleanup(self):
        """Clean up resources before exiting."""
        self.logger.info("Cleaning up resources")
        
        # Release camera
        if hasattr(self, 'camera'):
            self.camera.release()
            
        self.logger.info("Bird Classification System shutdown complete")


def signal_handler(sig, frame):
    """Handle signals for graceful shutdown."""
    global running
    print("\nShutdown signal received. Cleaning up...")
    running = False


def main():
    """Main entry point for the application."""
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Bird Classification System')
    parser.add_argument('--test-camera', action='store_true', help='Test camera functionality')
    parser.add_argument('--test-classifier', action='store_true', help='Test classifier functionality')
    parser.add_argument('--test-notifier', action='store_true', help='Test notification functionality')
    parser.add_argument('--status', action='store_true', help='Print status information and exit')
    args = parser.parse_args()
    
    # Initialize the application
    app = BirdClassificationApp()
    
    # Handle test commands
    if args.test_camera:
        app.test_camera()
        app.cleanup()
        return
        
    if args.test_classifier:
        app.test_classifier()
        app.cleanup()
        return
        
    if args.test_notifier:
        app.test_notifier()
        app.cleanup()
        return
        
    if args.status:
        app.print_status()
        app.cleanup()
        return
    
    # Run the main application
    app.run()


if __name__ == "__main__":
    main()