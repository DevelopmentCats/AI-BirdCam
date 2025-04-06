"""
Bird Classification Example

This script demonstrates how to integrate the camera module with the bird classification module
to create a complete bird classification system.
"""

import os
import time
import logging
import argparse
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the modules
try:
    from camera_module import CameraModule
    from classification_module import BirdClassifier
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Make sure camera_module.py and classification_module.py are in the current directory.")
    exit(1)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Bird Classification System")
    
    # Camera options
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--resolution", type=str, default="1280x720", help="Camera resolution (WxH)")
    parser.add_argument("--fps", type=int, default=30, help="Camera frame rate")
    
    # Classification options
    parser.add_argument("--model", type=str, default="bird_model.tflite", help="Path to the model file")
    parser.add_argument("--labels", type=str, default="bird_labels.txt", help="Path to the labels file")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top predictions to return")
    parser.add_argument("--threshold", type=float, default=0.1, help="Confidence threshold")
    
    # Operation mode
    parser.add_argument("--continuous", action="store_true", help="Run in continuous mode")
    parser.add_argument("--interval", type=float, default=2.0, help="Interval between captures in continuous mode (seconds)")
    parser.add_argument("--output_dir", type=str, default="bird_captures", help="Directory to save captured images")
    parser.add_argument("--save_images", action="store_true", help="Save captured images")
    
    return parser.parse_args()

def setup_output_directory(output_dir: str) -> None:
    """Create output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        except Exception as e:
            logger.error(f"Failed to create output directory: {e}")
            raise

def main():
    """Main function to run the bird classification system."""
    # Parse arguments
    args = parse_arguments()
    
    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except ValueError:
        logger.error(f"Invalid resolution format: {args.resolution}. Using default 1280x720.")
        resolution = (1280, 720)
    
    # Setup output directory if saving images
    if args.save_images:
        setup_output_directory(args.output_dir)
    
    # Initialize camera
    try:
        camera = CameraModule(
            camera_index=args.camera,
            resolution=resolution,
            frame_rate=args.fps
        )
        camera.initialize()
        logger.info("Camera initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize camera: {e}")
        return
    
    # Initialize classifier
    try:
        classifier = BirdClassifier(
            model_path=args.model,
            labels_path=args.labels,
            top_k=args.top_k,
            confidence_threshold=args.threshold
        )
        logger.info("Classifier initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize classifier: {e}")
        camera.release()
        return
    
    try:
        if args.continuous:
            # Continuous mode
            logger.info(f"Running in continuous mode with {args.interval}s interval. Press Ctrl+C to stop.")
            
            while True:
                # Capture and process frame
                frame = camera.capture_and_process()
                
                if frame is None:
                    logger.warning("Failed to capture frame")
                    continue
                
                # Classify the frame
                results = classifier.classify_image(frame)
                
                # Display results
                print("\n===== Bird Classification Results =====")
                if results:
                    for result in results:
                        confidence_pct = result['confidence'] * 100
                        print(f"{result['rank']}. {result['species']} - {confidence_pct:.2f}% confidence")
                else:
                    print("No birds detected or classification failed.")
                
                # Save the frame if requested
                if args.save_images and results:
                    # Use timestamp and top species for filename
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    top_species = results[0]['species'].replace(" ", "_")
                    filename = f"{timestamp}_{top_species}.jpg"
                    filepath = os.path.join(args.output_dir, filename)
                    
                    camera.save_image(frame, filepath)
                    logger.info(f"Saved image to {filepath}")
                
                # Wait for the specified interval
                time.sleep(args.interval)
        else:
            # Single capture mode
            logger.info("Capturing a single frame...")
            
            # Capture and process frame
            frame = camera.capture_and_process()
            
            if frame is None:
                logger.error("Failed to capture frame")
                return
            
            # Classify the frame
            results = classifier.classify_image(frame)
            
            # Display results
            print("\n===== Bird Classification Results =====")
            if results:
                for result in results:
                    confidence_pct = result['confidence'] * 100
                    print(f"{result['rank']}. {result['species']} - {confidence_pct:.2f}% confidence")
            else:
                print("No birds detected or classification failed.")
            
            # Save the frame if requested
            if args.save_images:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"{timestamp}_bird_capture.jpg"
                filepath = os.path.join(args.output_dir, filename)
                
                camera.save_image(frame, filepath)
                logger.info(f"Saved image to {filepath}")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Release camera resources
        camera.release()
        logger.info("Camera resources released")

if __name__ == "__main__":
    main()