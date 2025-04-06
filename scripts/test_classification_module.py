"""
Test script for the bird classification module.

This script demonstrates how to use the BirdClassifier class to:
1. Load a bird classification model
2. Process a sample image
3. Perform classification
4. Display the results
"""

import os
import sys
import numpy as np
import argparse
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the classification module
from classification_module import BirdClassifier, download_model, get_available_models

def load_sample_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load a sample image for testing.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Image as numpy array or None if loading fails
    """
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return None
    
    try:
        # Try using OpenCV first
        try:
            import cv2
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            logger.info(f"Loaded image using OpenCV: {image.shape}")
            return image
        except ImportError:
            # Fall back to PIL
            from PIL import Image
            image = np.array(Image.open(image_path).convert('RGB'))
            logger.info(f"Loaded image using PIL: {image.shape}")
            return image
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        return None

def download_sample_image() -> str:
    """
    Download a sample bird image for testing if none is provided.
    
    Returns:
        Path to the downloaded image
    """
    import urllib.request
    
    # Sample bird image URLs
    sample_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/4/45/Eopsaltria_australis_-_Mogo_Campground.jpg",  # Eastern Yellow Robin
        "https://upload.wikimedia.org/wikipedia/commons/a/ae/Carduelis_carduelis_close_up.jpg",  # European Goldfinch
        "https://upload.wikimedia.org/wikipedia/commons/e/ec/Cyanocitta_cristata_-_Blue_Jay_04.jpg"  # Blue Jay
    ]
    
    # Choose a random image
    import random
    sample_url = random.choice(sample_urls)
    
    # Download the image
    sample_path = "sample_bird.jpg"
    try:
        urllib.request.urlretrieve(sample_url, sample_path)
        logger.info(f"Downloaded sample image to {sample_path}")
        return sample_path
    except Exception as e:
        logger.error(f"Failed to download sample image: {e}")
        return ""

def display_results(results, image=None):
    """
    Display the classification results.
    
    Args:
        results: Classification results from BirdClassifier
        image: Optional image to display alongside results
    """
    print("\n===== Bird Classification Results =====")
    
    if not results:
        print("No birds detected or classification failed.")
        return
    
    print(f"Found {len(results)} potential bird species:\n")
    
    for result in results:
        confidence_pct = result['confidence'] * 100
        print(f"{result['rank']}. {result['species']} - {confidence_pct:.2f}% confidence")
    
    # If matplotlib is available and image is provided, display the image with results
    if image is not None:
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.imshow(image)
            
            # Add title with top prediction
            if results:
                top_species = results[0]['species']
                top_confidence = results[0]['confidence'] * 100
                plt.title(f"Top prediction: {top_species} ({top_confidence:.2f}%)")
            
            # Add text annotations for all predictions
            y_pos = 10
            for result in results:
                confidence_pct = result['confidence'] * 100
                text = f"{result['rank']}. {result['species']} - {confidence_pct:.2f}%"
                plt.text(10, y_pos, text, color='white', fontsize=12, 
                         bbox=dict(facecolor='black', alpha=0.7))
                y_pos += 30
            
            plt.axis('off')
            plt.tight_layout()
            
            # Save the visualization
            output_path = "classification_result.jpg"
            plt.savefig(output_path)
            print(f"\nVisualization saved to {output_path}")
            
            # Show the plot if running in an environment with display
            try:
                plt.show()
            except:
                pass
            
        except ImportError:
            logger.warning("Matplotlib not available. Skipping visualization.")

def main():
    """Main function to test the bird classification module."""
    parser = argparse.ArgumentParser(description="Test the bird classification module")
    parser.add_argument("--image", type=str, help="Path to a bird image for testing")
    parser.add_argument("--model", type=str, default="bird_model.tflite", 
                        help="Path to the TensorFlow Lite model")
    parser.add_argument("--labels", type=str, default="bird_labels.txt", 
                        help="Path to the labels file")
    parser.add_argument("--top_k", type=int, default=5, 
                        help="Number of top predictions to return")
    parser.add_argument("--threshold", type=float, default=0.1, 
                        help="Confidence threshold for predictions")
    
    args = parser.parse_args()
    
    # Get or download a sample image
    image_path = args.image
    if not image_path or not os.path.exists(image_path):
        logger.info("No valid image path provided. Downloading a sample image...")
        image_path = download_sample_image()
        if not image_path:
            logger.error("Failed to get a sample image. Exiting.")
            return
    
    # Load the image
    image = load_sample_image(image_path)
    if image is None:
        return
    
    # Initialize the classifier
    try:
        classifier = BirdClassifier(
            model_path=args.model,
            labels_path=args.labels,
            top_k=args.top_k,
            confidence_threshold=args.threshold
        )
        
        # Print model info
        model_info = classifier.get_model_info()
        print("\n===== Model Information =====")
        for key, value in model_info.items():
            print(f"{key}: {value}")
        
        # Classify the image
        results = classifier.classify_image(image)
        
        # Display the results
        display_results(results, image)
        
    except Exception as e:
        logger.error(f"Error during classification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()