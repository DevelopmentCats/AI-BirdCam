#!/usr/bin/env python3
"""
Example of integrating the notification module with a bird classification system.

This script demonstrates how to use the BirdNotifier class with the outputs
from a hypothetical camera_module.py and classification_module.py.
"""

import os
import time
from datetime import datetime
from notification_module import BirdNotifier

def load_config():
    """
    Load notification configuration.
    
    In a real application, this would load from a config file or environment variables.
    """
    # For demonstration purposes, we'll use environment variables
    # You would typically load this from a config file
    return {
        'discord_enabled': os.environ.get('DISCORD_ENABLED', 'false').lower() == 'true',
        'discord_webhook_url': os.environ.get('DISCORD_WEBHOOK_URL', ''),
        'mobile_enabled': os.environ.get('MOBILE_ENABLED', 'false').lower() == 'true',
        'pushbullet_api_key': os.environ.get('PUSHBULLET_API_KEY', '')
    }

def simulate_bird_detection_and_classification():
    """
    Simulate the bird detection and classification process.
    
    In a real application, this would use camera_module.py and classification_module.py.
    """
    # Simulate capturing an image
    print("Bird detected! Capturing image...")
    time.sleep(1)  # Simulate processing time
    image_path = "test_data/sample_bird.jpg"  # In a real app, this would be a newly captured image
    
    # Simulate classification
    print("Classifying bird...")
    time.sleep(1)  # Simulate processing time
    
    # Simulated classification result
    species = "Eastern Yellow Robin"
    confidence = 95.7
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"Bird classified as {species} with {confidence}% confidence")
    
    return {
        'species': species,
        'confidence': confidence,
        'timestamp': timestamp
    }, image_path

def main():
    """Run the integration example."""
    print("Starting Bird Classification System with Notifications")
    
    # Load configuration
    config = load_config()
    
    # Check if notifications are enabled
    if not config['discord_enabled'] and not config['mobile_enabled']:
        print("Warning: No notification methods are enabled. Set DISCORD_ENABLED=true or MOBILE_ENABLED=true")
        print("and provide the corresponding API keys/webhook URLs to enable notifications.")
    
    # Initialize the notifier
    notifier = BirdNotifier(config)
    
    # Simulate the bird detection and classification process
    bird_data, image_path = simulate_bird_detection_and_classification()
    
    # Send notifications
    print("Sending notifications...")
    results = notifier.notify(bird_data, image_path)
    
    # Report results
    if config['discord_enabled']:
        status = "succeeded" if results['discord'] else "failed"
        print(f"Discord notification {status}")
        
    if config['mobile_enabled']:
        status = "succeeded" if results['mobile'] else "failed"
        print(f"Mobile notification {status}")
    
    print("Done!")

if __name__ == "__main__":
    main()