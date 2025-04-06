#!/usr/bin/env python3
"""
Simple example of integrating the notification module with a bird classification system.

This script demonstrates how to use the BirdNotifier class with the outputs
from a hypothetical camera_module.py and classification_module.py.
"""

import os
import time
from datetime import datetime
from notification_module import BirdNotifier

def simulate_bird_detection_and_classification():
    """
    Simulate the bird detection and classification process.
    
    In a real application, this would use camera_module.py and classification_module.py.
    """
    print("Bird detected! Capturing image...")
    time.sleep(1)  # Simulate processing time
    
    # In a real application, this would be a newly captured image
    image_path = "test_data/sample_bird.jpg"
    
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
    print("Bird Classification System with Notifications")
    print("===========================================")
    print()
    
    # Configuration for the notification system
    # In a real application, this would be loaded from a config file
    config = {
        'discord_enabled': False,  # Set to True and provide webhook URL to enable
        'discord_webhook_url': '',
        'mobile_enabled': False,   # Set to True and provide API key to enable
        'pushbullet_api_key': ''
    }
    
    # Check if notifications are enabled
    if not config['discord_enabled'] and not config['mobile_enabled']:
        print("Note: Notifications are disabled in this example.")
        print("To enable notifications, set discord_enabled or mobile_enabled to True")
        print("and provide the corresponding API keys/webhook URLs.")
        print()
    
    # Initialize the notifier
    notifier = BirdNotifier(config)
    
    # Simulate the bird detection and classification process
    bird_data, image_path = simulate_bird_detection_and_classification()
    
    # Send notifications (if enabled)
    print("Sending notifications...")
    results = notifier.notify(bird_data, image_path)
    
    # Report results
    if config['discord_enabled']:
        status = "succeeded" if results['discord'] else "failed"
        print(f"Discord notification {status}")
        
    if config['mobile_enabled']:
        status = "succeeded" if results['mobile'] else "failed"
        print(f"Mobile notification {status}")
    
    if not config['discord_enabled'] and not config['mobile_enabled']:
        print("No notifications were sent (notifications are disabled).")
    
    print("\nIn a real application, this would be integrated with:")
    print("1. camera_module.py - to capture images when birds are detected")
    print("2. classification_module.py - to classify the bird species")
    print("3. A configuration system - to manage notification settings")
    
    print("\nExample completed.")

if __name__ == "__main__":
    main()