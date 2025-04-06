#!/usr/bin/env python3
"""
Demonstration of the Bird Classification Notification Module.

This script shows how to use the BirdNotifier class to send notifications
with bird classification results to Discord and mobile devices.
"""

import os
import sys
import argparse
from datetime import datetime
from notification_module import BirdNotifier, ConfigurationError

def main():
    """Run the notification demonstration."""
    print("Bird Classification Notification Module Demonstration")
    print("===================================================")
    print()
    
    # Check if we have the sample image
    image_path = "test_data/sample_bird.jpg"
    if not os.path.exists(image_path):
        print(f"Error: Sample image not found: {image_path}")
        print("Please run the setup script to download the sample image.")
        return 1
    
    # Sample bird classification data
    bird_data = {
        'species': 'Eastern Yellow Robin',
        'confidence': 95.7,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    print("This demonstration shows how to use the BirdNotifier class.")
    print("To send actual notifications, you need to provide your own API keys.")
    print()
    print("Sample bird classification data:")
    print(f"  Species: {bird_data['species']}")
    print(f"  Confidence: {bird_data['confidence']}%")
    print(f"  Timestamp: {bird_data['timestamp']}")
    print(f"  Image: {image_path}")
    print()
    
    # Demonstrate configuration options
    print("Configuration Options:")
    print("1. Discord Only")
    print("2. Mobile Only (Pushbullet)")
    print("3. Both Discord and Mobile")
    print("4. No Notifications (Disabled)")
    print()
    
    choice = input("Select a configuration option (1-4): ")
    
    # Set up configuration based on choice
    config = {
        'discord_enabled': False,
        'discord_webhook_url': '',
        'mobile_enabled': False,
        'pushbullet_api_key': ''
    }
    
    if choice == '1':
        config['discord_enabled'] = True
        config['discord_webhook_url'] = input("Enter Discord webhook URL: ")
    elif choice == '2':
        config['mobile_enabled'] = True
        config['pushbullet_api_key'] = input("Enter Pushbullet API key: ")
    elif choice == '3':
        config['discord_enabled'] = True
        config['discord_webhook_url'] = input("Enter Discord webhook URL: ")
        config['mobile_enabled'] = True
        config['pushbullet_api_key'] = input("Enter Pushbullet API key: ")
    elif choice == '4':
        print("All notifications disabled.")
    else:
        print("Invalid choice. Using option 4 (No Notifications).")
    
    # Initialize the notifier
    try:
        print("\nInitializing BirdNotifier...")
        notifier = BirdNotifier(config)
        
        # Send notifications
        print("Sending notifications...")
        results = notifier.notify(bird_data, image_path)
        
        # Report results
        print("\nNotification Results:")
        if config['discord_enabled']:
            status = "succeeded" if results['discord'] else "failed"
            print(f"  Discord: {status}")
            
        if config['mobile_enabled']:
            status = "succeeded" if results['mobile'] else "failed"
            print(f"  Mobile: {status}")
            
        if not config['discord_enabled'] and not config['mobile_enabled']:
            print("  No notifications were enabled.")
        
    except ConfigurationError as e:
        print(f"\nConfiguration error: {e}")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        return 1
    
    print("\nDemonstration completed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())