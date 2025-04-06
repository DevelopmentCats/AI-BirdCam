#!/usr/bin/env python3
"""
Test script for the Bird Classification Notification Module.

This script demonstrates sending notifications via Discord and Pushbullet
using the BirdNotifier class from notification_module.py.
"""

import os
import sys
import argparse
from datetime import datetime
from notification_module import BirdNotifier, ConfigurationError

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test Bird Classification Notification System')
    
    parser.add_argument('--discord-webhook', 
                        help='Discord webhook URL for notifications')
    
    parser.add_argument('--pushbullet-key',
                        help='Pushbullet API key for mobile notifications')
    
    parser.add_argument('--image', default='test_data/sample_bird.jpg',
                        help='Path to bird image (default: test_data/sample_bird.jpg)')
    
    parser.add_argument('--species', default='Eastern Yellow Robin',
                        help='Bird species name (default: Eastern Yellow Robin)')
    
    parser.add_argument('--confidence', type=float, default=95.7,
                        help='Classification confidence score (default: 95.7)')
    
    return parser.parse_args()

def main():
    """Run the notification test."""
    args = parse_arguments()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return 1
    
    # Determine which notification methods to enable
    discord_enabled = bool(args.discord_webhook)
    mobile_enabled = bool(args.pushbullet_key)
    
    if not discord_enabled and not mobile_enabled:
        print("Error: You must provide at least one notification method:")
        print("  --discord-webhook URL   : for Discord notifications")
        print("  --pushbullet-key KEY    : for mobile notifications")
        return 1
    
    # Configure the notifier
    config = {
        'discord_enabled': discord_enabled,
        'discord_webhook_url': args.discord_webhook,
        'mobile_enabled': mobile_enabled,
        'pushbullet_api_key': args.pushbullet_key
    }
    
    # Create bird classification data
    bird_data = {
        'species': args.species,
        'confidence': args.confidence,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Initialize and test the notifier
    try:
        print("Initializing BirdNotifier...")
        notifier = BirdNotifier(config)
        
        print(f"Sending notifications for {bird_data['species']} with {bird_data['confidence']}% confidence")
        results = notifier.notify(bird_data, args.image)
        
        # Report results
        if results['discord']:
            print("✅ Discord notification sent successfully")
        elif discord_enabled:
            print("❌ Discord notification failed")
            
        if results['mobile']:
            print("✅ Mobile notification sent successfully")
        elif mobile_enabled:
            print("❌ Mobile notification failed")
            
    except ConfigurationError as e:
        print(f"Configuration error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())