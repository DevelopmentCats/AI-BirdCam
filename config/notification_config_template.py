"""
Configuration template for the Bird Classification Notification System.

Copy this file to notification_config.py and fill in your API keys and webhook URLs.
"""

# Configuration for the notification system
NOTIFICATION_CONFIG = {
    # Discord notification settings
    'discord_enabled': True,  # Set to False to disable Discord notifications
    'discord_webhook_url': 'YOUR_DISCORD_WEBHOOK_URL_HERE',
    
    # Mobile notification settings (using Pushbullet)
    'mobile_enabled': True,  # Set to False to disable mobile notifications
    'pushbullet_api_key': 'YOUR_PUSHBULLET_API_KEY_HERE',
}