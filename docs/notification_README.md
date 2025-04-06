# Bird Classification Notification Module

This module provides functionality to send notifications with bird classification results to mobile devices and Discord.

## Features

- Send notifications with bird classification results to mobile devices via Pushbullet
- Send notifications with bird classification results to Discord via webhooks
- Include the captured bird image in the notifications
- Clean API that can be called from the main application
- Error handling for network issues, API failures, etc.
- Configuration options for notification settings
- Enable/disable each notification method independently

## Installation

1. Install the required dependencies:

```bash
pip install discord-webhook pushbullet.py
```

2. Copy the notification module files to your project:
   - `notification_module.py`: The main notification module
   - `notification_config_template.py`: Template for configuration

3. Create your configuration file:
   - Copy `notification_config_template.py` to `notification_config.py`
   - Fill in your Discord webhook URL and Pushbullet API key

## Usage

### Basic Usage

```python
from notification_module import BirdNotifier
from notification_config import NOTIFICATION_CONFIG

# Initialize the notifier
notifier = BirdNotifier(NOTIFICATION_CONFIG)

# Bird classification data
bird_data = {
    'species': 'Eastern Yellow Robin',
    'confidence': 95.7,
    'timestamp': '2025-04-06 21:15:30'  # Optional, will use current time if not provided
}

# Path to the bird image
image_path = 'path/to/bird_image.jpg'

# Send notifications
results = notifier.notify(bird_data, image_path)

# Check results
if results['discord']:
    print("Discord notification sent successfully")
if results['mobile']:
    print("Mobile notification sent successfully")
```

### Testing

You can use the included test script to test your notification setup:

```bash
# Test Discord notifications
python test_notification.py --discord-webhook YOUR_DISCORD_WEBHOOK_URL

# Test Pushbullet notifications
python test_notification.py --pushbullet-key YOUR_PUSHBULLET_API_KEY

# Test both notification methods
python test_notification.py --discord-webhook YOUR_DISCORD_WEBHOOK_URL --pushbullet-key YOUR_PUSHBULLET_API_KEY

# Use a custom image and bird data
python test_notification.py --discord-webhook YOUR_DISCORD_WEBHOOK_URL --image path/to/image.jpg --species "Blue Jay" --confidence 92.5
```

## Getting API Keys and Webhook URLs

### Discord Webhook URL

1. Open Discord and go to the server where you want to receive notifications
2. Right-click on a text channel and select "Edit Channel"
3. Go to "Integrations" > "Webhooks" > "New Webhook"
4. Customize the name and avatar if desired
5. Click "Copy Webhook URL"

### Pushbullet API Key

1. Go to https://www.pushbullet.com/
2. Sign up or log in
3. Go to "Settings" > "Account"
4. Click "Create Access Token"
5. Copy your API key

## Integration with Bird Classification System

This notification module is designed to work with the outputs from the camera_module.py and classification_module.py. When a bird is detected and classified, you can call the notification module to send notifications with the results.

```python
# Example integration with bird classification system
def on_bird_classified(species, confidence, image_path):
    bird_data = {
        'species': species,
        'confidence': confidence,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    notifier.notify(bird_data, image_path)
```