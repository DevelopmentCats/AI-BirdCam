"""
Configuration file for the Bird Classification System.

This file contains all the configuration parameters for:
- Camera settings
- Classification model settings
- Notification settings
- Application settings
"""

# Camera settings
CAMERA_CONFIG = {
    'camera_index': 0,                # Camera device index (0 for default camera)
    'resolution': (1920, 1080),       # Camera resolution (width, height)
    'frame_rate': 30,                 # Camera frame rate
    'auto_focus': True,               # Enable/disable auto-focus
    'target_size': (224, 224),        # Target size for processed images
    'normalize': True,                # Normalize image pixel values
}

# Classification settings
CLASSIFICATION_CONFIG = {
    'model_path': 'models/bird_model.tflite',  # Path to the TFLite model
    'labels_path': 'models/bird_labels.txt',   # Path to the labels file
    # URLs for automatic model download if files don't exist
    'model_url': 'https://tfhub.dev/google/lite-model/aiy/vision/classifier/birds_V1/3',
    'labels_url': 'https://www.gstatic.com/aihub/tfhub/labelmaps/aiy_birds_V1_labelmap.csv',
    'top_k': 3,                       # Number of top predictions to return
    'confidence_threshold': 0.7,      # Minimum confidence threshold for valid detection
}

# Notification settings
NOTIFICATION_CONFIG = {
    # Discord notification settings
    'discord_enabled': False,         # Set to True to enable Discord notifications
    'discord_webhook_url': '',        # Your Discord webhook URL
    
    # Mobile notification settings (using Pushbullet)
    'mobile_enabled': False,          # Set to True to enable mobile notifications
    'pushbullet_api_key': '',         # Your Pushbullet API key
}

# Application settings
APP_CONFIG = {
    'capture_interval': 5,            # Time between captures in seconds
    'motion_detection': True,         # Enable/disable motion detection
    'motion_threshold': 25,           # Motion detection sensitivity (0-100)
    'log_level': 'INFO',              # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    'log_file': 'bird_classifier.log',# Log file path
    'image_save_dir': 'captured_birds',  # Directory to save captured bird images
    'save_all_images': False,         # Save all captured images or only those with birds
    'max_stored_images': 100,         # Maximum number of images to store
}