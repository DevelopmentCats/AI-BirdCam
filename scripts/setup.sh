#!/bin/bash

# US AI Bird Cam Setup Script
# This script automates the installation process for the US AI Bird Cam project

echo "===== US AI Bird Cam Setup ====="
echo "Starting installation process..."

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (use sudo)"
  exit 1
fi

# Create project directory structure
echo "Creating project directory structure..."
PROJECT_DIR="$HOME/us-ai-bird-cam"
mkdir -p "$PROJECT_DIR/models"
mkdir -p "$PROJECT_DIR/captures"
mkdir -p "$PROJECT_DIR/logs"

# Navigate to project directory
cd "$PROJECT_DIR" || exit 1

# Update system
echo "Updating system packages..."
apt update
apt upgrade -y

# Install system dependencies
echo "Installing system dependencies..."
apt install -y \
  python3-pip \
  python3-venv \
  libatlas-base-dev \
  libopenjp2-7 \
  libtiff5 \
  libavcodec-dev \
  libavformat-dev \
  libswscale-dev \
  git \
  wget \
  v4l-utils

# Enable camera interface
echo "Enabling camera interface..."
if command -v raspi-config > /dev/null; then
  raspi-config nonint do_camera 0
  echo "Camera interface enabled"
else
  echo "Warning: raspi-config not found. Please enable camera interface manually."
fi

# Create and activate Python virtual environment
echo "Setting up Python virtual environment..."
python3 -m venv "$PROJECT_DIR/birdcam_env"
source "$PROJECT_DIR/birdcam_env/bin/activate"

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r "$PROJECT_DIR/requirements.txt"

# Download TensorFlow Lite bird species recognition model
echo "Downloading bird species recognition model..."
MODEL_DIR="$PROJECT_DIR/models"
mkdir -p "$MODEL_DIR"

# Download the model
wget -O "$MODEL_DIR/bird_model.tflite" "https://tfhub.dev/google/lite-model/aiy/vision/classifier/birds_V1/3?lite-format=tflite"

# Check if model download was successful
if [ ! -f "$MODEL_DIR/bird_model.tflite" ]; then
  echo "Failed to download model. Please check your internet connection and try again."
  echo "You can manually download the model from https://tfhub.dev/google/lite-model/aiy/vision/classifier/birds_V1/3"
  echo "and place it in the $MODEL_DIR directory."
else
  echo "Model downloaded successfully."
fi

# Download bird species labels
wget -O "$MODEL_DIR/bird_labels.txt" "https://raw.githubusercontent.com/google-coral/test_data/master/bird_labels.txt"

# Create example configuration file
echo "Creating example configuration file..."
cat > "$PROJECT_DIR/config.example.json" << EOL
{
  "camera_resolution": [1920, 1080],
  "camera_framerate": 30,
  "detection_interval": 1,
  "confidence_threshold": 0.7,
  "save_detected_images": true,
  "save_path": "./captures",
  "enable_video": true,
  "video_length": 10,
  "enable_logging": true,
  "log_file": "./logs/birdcam.log",
  "enable_motion_detection": false,
  "motion_sensitivity": 80,
  "model_path": "./models/bird_model.tflite",
  "labels_path": "./models/bird_labels.txt"
}
EOL

# Copy example config to actual config if it doesn't exist
if [ ! -f "$PROJECT_DIR/config.json" ]; then
  cp "$PROJECT_DIR/config.example.json" "$PROJECT_DIR/config.json"
  echo "Created default configuration file."
fi

# Create systemd service file for autostart
echo "Setting up autostart service..."
SERVICE_FILE="/etc/systemd/system/birdcam.service"

cat > "$SERVICE_FILE" << EOL
[Unit]
Description=US AI Bird Cam Service
After=network.target

[Service]
User=$USER
WorkingDirectory=$PROJECT_DIR
ExecStart=$PROJECT_DIR/birdcam_env/bin/python3 $PROJECT_DIR/birdcam.py
Restart=always
RestartSec=10
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=birdcam

[Install]
WantedBy=multi-user.target
EOL

# Create a simple starter script
cat > "$PROJECT_DIR/birdcam.py" << EOL
#!/usr/bin/env python3
"""
US AI Bird Cam - Main Application

This is a placeholder script. Replace with your actual implementation.
"""
import json
import logging
import os
import time
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/birdcam.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BirdCam")

def main():
    logger.info("US AI Bird Cam starting...")
    
    # Load configuration
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
            logger.info(f"Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return
    
    # Check if model exists
    model_path = config.get("model_path", "./models/bird_model.tflite")
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        return
    
    logger.info("Bird detection model found")
    logger.info("System initialized and ready")
    logger.info("Waiting for implementation...")
    
    # Placeholder for actual implementation
    while True:
        logger.info("System running... (placeholder)")
        time.sleep(60)

if __name__ == "__main__":
    main()
EOL

# Make the script executable
chmod +x "$PROJECT_DIR/birdcam.py"

# Enable and start the service
systemctl daemon-reload
systemctl enable birdcam.service
systemctl start birdcam.service

echo "===== Installation Complete ====="
echo "The US AI Bird Cam has been installed and configured."
echo "Service status: $(systemctl is-active birdcam.service)"
echo ""
echo "To check the service status: sudo systemctl status birdcam.service"
echo "To view logs: sudo journalctl -u birdcam.service"
echo "Configuration file: $PROJECT_DIR/config.json"
echo ""
echo "For more information, please refer to the README.md file."