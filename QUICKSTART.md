# US AI Bird Cam - Quick Start Guide

This guide provides the minimal steps needed to get your US AI Bird Cam up and running quickly.

## Hardware Setup

1. **Assemble the hardware:**
   - Connect the camera module to the Raspberry Pi's camera port
   - Insert the microSD card with Raspberry Pi OS installed
   - Connect power supply

2. **Position the camera:**
   - Point the camera toward the area where birds are likely to appear
   - Ensure the camera has a clear view without obstructions
   - For outdoor setups, place the camera in a weatherproof enclosure

## Software Installation

### Option 1: Automatic Installation (Recommended)

1. **Download the project:**
   ```bash
   git clone https://github.com/DevelopmentCats/ai-birdcam.git
   cd us-ai-bird-cam
   ```

2. **Run the setup script:**
   ```bash
   chmod +x setup.sh
   sudo ./setup.sh
   ```

3. **Wait for installation to complete**
   - The script will install all necessary dependencies
   - Download the bird recognition model
   - Set up the autostart service

### Option 2: Manual Installation

1. **Update your system:**
   ```bash
   sudo apt update
   sudo apt upgrade -y
   ```

2. **Install dependencies:**
   ```bash
   sudo apt install -y python3-pip python3-venv libatlas-base-dev libopenjp2-7 libtiff5 libavcodec-dev libavformat-dev libswscale-dev
   ```

3. **Set up the project:**
   ```bash
   python3 -m venv birdcam_env
   source birdcam_env/bin/activate
   pip install -r requirements.txt
   ```

4. **Enable the camera:**
   ```bash
   sudo raspi-config
   ```
   Navigate to Interface Options → Camera → Enable

## Configuration

1. **Edit the configuration file:**
   ```bash
   nano config.json
   ```

2. **Key settings to consider:**
   - `camera_resolution`: Set according to your camera capabilities
   - `confidence_threshold`: Lower for more detections, higher for more accuracy
   - `save_path`: Where images and videos will be stored

## Starting the System

### If you used automatic installation:
The system starts automatically on boot. To check status:
```bash
sudo systemctl status birdcam.service
```

### To start manually:
```bash
cd us-ai-bird-cam
source birdcam_env/bin/activate
python3 birdcam.py
```

## Viewing Results

- Captured images and videos are stored in the `captures` directory
- Detection logs are available in the `logs` directory

## Troubleshooting

If the system isn't working:

1. **Check camera connection:**
   ```bash
   libcamera-hello
   ```

2. **Verify service status:**
   ```bash
   sudo systemctl status birdcam.service
   ```

3. **View logs:**
   ```bash
   cat logs/birdcam.log
   ```

For more detailed information and advanced configuration options, please refer to the full README.md documentation.
