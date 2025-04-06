# US AI Bird Cam

## Project Overview

The US AI Bird Cam is an intelligent bird identification system built on Raspberry Pi that uses machine learning to detect, identify, and record bird species in real-time. This project combines hardware and software components to create an automated bird watching station that can be deployed in various environments.

### Features

- Real-time bird detection and species identification
- Automatic image and video capture when birds are detected
- Species logging with timestamps
- Support for day and night monitoring (with appropriate camera)
- Optional remote access capabilities
- Low power consumption for extended deployment
- Customizable detection sensitivity and notification settings

## Hardware Requirements

### Essential Components

- **Raspberry Pi**: Raspberry Pi 4 Model B (2GB RAM or higher recommended)
- **Camera**: One of the following options:
  - Raspberry Pi Camera Module 3 (recommended for best quality)
  - Raspberry Pi Camera Module 3 Wide Angle (120° field of view, better for capturing broader scenes)
  - Raspberry Pi Camera Module 3 NoIR (for low-light/night wildlife photography)
  - Raspberry Pi Camera Module V2 (budget option, 8MP Sony IMX219 sensor)
- **Storage**: MicroSD card (32GB or larger recommended)
- **Power Supply**: Official Raspberry Pi USB-C power supply (5.1V, 3A)

### Optional Components

- **Motion Sensor**: PIR motion sensor (to trigger recording only when movement is detected)
- **Weatherproof Case**: For outdoor deployment
- **External Storage**: USB drive for storing large amounts of footage
- **Google Coral USB Accelerator**: For faster AI inference
- **Small Microphone**: For audio recording (if using sound-based identification)
- **5V Fan**: For cooling in hot environments or during extended operation

## Software Prerequisites

- Raspberry Pi OS (Bullseye or newer)
- Python 3.7 or higher
- Internet connection (for initial setup and model download)

## Installation

### Automatic Installation

For quick setup, use the provided installation script:

```bash
chmod +x setup.sh
./setup.sh
```

### Manual Installation

1. Update your Raspberry Pi:
   ```bash
   sudo apt update
   sudo apt upgrade -y
   ```

2. Install system dependencies:
   ```bash
   sudo apt install -y python3-pip python3-venv libatlas-base-dev libopenjp2-7 libtiff5 libavcodec-dev libavformat-dev libswscale-dev
   ```

3. Create and activate a virtual environment:
   ```bash
   python3 -m venv birdcam_env
   source birdcam_env/bin/activate
   ```

4. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Download the TensorFlow Lite bird species recognition model:
   ```bash
   mkdir -p models
   # The setup script will handle model download
   ```

6. Configure the application:
   ```bash
   cp config.example.json config.json
   # Edit config.json with your preferred settings
   ```

## Configuration

The `config.json` file contains all configurable options for the US AI Bird Cam:

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `camera_resolution` | Camera resolution as [width, height] | [1920, 1080] |
| `camera_framerate` | Camera framerate | 30 |
| `detection_interval` | Seconds between detection attempts | 1 |
| `confidence_threshold` | Minimum confidence score (0-1) for positive identification | 0.7 |
| `save_detected_images` | Whether to save images of detected birds | true |
| `save_path` | Directory to save images and videos | "./captures" |
| `enable_video` | Enable video recording | true |
| `video_length` | Length of video to record after detection (seconds) | 10 |
| `enable_logging` | Enable logging of detections | true |
| `log_file` | Path to log file | "./birdcam.log" |
| `enable_motion_detection` | Use motion detection to trigger AI processing | false |
| `motion_sensitivity` | Motion detection sensitivity (1-100) | 80 |

## Usage

### Starting the Application

1. Navigate to the project directory:
   ```bash
   cd us-ai-bird-cam
   ```

2. Activate the virtual environment:
   ```bash
   source birdcam_env/bin/activate
   ```

3. Run the application:
   ```bash
   python3 birdcam.py
   ```

### Setting Up Autostart

To configure the application to start automatically on boot:

1. The setup script creates a systemd service file
2. Enable the service:
   ```bash
   sudo systemctl enable birdcam.service
   sudo systemctl start birdcam.service
   ```

3. Check status:
   ```bash
   sudo systemctl status birdcam.service
   ```

### Accessing Captured Images and Data

- Images and videos are stored in the configured `save_path` directory
- Detection logs are available in the configured log file
- For remote access, consider setting up a Samba share or SSH access

## Troubleshooting

### Camera Issues

- **Error: "Failed to initialize camera"**
  - Ensure the camera is properly connected
  - Enable the camera interface: `sudo raspi-config` → Interface Options → Camera
  - Check if the camera works with `libcamera-hello`

- **Poor image quality**
  - Adjust camera settings in config.json
  - Ensure adequate lighting
  - Clean the camera lens

### Detection Issues

- **No birds detected**
  - Lower the confidence threshold in config.json
  - Ensure the camera has a clear view of bird activity
  - Check if the model file was downloaded correctly

- **False positives**
  - Increase the confidence threshold
  - Adjust camera position to reduce movement in the background

### Performance Issues

- **High CPU usage / overheating**
  - Lower the camera resolution and framerate
  - Increase detection interval
  - Add a cooling fan
  - Consider adding a Google Coral USB Accelerator

- **Application crashes**
  - Check logs for errors: `journalctl -u birdcam.service`
  - Ensure sufficient free space on SD card
  - Verify all dependencies are installed correctly

## Limitations and Future Improvements

### Current Limitations

- Species identification is limited to the birds included in the model's training data
- Performance may vary based on lighting conditions and distance to birds
- High CPU usage during continuous operation may cause thermal throttling
- Limited battery operation time if deployed off-grid

### Potential Improvements

- Integration with cloud services for enhanced identification
- Mobile app for remote monitoring and notifications
- Support for multiple cameras
- Solar power integration for off-grid deployment
- Addition of weather data correlation
- Bird call audio recognition integration
- Web interface for easy configuration and viewing

## Credits and Acknowledgments

- TensorFlow and TensorFlow Lite for providing the machine learning framework
- The bird species identification models are based on datasets of North American birds
- Raspberry Pi Foundation for creating accessible computing hardware
- Open source libraries that make this project possible:
  - OpenCV for computer vision capabilities
  - NumPy for numerical processing
  - BirdNET for audio-based bird identification (if used)

## License

This project is released under the MIT License. See the LICENSE file for details.

## Contributing

Contributions to the US AI Bird Cam project are welcome! Please feel free to submit pull requests or open issues to improve the documentation, code, or features.