# Climbing Project: RPi + ESP32 Laser Projection

This project implements a laser-guided climbing route projection system. It consists of a Flask web interface running on a Raspberry Pi (or similar) that uses OpenCV for computer vision and homography, and an ESP32 driver that controls a galvo-based laser system via a serial protocol.

## Project Structure

- `web_interface/`: Flask application and web assets.
- `vision/`: Core computer vision logic (homography, route detection).
- `ESP_driver/`: PlatformIO project for the ESP32-S3 laser controller.
- `debug_scripts/`: Standalone utilities for calibration, testing, and manual control.
- `pyproject.toml`: Python project configuration and dependencies.

## Hardware Requirements

- Raspberry Pi (with Camera)
- ESP32-S3 (connected via USB/Serial)
- Galvo Laser System (controlled by ESP32)

## Installation

### 1. Python Environment (RPi)

Install the project in editable mode:

```bash
pip install -e .
```

To install development tools (black, flake8, etc.):

```bash
pip install -e ".[dev]"
```

### 2. ESP32 Firmware

Use [PlatformIO](https://platformio.org/) to build and upload the firmware in the `ESP_driver` directory.

## Usage

### Starting the Web App

Run the Flask application from the project root:

```bash
python web_interface/src/app.py
```

The interface will be available at `http://<rpi-ip>:5000`.

### Calibration

1. Navigate to the `/calibrate` route in the web app.
2. Follow the on-screen instructions to map the 4 corners of your climbing wall to the laser's DAC space (0-4095).
3. Save the calibration to generate `vision/homography.json`.

### Debugging

Utilities are available in `debug_scripts/`. Run them as modules:

```bash
# Test serial connection
python -m debug_scripts.serial_test

# Initialize the database manually
python -m debug_scripts.init_db
```

## Git Workflow

This project is organized as a monorepo. 
- Avoid deleting `vision/homography.json` as it contains your specific wall calibration.
- The `web_interface/src/static/uploads/` directory is ignored by git (except for `.gitkeep`) to keep the repo clean of user-uploaded images.
