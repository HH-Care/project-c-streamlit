# Drone Tracking Application

A Streamlit-based application for tracking drones and analyzing their movement using YOLOv8 and computer vision.

## Features

- Real-time object detection and tracking of drones
- 3D movement analysis (horizontal, vertical, and depth direction)
- Speed calculation with calibration options
- Interactive object selection
- Trajectory visualization

## Requirements

- Python 3.8 or higher
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository
2. Install dependencies:
```
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```
streamlit run app.py
```

2. In the application:
   - Upload a video file or select a sample video
   - When the video loads, select an object to track from the detected objects
   - Adjust tracking parameters in the sidebar if needed
   - View tracking results in real-time

## Project Structure

- `app.py`: Streamlit application
- `final_tracking.py`: Original tracking implementation (reference)
- `models/`: Directory containing YOLOv8 model (you need to place exp1.pt here)
- `videos/`: Sample videos

## Notes

- The application requires a trained YOLOv8 model file named `exp1.pt` in the `models/` directory
- For optimal performance, GPU acceleration is recommended 