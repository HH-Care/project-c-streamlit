# Drone Tracking Application - User Guide

This guide will walk you through using the Streamlit-based drone tracking application.

## Prerequisites

Before running the application, ensure you have:

1. Installed all dependencies: `pip install -r requirements.txt`
2. A trained YOLOv8 model in the `models/` directory (named `exp1.pt`)
3. Video files of drones to analyze (or use the sample video)

## Running the Application

1. Start the application:
   ```
   streamlit run app.py
   ```

2. The application will open in your default web browser.

## Using the Application

### Step 1: Select Video Source

1. In the sidebar, choose either:
   - **Upload Video**: Upload your own drone video file (.mp4, .avi, .mov)
   - **Use Sample Video**: Use the included sample video (videos/deone.mp4)

### Step 2: Select Object to Track

1. Once the video loads, you'll see the first frame with detected objects.
2. Below the frame, you'll see thumbnails of each detected object with their IDs.
3. Click the "Track Object X" button under the object you want to track.

### Step 3: Track and Analyze

1. After selecting an object, tracking will begin automatically.
2. You can:
   - Use the **Play/Pause** button to control video playback
   - View real-time tracking information in the right sidebar:
     - Object ID
     - Direction (horizontal/vertical)
     - Depth movement (toward/away from camera)
     - Speed (km/h)
   - Observe visual indicators on the video:
     - Green bounding box around tracked object
     - Orange arrow showing movement direction
     - Purple arrow showing depth movement
     - Path history trail

### Step 4: Adjust Parameters (Optional)

In the sidebar, you can adjust:

1. **FPS**: Video frames per second (if not detected automatically)
2. **Pixels per meter**: Calibration for distance measurement (affects speed calculation)
3. **Speed calibration factor**: Fine-tune speed measurements
4. **Reset Tracking**: Clear all tracking data and start over

## Troubleshooting

- If no objects are detected, try a different video or check the model.
- If tracking is inaccurate, try adjusting the calibration parameters.
- If the application runs slowly, try reducing the video resolution.

## Advanced Features

### 3D Movement Analysis

The application analyzes movement in three dimensions:

1. **Horizontal**: Left/Right movement in the plane
2. **Vertical**: Up/Down movement in the plane
3. **Depth**: Toward/Away from the camera (based on object size changes)

### Speed Calculation

Speed is calculated based on:
- Pixel movement between frames
- FPS of the video
- Pixels-per-meter calibration
- Applied calibration factor

Adjust these parameters for accurate speed measurements in different video conditions. 