import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
import tempfile
import os
from collections import defaultdict
import time
import warnings
import logging
from PIL import Image
import base64
from io import BytesIO

# For the drawable canvas - now uncommented since it's installed
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except Exception as e:
    CANVAS_AVAILABLE = False
    st.warning(f"Canvas component not available: {str(e)}. Will use manual selection instead.")

# Suppress Streamlit file watcher errors
logging.getLogger('streamlit.watcher.local_sources_watcher').setLevel(logging.ERROR)

# Setup a custom filter for PyTorch/YOLO warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', category=UserWarning, module='ultralytics')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='torch')

# Use try-except to handle startup errors with PyTorch/Streamlit interaction
try:
    import torch._classes
except (ImportError, RuntimeError):
    pass

# Set page config
st.set_page_config(
    page_title="Object Tracker", 
    page_icon="🚁", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
def init_session_state():
    if 'track_history' not in st.session_state:
        st.session_state.track_history = defaultdict(list)
    if 'box_size_history' not in st.session_state:
        st.session_state.box_size_history = defaultdict(list)
    if 'direction_history' not in st.session_state:
        st.session_state.direction_history = {}
    if 'speed_history' not in st.session_state:
        st.session_state.speed_history = {}
    if 'video_path' not in st.session_state:
        st.session_state.video_path = None
    if 'video_source' not in st.session_state:
        st.session_state.video_source = None
    if 'selected_object_id' not in st.session_state:
        st.session_state.selected_object_id = None
    if 'calibration_factor' not in st.session_state:
        st.session_state.calibration_factor = 1.0
    if 'pixels_per_meter' not in st.session_state:
        st.session_state.pixels_per_meter = 100
    if 'processing_phase' not in st.session_state:
        st.session_state.processing_phase = "upload"  # Phases: upload, draw, track
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0
    if 'fps_value' not in st.session_state:
        st.session_state.fps_value = 0.0
    if 'latency_value' not in st.session_state:
        st.session_state.latency_value = 0.0
    if 'user_roi' not in st.session_state:
        st.session_state.user_roi = None  # Store user-drawn bounding box
    if 'first_frame' not in st.session_state:
        st.session_state.first_frame = None  # Store first frame for drawing

# Direction prediction function
def predict_direction(points, box_sizes, num_points=5):
    """
    Determine direction of movement based on recent trajectory points
    and changing object size to infer 3D movement.
    """
    if len(points) < num_points or len(box_sizes) < num_points:
        return "Unknown", "Unknown", "Unknown", None

    # Get the recent points and box sizes for direction analysis
    recent_points = points[-num_points:]
    recent_boxes = box_sizes[-num_points:]

    # Calculate the overall movement vector in the 2D plane
    start_x, start_y = recent_points[0]
    end_x, end_y = recent_points[-1]
    dx = end_x - start_x
    dy = start_y - end_y  # Invert y because screen coordinates have y increasing downward

    # Calculate magnitude of horizontal/vertical movement
    magnitude = math.sqrt(dx * dx + dy * dy)

    # If barely moving horizontally/vertically, could still be moving in depth
    if magnitude < 5:  # Threshold for minimum movement in image plane
        horizontal_direction = "Stationary"
        vertical_direction = "Stationary"
        angle = 0
    else:
        # Calculate angle in degrees for 2D plane movement
        angle = math.degrees(math.atan2(dy, dx))

        # Normalize angle to 0-360 range
        if angle < 0:
            angle += 360

        # Determine horizontal direction
        if 112.5 <= angle < 247.5:  # Left half of circle
            horizontal_direction = "Left"
        elif 292.5 <= angle or angle < 67.5:  # Right half of circle
            horizontal_direction = "Right"
        else:
            horizontal_direction = "Stationary"

        # Determine vertical direction
        if 22.5 <= angle < 157.5:  # Upper half of circle
            vertical_direction = "Up"
        elif 202.5 <= angle < 337.5:  # Lower half of circle
            vertical_direction = "Down"
        else:
            vertical_direction = "Stationary"

    # Analyze 3D movement (toward/away from camera) based on changing object size
    start_w, start_h = recent_boxes[0]
    end_w, end_h = recent_boxes[-1]

    # Calculate the ratio of size change
    size_start = start_w * start_h
    size_end = end_w * end_h

    # Avoid division by zero
    if size_start == 0:
        size_start = 1

    size_ratio = size_end / size_start
    size_change_threshold = 0.05  # 5% change threshold

    # Determine if object is moving toward or away based on size change
    if size_ratio > (1 + size_change_threshold):
        depth_direction = "Toward Camera"
    elif size_ratio < (1 - size_change_threshold):
        depth_direction = "Away from Camera"
    else:
        depth_direction = "Same Distance"

    return horizontal_direction, vertical_direction, depth_direction, angle

# Speed calculation function
def calculate_speed(points, fps, pixels_per_meter, num_points=10):
    """Calculate speed based on points trajectory."""
    if len(points) < num_points:
        return None, None

    # Use recent points for speed calculation
    recent_points = points[-num_points:]

    # Calculate total distance in pixels
    total_distance = 0
    for i in range(1, len(recent_points)):
        x1, y1 = recent_points[i - 1]
        x2, y2 = recent_points[i]
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        total_distance += dist

    # Average distance per frame
    distance_per_frame = total_distance / (len(recent_points) - 1)

    # Convert to distance per second
    distance_per_second = distance_per_frame * fps

    # Convert pixels to meters
    distance_meters_per_second = distance_per_second / pixels_per_meter

    # Convert to km/h
    speed_kmh = distance_meters_per_second * 3.6  # 3.6 is the conversion factor from m/s to km/h

    return speed_kmh, distance_per_frame

# Object selection from a frame with YOLO detections
def select_object_from_detections(frame, detection_results):
    """Display frame with all detections and let user select one"""
    # Create a copy of the frame to draw on
    annotated_frame = frame.copy()
    
    # Check if valid detections exist
    if detection_results[0].boxes.id is not None:
        boxes = detection_results[0].boxes.xywh.cpu().numpy()
        ids = detection_results[0].boxes.id.cpu().numpy().astype(int)
        
        # Draw all detected objects with their IDs
        for i, box in enumerate(boxes):
            x, y, w, h = box
            obj_id = ids[i]
            
            # Calculate box coordinates
            x1, y1 = int(x - w/2), int(y - h/2)
            x2, y2 = int(x + w/2), int(y + h/2)
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw ID
            cv2.putText(annotated_frame, f"ID: {obj_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return annotated_frame, ids.tolist() if detection_results[0].boxes.id is not None else []

# Draw tracking visualization on frame
def draw_tracking_visualization(frame, track_id, track, box_sizes, direction_data, speed_data, 
                              frame_count, calibration_factor, fps=None, latency=None):
    """Draw tracking visualization elements on the frame"""
    annotated_frame = frame.copy()
    
    # Extract box coordinates for the current frame
    x, y, w, h = box_sizes[-1][0], box_sizes[-1][1], box_sizes[-1][2], box_sizes[-1][3]
    
    # Draw bounding box for tracked object
    x1, y1 = int(x - w / 2), int(y - h / 2)
    x2, y2 = int(x + w / 2), int(y + h / 2)
    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw the path (tracking line)
    if len(track) > 1:
        points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_frame, [points], isClosed=False,
                      color=(230, 230, 230), thickness=2)
    
    # Draw a direction arrow for 2D plane visualization
    if direction_data and len(track) > 5:
        # Skip if completely stationary
        if (direction_data['horizontal'] != "Stationary" or
                direction_data['vertical'] != "Stationary"):
            
            angle = direction_data['angle']
            # Get last position
            arrow_start = (int(track[-1][0]), int(track[-1][1]))
            
            # Calculate arrow end point based on angle
            arrow_length = 40
            if angle is not None:
                end_x = arrow_start[0] + arrow_length * math.cos(math.radians(angle))
                end_y = arrow_start[1] - arrow_length * math.sin(math.radians(angle))
                arrow_end = (int(end_x), int(end_y))
                
                # Draw the direction arrow
                cv2.arrowedLine(annotated_frame, arrow_start, arrow_end,
                                (0, 165, 255), 2, tipLength=0.3)
    
    # Visualize depth movement with a vertical arrow
    if direction_data:
        depth_dir = direction_data['depth']
        if depth_dir != "Same Distance":
            # Draw a vertical arrow indicating toward/away movement
            center_x, center_y = int(x), int(y)
            
            # Arrow pointing up when moving toward camera, down when moving away
            if depth_dir == "Toward Camera":
                cv2.arrowedLine(annotated_frame,
                                (center_x, center_y),
                                (center_x, center_y - 40),
                                (255, 0, 255), 2, tipLength=0.3)
            else:  # Away from Camera
                cv2.arrowedLine(annotated_frame,
                                (center_x, center_y),
                                (center_x, center_y + 40),
                                (255, 0, 255), 2, tipLength=0.3)
    
    # Create a dark overlay at the TOP left for text display
    info_panel_height = 170  # Adjusted to make room for FPS/latency
    info_panel_width = 300  # Adjust width as needed
    overlay = annotated_frame.copy()
    cv2.rectangle(overlay, (10, 10),
                  (10 + info_panel_width, 10 + info_panel_height), (220, 220, 220), -1)
    
    # Apply the overlay with transparency
    alpha = 0.7  # Transparency factor
    cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)
    
    # Display information in the TOP-left overlay area (All text in BLACK)
    y_position = 35  # Starting position for text
    
    # Display object ID
    cv2.putText(annotated_frame, f"Object ID: {track_id}", (20, y_position),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    y_position += 20
    
    # Display direction information if available
    if direction_data:
        direction_text = f"Direction: {direction_data['horizontal']} | {direction_data['vertical']}"
        cv2.putText(annotated_frame, direction_text, (20, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        y_position += 20
        
        depth_text = f"Depth: {direction_data['depth']}"
        cv2.putText(annotated_frame, depth_text, (20, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        y_position += 20
    
    # Display speed information if available
    if speed_data:
        speed_text = f"Speed: {speed_data['kmh']:.1f} km/h"
        cv2.putText(annotated_frame, speed_text, (20, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        y_position += 20
    
    # Display frame count at the bottom of the info panel
    cv2.putText(annotated_frame, f"Frame: {frame_count}", (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Display calibration factor
    cv2.putText(annotated_frame, f"Cal Factor: {calibration_factor:.2f}", (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Display FPS and latency information
    if fps is not None:
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    if latency is not None:
        cv2.putText(annotated_frame, f"Latency: {latency:.2f} ms", (150, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return annotated_frame

# Helper function to convert PIL Image to base64 encoded image
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# Add new resize function
def resize_frame_to_fit(frame, max_width=1200, max_height=800):
    """Resize frame to fit within specified dimensions while maintaining aspect ratio."""
    height, width = frame.shape[:2]
    
    # If frame is already smaller than max dimensions, return as is
    if width <= max_width and height <= max_height:
        return frame
    
    # Calculate aspect ratio
    aspect_ratio = width / height
    
    # Determine new dimensions based on aspect ratio
    if width > max_width or height > max_height:
        if aspect_ratio > 1:  # Width is greater than height
            new_width = max_width
            new_height = int(new_width / aspect_ratio)
            if new_height > max_height:  # If height still exceeds max_height
                new_height = max_height
                new_width = int(new_height * aspect_ratio)
        else:  # Height is greater than width
            new_height = max_height
            new_width = int(new_height * aspect_ratio)
            if new_width > max_width:  # If width still exceeds max_width
                new_width = max_width
                new_height = int(new_width / aspect_ratio)
    else:
        return frame  # No resizing needed
    
    # Resize the frame
    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_frame

def app():
    # Initialize session state
    init_session_state()
    
    # Main title
    st.title("🚁 Object Tracker")
    
    # Create two columns for the main layout
    main_col1, main_col2 = st.columns([2, 1])
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Settings")
        
        # Video source selection
        st.subheader("Video Source")
        video_source = st.radio(
            "Select video source:",
            ["Upload Video", "Use Sample Video"]
        )
        
        if video_source == "Upload Video":
            uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
            if uploaded_file is not None:
                # Save uploaded file to a temporary file
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
                tfile.write(uploaded_file.read())
                st.session_state.video_path = tfile.name
                st.session_state.video_source = "uploaded"
        else:
            # List and select sample videos
            sample_videos = os.listdir("videos") if os.path.exists("videos") else []
            if sample_videos:
                selected_sample = st.selectbox("Select sample video:", sample_videos)
                st.session_state.video_path = os.path.join("videos", selected_sample)
                st.session_state.video_source = "sample"
            else:
                st.error("No sample videos found in the 'videos' directory")
        
        # Only show processing options if video is selected
        if st.session_state.video_path:
            st.subheader("Processing")
            
            # Detection/tracking settings
            if st.session_state.processing_phase == "upload":
                if st.button("Start Drawing Selection"):
                    st.session_state.processing_phase = "draw"
                    st.experimental_rerun()
            
            elif st.session_state.processing_phase == "draw":
                st.info("Draw a bounding box around the object to track")
                if st.button("Start Tracking"):
                    if st.session_state.user_roi is not None:
                        st.session_state.processing_phase = "track"
                        st.experimental_rerun()
                    else:
                        st.error("Please draw a bounding box first")
                
                if st.button("Reset Video"):
                    st.session_state.processing_phase = "upload"
                    st.session_state.user_roi = None
                    st.session_state.first_frame = None
                    st.experimental_rerun()
            
            elif st.session_state.processing_phase == "track":
                # Calibration settings
                st.subheader("Speed Calibration")
                new_calibration = st.slider(
                    "Calibration Factor:", 
                    min_value=0.1, 
                    max_value=5.0, 
                    value=float(st.session_state.calibration_factor),
                    step=0.1
                )
                
                if new_calibration != st.session_state.calibration_factor:
                    st.session_state.calibration_factor = new_calibration
                
                pixels_per_meter = st.number_input(
                    "Pixels per meter:", 
                    min_value=10, 
                    max_value=1000, 
                    value=int(st.session_state.pixels_per_meter),
                    step=10
                )
                
                if pixels_per_meter != st.session_state.pixels_per_meter:
                    st.session_state.pixels_per_meter = pixels_per_meter
                
                if st.button("Stop Tracking"):
                    st.session_state.processing_phase = "upload"
                    # Reset tracking data
                    st.session_state.track_history = defaultdict(list)
                    st.session_state.box_size_history = defaultdict(list)
                    st.session_state.direction_history = {}
                    st.session_state.speed_history = {}
                    st.session_state.selected_object_id = None
                    st.session_state.frame_count = 0
                    st.session_state.fps_value = 0.0
                    st.session_state.latency_value = 0.0
                    st.session_state.user_roi = None
                    st.session_state.first_frame = None
                    st.experimental_rerun()
    
    # Main content area
    with main_col1:
        # Display area for video/processing
        if st.session_state.processing_phase == "upload":
            st.info("Please select a video source and start processing")
            
            # Display a sample image or logo
            st.image("https://images.unsplash.com/photo-1473968512647-3e447244af8f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8ZHJvbmV8ZW58MHx8MHx8fDA%3D&auto=format&fit=crop&w=500&q=60", 
                    caption="Object Tracker",
                    use_column_width=True)
            
        elif st.session_state.processing_phase == "draw":
            # Load the first frame if not already loaded
            if st.session_state.first_frame is None:
                # Open video file
                cap = cv2.VideoCapture(st.session_state.video_path)
                if not cap.isOpened():
                    st.error(f"Error: Could not open video file")
                    return
                
                # Read first frame
                success, frame = cap.read()
                if not success:
                    st.error("Failed to read video")
                    cap.release()
                    return
                
                # Resize the first frame to fit UI and store it
                frame = resize_frame_to_fit(frame)
                st.session_state.first_frame = frame
                cap.release()
            
            # Show drawing interface with instructions
            st.subheader("Select Object to Track")
            st.write("Draw a bounding box around the object you want to track")
            
            # Create a placeholder for the drawing area
            drawing_placeholder = st.empty()
            
            # Use streamlit-drawable-canvas for interactive drawing if available
            if st.session_state.first_frame is not None:
                try:
                    if CANVAS_AVAILABLE:
                        # Convert the frame to PIL Image
                        pil_image = Image.fromarray(cv2.cvtColor(st.session_state.first_frame, cv2.COLOR_BGR2RGB))
                        
                        # Create a canvas for drawing
                        canvas_result = st_canvas(
                            fill_color="rgba(255, 165, 0, 0.3)",
                            stroke_width=2,
                            stroke_color="#FF0000",
                            background_image=pil_image,
                            drawing_mode="rect",
                            key="canvas",
                            height=st.session_state.first_frame.shape[0],
                            width=st.session_state.first_frame.shape[1],
                        )
                        
                        # Extract bounding box coordinates if drawn
                        if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
                            # Get rectangle coordinates
                            rect = canvas_result.json_data["objects"][0]
                            x0, y0, width, height = rect["left"], rect["top"], rect["width"], rect["height"]
                            
                            # Convert to center coordinates and dimensions format used by YOLO
                            center_x = x0 + width/2
                            center_y = y0 + height/2
                            
                            # Store in session state
                            st.session_state.user_roi = [center_x, center_y, width, height]
                            st.success("Bounding box drawn! Click 'Start Tracking' in the sidebar to begin tracking.")
                    else:
                        # Fallback to manual selection
                        drawing_placeholder.image(
                            cv2.cvtColor(st.session_state.first_frame, cv2.COLOR_BGR2RGB),
                            caption="Manual selection mode",
                            use_column_width=True
                        )
                        
                        st.write("Please enter bounding box coordinates manually:")
                        x = st.number_input("X coordinate (top-left)", 0, st.session_state.first_frame.shape[1], 100)
                        y = st.number_input("Y coordinate (top-left)", 0, st.session_state.first_frame.shape[0], 100)
                        w = st.number_input("Width", 10, st.session_state.first_frame.shape[1], 100)
                        h = st.number_input("Height", 10, st.session_state.first_frame.shape[0], 100)
                        
                        if st.button("Set Bounding Box"):
                            # Convert to center coordinates and dimensions format used by YOLO
                            center_x = x + w/2
                            center_y = y + h/2
                            st.session_state.user_roi = [center_x, center_y, w, h]
                            
                            # Draw the box on the frame to preview
                            preview_frame = st.session_state.first_frame.copy()
                            cv2.rectangle(
                                preview_frame, 
                                (int(x), int(y)), 
                                (int(x + w), int(y + h)), 
                                (0, 255, 0), 
                                2
                            )
                            drawing_placeholder.image(
                                cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB),
                                caption="Selected object to track",
                                use_column_width=True
                            )
                            
                            st.success("Bounding box set! Click 'Start Tracking' in the sidebar to begin tracking.")
                
                except Exception as e:
                    # Display the first frame for manual selection
                    drawing_placeholder.image(
                        cv2.cvtColor(st.session_state.first_frame, cv2.COLOR_BGR2RGB),
                        caption="Manual selection mode",
                        use_column_width=True
                    )
                    
                    st.write("Please enter bounding box coordinates manually:")
                    x = st.number_input("X coordinate (top-left)", 0, st.session_state.first_frame.shape[1], 100)
                    y = st.number_input("Y coordinate (top-left)", 0, st.session_state.first_frame.shape[0], 100)
                    w = st.number_input("Width", 10, st.session_state.first_frame.shape[1], 100)
                    h = st.number_input("Height", 10, st.session_state.first_frame.shape[0], 100)
                    
                    if st.button("Set Bounding Box"):
                        # Convert to center coordinates and dimensions format used by YOLO
                        center_x = x + w/2
                        center_y = y + h/2
                        st.session_state.user_roi = [center_x, center_y, w, h]
                        
                        # Draw the box on the frame to preview
                        preview_frame = st.session_state.first_frame.copy()
                        cv2.rectangle(
                            preview_frame, 
                            (int(x), int(y)), 
                            (int(x + w), int(y + h)), 
                            (0, 255, 0), 
                            2
                        )
                        drawing_placeholder.image(
                            cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB),
                            caption="Selected object to track",
                            use_column_width=True
                        )
                        
                        st.success("Bounding box set! Click 'Start Tracking' in the sidebar to begin tracking.")
                
                # Always show the manual option in an expander if the canvas is working
                if CANVAS_AVAILABLE:
                    with st.expander("Manual coordinate input (alternative)"):
                        x = st.number_input("X coordinate (top-left)", 0, st.session_state.first_frame.shape[1], 100, key="manual_x")
                        y = st.number_input("Y coordinate (top-left)", 0, st.session_state.first_frame.shape[0], 100, key="manual_y")
                        w = st.number_input("Width", 10, st.session_state.first_frame.shape[1], 100, key="manual_w")
                        h = st.number_input("Height", 10, st.session_state.first_frame.shape[0], 100, key="manual_h")
                        
                        if st.button("Set Bounding Box Manually", key="manual_set"):
                            # Convert to center coordinates and dimensions format used by YOLO
                            center_x = x + w/2
                            center_y = y + h/2
                            st.session_state.user_roi = [center_x, center_y, w, h]
                            
                            # Draw the box on the frame to preview
                            preview_frame = st.session_state.first_frame.copy()
                            cv2.rectangle(
                                preview_frame, 
                                (int(x), int(y)), 
                                (int(x + w), int(y + h)), 
                                (0, 255, 0), 
                                2
                            )
                            drawing_placeholder.image(
                                cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB),
                                caption="Selected object to track",
                                use_column_width=True
                            )
                            
                            st.success("Bounding box set! Click 'Start Tracking' in the sidebar to begin tracking.")
            
        elif st.session_state.processing_phase == "track":
            # Setup tracking display
            video_placeholder = st.empty()
            
            # Load the model
            model = YOLO("models/best.pt")
            
            # Open the video file
            cap = cv2.VideoCapture(st.session_state.video_path)
            if not cap.isOpened():
                st.error(f"Error: Could not open video file")
                return
            
            # Get video info
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create a progress bar
            progress_bar = st.progress(0)
            frame_text = st.empty()
            
            # Reset frame count if needed
            if st.session_state.frame_count == 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                # Resume from last position if applicable
                cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_count)
            
            # Variables for FPS calculation
            frame_times = []
            processing_times = []
            fps_display = 0
            latency_ms = 0
            
            # Process video frames
            try:
                prev_frame_time = time.time()
                first_frame = True
                
                while cap.isOpened():
                    # Measure frame start time for latency calculation
                    frame_start_time = time.time()
                    
                    # Read a frame
                    success, frame = cap.read()
                    if not success:
                        break
                    
                    # Resize frame to fit UI if needed
                    frame = resize_frame_to_fit(frame)
                    
                    st.session_state.frame_count += 1
                    
                    # Update progress
                    progress = min(st.session_state.frame_count / total_frames, 1.0)
                    progress_bar.progress(progress)
                    frame_text.text(f"Processing frame: {st.session_state.frame_count}/{total_frames}")
                    
                    # Calculate actual FPS (not just the video's FPS)
                    current_time = time.time()
                    time_elapsed = current_time - prev_frame_time
                    prev_frame_time = current_time
                    
                    # Add to the list of frame times (keep only last 30)
                    frame_times.append(time_elapsed)
                    if len(frame_times) > 30:
                        frame_times.pop(0)
                    
                    # Calculate average FPS from recent frames
                    if frame_times:
                        avg_frame_time = sum(frame_times) / len(frame_times)
                        fps_display = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                        # Store in session state
                        st.session_state.fps_value = fps_display
                    
                    # Run YOLO tracking with user-drawn ROI in the first frame
                    if first_frame and st.session_state.user_roi is not None:
                        try:
                            # First attempt: using boxes parameter (for newer YOLO versions)
                            results = model.track(
                                frame, 
                                persist=True, 
                                boxes=np.array([st.session_state.user_roi]),
                                classes=np.array([0])  # Assume class 0 (first class in model)
                            )
                            first_frame = False
                            
                            # Generate a tracking ID if none is present
                            if results[0].boxes.id is None:
                                st.session_state.selected_object_id = 1  # Assign ID 1 to the user-selected object
                            else:
                                st.session_state.selected_object_id = int(results[0].boxes.id[0])
                                
                        except Exception as e:
                            st.warning(f"First tracking attempt failed: {str(e)}")
                            try:
                                # Second attempt: standard tracking then find object in ROI
                                results = model.track(frame, persist=True)
                                first_frame = False
                                
                                # If no IDs yet, run standard detection
                                if results[0].boxes.id is None:
                                    results = model(frame)
                                
                                # Find object closest to our ROI
                                if len(results[0].boxes.xywh) > 0:
                                    boxes = results[0].boxes.xywh.cpu().numpy()
                                    roi_center_x, roi_center_y = st.session_state.user_roi[0], st.session_state.user_roi[1]
                                    
                                    # Calculate distance from each detected box to our ROI
                                    min_dist = float('inf')
                                    closest_idx = 0
                                    
                                    for i, box in enumerate(boxes):
                                        box_x, box_y = box[0], box[1]
                                        dist = ((box_x - roi_center_x)**2 + (box_y - roi_center_y)**2)**0.5
                                        if dist < min_dist:
                                            min_dist = dist
                                            closest_idx = i
                                    
                                    # Use the closest object's ID
                                    if results[0].boxes.id is not None:
                                        st.session_state.selected_object_id = int(results[0].boxes.id[closest_idx])
                                    else:
                                        st.session_state.selected_object_id = 1
                                else:
                                    # No objects detected
                                    st.session_state.selected_object_id = 1
                            except Exception as e2:
                                st.error(f"Backup tracking also failed: {str(e2)}")
                                # Final fallback: just run standard tracking
                                results = model.track(frame, persist=True)
                                first_frame = False
                                st.session_state.selected_object_id = 1
                    else:
                        # Continue tracking in subsequent frames
                        results = model.track(frame, persist=True)
                    
                    # Calculate processing latency
                    processing_time = (time.time() - frame_start_time) * 1000  # Convert to ms
                    processing_times.append(processing_time)
                    if len(processing_times) > 30:
                        processing_times.pop(0)
                    
                    # Calculate average latency
                    latency_ms = sum(processing_times) / len(processing_times) if processing_times else 0
                    # Store in session state
                    st.session_state.latency_value = latency_ms
                    
                    # Only continue if tracking IDs are available
                    if results[0].boxes.id is not None:
                        boxes = results[0].boxes.xywh.cpu().numpy()
                        track_ids = results[0].boxes.id.int().cpu().tolist()
                        
                        # Find our selected object
                        selected_box = None
                        for i, track_id in enumerate(track_ids):
                            if track_id == st.session_state.selected_object_id:
                                selected_box = boxes[i]
                                box_x, box_y, box_w, box_h = selected_box
                                
                                # Track position and box size history
                                track = st.session_state.track_history[track_id]
                                track.append((float(box_x), float(box_y)))  # center point
                                
                                # Track box size history (for depth analysis)
                                box_sizes = st.session_state.box_size_history[track_id]
                                box_sizes.append((float(box_x), float(box_y), float(box_w), float(box_h)))
                                
                                # Limit history length
                                max_history = 60
                                if len(track) > max_history:
                                    track.pop(0)
                                if len(box_sizes) > max_history:
                                    box_sizes.pop(0)
                                
                                # Calculate direction and speed every 5 frames
                                if st.session_state.frame_count % 5 == 0 and len(track) > 5 and len(box_sizes) > 5:
                                    # Extract center points and box dimensions for analysis
                                    track_points = [(point[0], point[1]) for point in track]
                                    box_dimensions = [(box[2], box[3]) for box in box_sizes]
                                    
                                    # Calculate direction
                                    horizontal, vertical, depth, angle = predict_direction(track_points, box_dimensions)
                                    
                                    # Calculate speed
                                    speed_kmh, px_per_frame = calculate_speed(
                                        track_points, video_fps, st.session_state.pixels_per_meter)
                                    
                                    # Apply calibration factor
                                    if speed_kmh is not None:
                                        speed_kmh *= st.session_state.calibration_factor
                                    
                                    # Store in session state
                                    st.session_state.direction_history[track_id] = {
                                        'horizontal': horizontal,
                                        'vertical': vertical,
                                        'depth': depth,
                                        'angle': angle
                                    }
                                    
                                    if speed_kmh is not None:
                                        st.session_state.speed_history[track_id] = {
                                            'kmh': speed_kmh,
                                            'px_per_frame': px_per_frame
                                        }
                        
                        # Get direction and speed data for visualization
                        direction_data = st.session_state.direction_history.get(st.session_state.selected_object_id)
                        speed_data = st.session_state.speed_history.get(st.session_state.selected_object_id)
                        
                        # Draw visualization if box was found
                        if selected_box is not None and len(st.session_state.box_size_history[st.session_state.selected_object_id]) > 0:
                            # Draw tracking visualization
                            annotated_frame = draw_tracking_visualization(
                                frame, 
                                st.session_state.selected_object_id,
                                st.session_state.track_history[st.session_state.selected_object_id],
                                st.session_state.box_size_history[st.session_state.selected_object_id],
                                direction_data,
                                speed_data,
                                st.session_state.frame_count,
                                st.session_state.calibration_factor,
                                fps_display,
                                latency_ms
                            )
                            
                            # Display the frame
                            annotated_frame = resize_frame_to_fit(annotated_frame)
                            video_placeholder.image(
                                cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                                caption=f"Tracking Object ID: {st.session_state.selected_object_id}",
                                use_column_width=True
                            )
                        else:
                            # If the object is lost, show the frame with a message
                            cv2.putText(
                                frame, 
                                "Object lost! Trying to re-detect...", 
                                (50, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                1, 
                                (0, 0, 255), 
                                2
                            )
                            frame = resize_frame_to_fit(frame)
                            video_placeholder.image(
                                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                caption="Object lost, attempting to re-detect",
                                use_column_width=True
                            )
                    else:
                        # If no tracking IDs are available, show the frame with a message
                        cv2.putText(
                            frame, 
                            "No tracking IDs found!", 
                            (50, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1, 
                            (0, 0, 255), 
                            2
                        )
                        frame = resize_frame_to_fit(frame)
                        video_placeholder.image(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                            caption="No tracking IDs found",
                            use_column_width=True
                        )
                    
                    # Control playback speed - using a dynamic sleep to maintain video FPS
                    processing_end_time = time.time()
                    processing_duration = processing_end_time - frame_start_time
                    target_duration = 1.0 / video_fps
                    
                    if processing_duration < target_duration:
                        sleep_time = target_duration - processing_duration
                        time.sleep(sleep_time)
                    
                    # Check for stop signal
                    if st.session_state.processing_phase != "track":
                        break
                    
                # Video finished
                st.success("Video processing complete")
                
            except Exception as e:
                st.error(f"Error during tracking: {str(e)}")
            
            finally:
                cap.release()
    
    # Right column for tracking information
    with main_col2:
        if st.session_state.processing_phase == "track":
            # Display tracking information in a more compact format
            st.subheader("Tracking Information")
            
            # Direction information
            direction_data = st.session_state.direction_history.get(st.session_state.selected_object_id)
            if direction_data:
                st.markdown("### Direction Analysis")
                st.metric("Horizontal", direction_data['horizontal'])
                st.metric("Vertical", direction_data['vertical'])
                st.metric("Depth", direction_data['depth'])
                st.metric("Angle", f"{direction_data['angle']:.1f}°")
            
            # Speed information
            speed_data = st.session_state.speed_history.get(st.session_state.selected_object_id)
            if speed_data:
                st.markdown("### Speed Analysis")
                st.metric("Speed", f"{speed_data['kmh']:.1f} km/h")
                st.metric("Movement", f"{speed_data['px_per_frame']:.1f} px/frame")
                st.metric("Calibration", f"{st.session_state.calibration_factor:.2f}")
                st.metric("Frame", f"{st.session_state.frame_count}")
            
            # Performance metrics
            st.markdown("### Performance Metrics")
            st.metric("FPS", f"{st.session_state.fps_value:.1f}")
            st.metric("Latency", f"{st.session_state.latency_value:.1f} ms")

if __name__ == "__main__":
    app() 