import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
import tempfile
import os
from collections import defaultdict
import time

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
        st.session_state.processing_phase = "upload"  # Phases: upload, detect, select, track
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0

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
                              frame_count, calibration_factor):
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
    info_panel_height = 150  # Adjust based on how much info you're displaying
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
    
    return annotated_frame

def app():
    # Initialize session state
    init_session_state()
    
    # Main title
    st.title("🚁 Object Tracker")
    
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
                if st.button("Start Object Detection"):
                    st.session_state.processing_phase = "detect"
                    st.rerun()()
            
            elif st.session_state.processing_phase == "detect" or st.session_state.processing_phase == "select":
                if st.button("Reset Video"):
                    st.session_state.processing_phase = "upload"
                    st.rerun()()
            
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
                    st.rerun()()
    
    # Main content area
    main_container = st.container()
    
    with main_container:
        # Display area for video/processing
        if st.session_state.processing_phase == "upload":
            st.info("Please select a video source and start processing")
            
            # Display a sample image or logo
            st.image("https://images.unsplash.com/photo-1473968512647-3e447244af8f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8ZHJvbmV8ZW58MHx8MHx8fDA%3D&auto=format&fit=crop&w=500&q=60", 
                    caption="Object Tracker")
            
        elif st.session_state.processing_phase == "detect":
            st.info("Analyzing video for objects to track...")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Load the model
            model = YOLO("models/exp1.pt")
            
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
                
            # Process with YOLO for initial detection
            results = model.track(frame, persist=True)
            
            # Generate annotated frame with all detections
            annotated_frame, object_ids = select_object_from_detections(frame, results)
            
            # Display frame with all detections
            video_placeholder = st.empty()
            video_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), 
                                   caption="Detected Objects")
            
            status_text.text("Detection complete. Please select an object to track.")
            progress_bar.progress(100)
            
            # Selection interface
            st.subheader("Select Object to Track")
            
            if object_ids:
                selected_id = st.selectbox("Select object ID to track:", object_ids)
                
                if st.button("Start Tracking"):
                    st.session_state.selected_object_id = int(selected_id)
                    st.session_state.processing_phase = "track"
                    cap.release()
                    st.rerun()()
            else:
                st.warning("No objects detected in the first frame. Try a different video or frame.")
            
        elif st.session_state.processing_phase == "track":
            # Setup tracking display
            video_placeholder = st.empty()
            info_col1, info_col2 = st.columns(2)
            
            with info_col1:
                direction_info = st.empty()
            
            with info_col2:
                speed_info = st.empty()
            
            # Load the model
            model = YOLO("models/exp1.pt")
            
            # Open the video file
            cap = cv2.VideoCapture(st.session_state.video_path)
            if not cap.isOpened():
                st.error(f"Error: Could not open video file")
                return
            
            # Get video info
            fps = cap.get(cv2.CAP_PROP_FPS)
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
            
            # Process video frames
            try:
                while cap.isOpened():
                    # Read a frame
                    success, frame = cap.read()
                    if not success:
                        break
                    
                    st.session_state.frame_count += 1
                    
                    # Update progress
                    progress = min(st.session_state.frame_count / total_frames, 1.0)
                    progress_bar.progress(progress)
                    frame_text.text(f"Processing frame: {st.session_state.frame_count}/{total_frames}")
                    
                    # Run YOLO tracking
                    results = model.track(frame, persist=True)
                    
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
                                        track_points, fps, st.session_state.pixels_per_meter)
                                    
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
                                st.session_state.calibration_factor
                            )
                            
                            # Display the frame
                            video_placeholder.image(
                                cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                                caption=f"Tracking Object ID: {st.session_state.selected_object_id}"
                            )
                            
                            # Update info displays
                            if direction_data:
                                direction_html = f"""
                                <div style="padding: 10px; border-radius: 5px; background-color: #f0f0f0;">
                                    <h4>Direction Analysis</h4>
                                    <p>Horizontal: <b>{direction_data['horizontal']}</b></p>
                                    <p>Vertical: <b>{direction_data['vertical']}</b></p>
                                    <p>Depth: <b>{direction_data['depth']}</b></p>
                                    <p>Angle: <b>{direction_data['angle']:.1f}°</b></p>
                                </div>
                                """
                                direction_info.markdown(direction_html, unsafe_allow_html=True)
                            
                            if speed_data:
                                speed_html = f"""
                                <div style="padding: 10px; border-radius: 5px; background-color: #f0f0f0;">
                                    <h4>Speed Analysis</h4>
                                    <p>Speed: <b>{speed_data['kmh']:.1f} km/h</b></p>
                                    <p>Movement: <b>{speed_data['px_per_frame']:.1f} px/frame</b></p>
                                    <p>Calibration: <b>{st.session_state.calibration_factor:.2f}</b></p>
                                </div>
                                """
                                speed_info.markdown(speed_html, unsafe_allow_html=True)
                    
                    # Control playback speed
                    time.sleep(1/fps)  # Play at original speed
                    
                    # Check for stop signal
                    if st.session_state.processing_phase != "track":
                        break
                    
                # Video finished
                st.success("Video processing complete")
                
            except Exception as e:
                st.error(f"Error during tracking: {str(e)}")
            
            finally:
                cap.release()

if __name__ == "__main__":
    app() 