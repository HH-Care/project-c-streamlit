# pip install opencv-python==4.11.0.86
# pip install --upgrade ultralytics==8.3.97

from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import math

# Global variables for object selection
selecting = False
selection_complete = False
start_point = None
end_point = None
selected_object_id = None
selection_method = "bbox"  # "bbox" or "click"
frame = None


# Direction prediction function
def predict_direction(points, box_sizes, num_points=5):
    """
    Determine direction of movement based on recent trajectory points
    and changing object size to infer 3D movement.

    Args:
        points: List of (x,y) center coordinates
        box_sizes: List of (width, height) values corresponding to bounding box sizes
        num_points: Number of points to consider for direction analysis

    Returns:
        horizontal_direction: Left/Right direction
        vertical_direction: Up/Down direction (in image plane)
        depth_direction: Whether object is moving toward or away from camera
        angle: Angle of movement in degrees
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
    """
    Calculate speed based on points trajectory.

    Args:
        points: List of (x,y) center coordinates
        fps: Frames per second of the video
        pixels_per_meter: Conversion from pixels to meters
        num_points: Number of points to consider

    Returns:
        speed_kmh: Speed in km/h
        distance_pixels: Distance in pixels per frame
    """
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


# Callback function for mouse events (for interactive selection)
def select_object(event, x, y, flags, param):
    global selecting, selection_complete, start_point, end_point, frame, selection_method

    if selection_method == "bbox":
        # Bounding box selection
        if event == cv2.EVENT_LBUTTONDOWN:
            selecting = True
            selection_complete = False
            start_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and selecting:
            end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            selecting = False
            selection_complete = True
            end_point = (x, y)

    elif selection_method == "click" and event == cv2.EVENT_LBUTTONDOWN:
        # Click-based selection with edge detection
        mask = np.zeros(frame.shape[:2], np.uint8)

        # Parameters for grabCut
        rect = (x - 40, y - 40, 80, 80)  # Create a region around the clicked point
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        # Apply grabCut to segment the object
        cv2.grabCut(frame, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        # Find contours in the mask
        contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Find the largest contour (presumably our object)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            start_point = (x, y)
            end_point = (x + w, y + h)
            selection_complete = True


def main():
    global frame, selection_method, selecting, selection_complete, start_point, end_point, selected_object_id

    # Load the YOLOv8 model
    model = YOLO("models/exp1.pt")

    # Open the video file
    video_path = "videos/deone.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Optional: Save the output video
    save_output = True
    output_path = "yyyfina_output_tracked_video.mp4"
    writer = None

    # Store the track history
    track_history = defaultdict(list)
    box_size_history = defaultdict(list)

    # Store direction and speed data
    direction_history = {}
    speed_history = {}

    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Conversion factor: pixels to meters
    # This is an approximation and should be calibrated for your specific video
    # For example, if you know an object in the scene is 1 meter and it's 100 pixels wide
    # then pixels_per_meter would be 100
    pixels_per_meter = 100  # This needs to be calibrated for accurate speed

    # Create a calibration parameter that the user can adjust
    calibration_factor = 1.0  # Will be used to manually adjust speed calculations

    # Enable speed calibration mode
    calibration_mode = False

    # Create window and set mouse callback for selection
    cv2.namedWindow('Select Object')
    cv2.setMouseCallback('Select Object', select_object)

    # First frame for object selection
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from video source")
        return

    selection_frame = frame.copy()

    # Selection phase
    while not selection_complete:
        display_frame = selection_frame.copy()

        if selection_method == "bbox" and start_point and end_point:
            cv2.rectangle(display_frame, start_point, end_point, (0, 255, 0), 2)

        # Display instructions
        if selection_method == "bbox":
            instructions = "Draw a box around the object to track (M to switch to click mode, Q to quit)"
        else:
            instructions = "Click on the object to track (M to switch to box mode, Q to quit)"

        cv2.putText(display_frame, instructions, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Select Object', display_frame)
        key = cv2.waitKey(1) & 0xFF

        # Toggle selection method
        if key == ord('m'):
            selection_method = "click" if selection_method == "bbox" else "bbox"
            print(f"Changed selection method to: {selection_method}")
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    cv2.destroyWindow('Select Object')

    # Calculate initial position of selected object
    if start_point and end_point:
        init_x = (start_point[0] + end_point[0]) // 2
        init_y = (start_point[1] + end_point[1]) // 2
        init_w = abs(end_point[0] - start_point[0])
        init_h = abs(end_point[1] - start_point[1])  # Fixed this line - was using start_point[0]

        # Reset video to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Initialize tracking with selected object
        success, frame = cap.read()
        results = model.track(frame, persist=True)

        # Find closest detection to our selection
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()

            min_distance = float('inf')
            for i, box in enumerate(boxes):
                x, y, w, h = box
                distance = np.sqrt((x - init_x) ** 2 + (y - init_y) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    selected_object_id = int(ids[i])

    # Create window for tracked video
    cv2.namedWindow('Object Tracking')

    # Get video dimensions for text positioning
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Loop through video frames
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1

        # Run YOLO tracking
        results = model.track(frame, persist=True)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id

        # Annotate frame
        annotated_frame = frame.copy()  # Use a clean copy for custom annotation

        # Process key presses for calibration mode
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            calibration_mode = not calibration_mode
            print(f"Calibration mode: {'ON' if calibration_mode else 'OFF'}")
        elif calibration_mode:
            if key == ord('+') or key == ord('='):
                calibration_factor *= 1.1  # Increase by 10%
                print(f"Calibration factor: {calibration_factor:.2f}")
            elif key == ord('-'):
                calibration_factor *= 0.9  # Decrease by 10%
                print(f"Calibration factor: {calibration_factor:.2f}")
        elif key == ord('q'):
            break

        # Create a dark overlay at the TOP left for text display
        info_panel_height = 150  # Adjust based on how much info you're displaying
        info_panel_width = 300  # Adjust width as needed
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (10, 10),
                      (10 + info_panel_width, 10 + info_panel_height), (220, 220, 220), -1)

        # Apply the overlay with transparency
        alpha = 0.7  # Transparency factor
        cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)

        # Only continue if tracking IDs are available
        if track_ids is not None:
            track_ids = track_ids.int().cpu().tolist()

            for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
                # Only draw the selected object if we have one
                if selected_object_id is not None and track_id != selected_object_id:
                    continue

                x, y, w, h = box

                # Draw bounding box for tracked object
                x1, y1 = int(x - w / 2), int(y - h / 2)
                x2, y2 = int(x + w / 2), int(y + h / 2)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Track position history
                track = track_history[track_id]
                track.append((float(x), float(y)))  # center point

                # Track box size history (for depth analysis)
                box_sizes = box_size_history[track_id]
                box_sizes.append((float(w), float(h)))

                # Limit history length
                max_history = 60  # Adjust based on your needs
                if len(track) > max_history:
                    track.pop(0)
                if len(box_sizes) > max_history:
                    box_sizes.pop(0)

                # Draw the path (tracking line)
                if len(track) > 1:
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False,
                                  color=(230, 230, 230), thickness=2)

                # Calculate and display 3D direction every 5 frames
                if frame_count % 5 == 0 and len(track) > 5 and len(box_sizes) > 5:
                    horizontal, vertical, depth, angle = predict_direction(track, box_sizes)

                    # Calculate speed in km/h
                    speed_kmh, px_per_frame = calculate_speed(track, fps, pixels_per_meter)

                    # Apply calibration factor
                    if speed_kmh is not None:
                        speed_kmh *= calibration_factor

                    # Add to direction and speed history
                    direction_history[track_id] = {
                        'horizontal': horizontal,
                        'vertical': vertical,
                        'depth': depth,
                        'angle': angle
                    }

                    if speed_kmh is not None:
                        speed_history[track_id] = {
                            'kmh': speed_kmh,
                            'px_per_frame': px_per_frame
                        }

                # Draw a direction arrow for 2D plane visualization
                if track_id in direction_history and len(track) > 5:
                    dir_data = direction_history[track_id]
                    # Skip if completely stationary
                    if (dir_data['horizontal'] != "Stationary" or
                            dir_data['vertical'] != "Stationary"):

                        angle = dir_data['angle']
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
                if track_id in direction_history:
                    depth_dir = direction_history[track_id]['depth']
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

                # Display information in the TOP-left overlay area (All text in BLACK)
                y_position = 35  # Starting position for text

                # Display object ID
                cv2.putText(annotated_frame, f"Object ID: {track_id}", (20, y_position),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                y_position += 20

                # Display direction information if available
                if track_id in direction_history:
                    dir_data = direction_history[track_id]
                    direction_text = f"Direction: {dir_data['horizontal']} | {dir_data['vertical']}"
                    cv2.putText(annotated_frame, direction_text, (20, y_position),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    y_position += 20

                    depth_text = f"Depth: {dir_data['depth']}"
                    cv2.putText(annotated_frame, depth_text, (20, y_position),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    y_position += 20

                # Display speed information if available
                if track_id in speed_history:
                    speed_data = speed_history[track_id]
                    speed_text = f"Speed: {speed_data['kmh']:.1f} km/h"
                    cv2.putText(annotated_frame, speed_text, (20, y_position),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    y_position += 20

        # Display frame count and calibration status at the bottom of the info panel
        cv2.putText(annotated_frame, f"Frame: {frame_count}", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        if calibration_mode:
            cv2.putText(annotated_frame,
                        f"CAL MODE (Factor: {calibration_factor:.2f}, +/- to adjust)",
                        (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Show the frame
        cv2.imshow('Object Tracking', annotated_frame)

        # Save the frame to output video
        if save_output:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            writer.write(annotated_frame)

    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
