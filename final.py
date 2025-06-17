import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict
import time
from collections import deque
import random

def get_color(track_id):
    random.seed(track_id)  # Makes the color consistent for the same ID
    return (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255)) 
# Parameters for traffic jam detection

vehicle_rate_log = deque(maxlen=20)  # Store last 6 (timestamp, count) entries
jam_threshold = 0.3

# Initialize model and tracker
model = YOLO('best_mosaic.pt')
tracker = DeepSort(max_age=30)

# Store past trajectories
trajectories = defaultdict(list)
id_colors = {}  # Store color for each ID
timestamps = {} 
LINE_Y = 500
# Load your video
cap = cv2.VideoCapture("traffic.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('output_analysis.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
total_count = 0 
visible_count = 0 
last_positions={}
FAST_THRESHOLD=150
SLOW_THRESHOLD=60

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get YOLO detections
    results = model(frame)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

    # Update tracks
    counted_ids = set()
    tracks = tracker.update_tracks(detections, frame=frame)
    active_track_ids = set()  # <--- FIXED: properly initialize
    cv2.line(frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (0, 255, 255), 2)

     # ← add before the loop

    visible_count = 0  # count of vehicles currently in frame

    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue

        track_id = track.track_id
        active_track_ids.add(track_id)

        l, t, r, b = track.to_ltrb()
        center_x = int((l + r) / 2)
        center_y = int((t + b) / 2)

        # Color assignment
        if track_id not in id_colors:
            id_colors[track_id] = get_color(track_id)
        color = id_colors[track_id]


        # Count logic using movement across the line
        prev_pos = last_positions.get(track_id)
        if prev_pos:
            prev_y = prev_pos[1]
            # Count if object moved from above to below the line
            if prev_y < LINE_Y and center_y >= LINE_Y and track_id not in counted_ids:
                counted_ids.add(track_id)
                total_count += 1
        last_positions[track_id] = (center_x, center_y)

        visible_count += 1
            
        
    
    # Predict next 5 future points (or as many as you want)
    
        # Track current frame vehicle count
        
        # Get predicted next box
    #     predicted_box = track.get_kalman_prediction()
    #     if predicted_box:
    #         pl, pt, pr, pb = predicted_box
    #         pl, pt, pr, pb = map(int, predicted_box)
    #         center_pred = ((pl + pr) / 2, (pt + pb) / 2)

    # # Compute velocity vector (dx, dy)
    #         dx = center_pred[0] - center_x
    #         dy = center_pred[1] - center_y

    # # Optional: compute speed (magnitude of velocity)
    #         distance = (dx * 2 + dy * 2) ** 0.5

    #         velocity=distance/dt

        
    #         cv2.putText(frame, f"Vel:{velocity}", (int(center_x), int(center_y) - 10),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


        


        # Store current position
        trajectories[track_id].append((center_x, center_y))
        # --- Access Kalman filter velocity (vx, vy) ---
        if hasattr(track, 'mean'):  # Ensure internal state exists
            vx = track.mean[4]  # Velocity in x-direction (pixels/frame)
            vy = track.mean[5]  # Velocity in y-direction
            vel = (vx*2 + vy*2) * 0.5  # Euclidean speed in pixels/frame

            # Convert to pixels/second using FPS
             # Fallback to 30 if unknown
            speed_kalman = vel * fps
            if speed_kalman < SLOW_THRESHOLD:
                speed_category = "Slow"
                speed_color = (255, 0, 0)
            elif speed_kalman > FAST_THRESHOLD:
                speed_category = "Fast"
                speed_color = (0, 0, 255)
            else:
                speed_category = "Medium"
                speed_color = (0, 255, 255)

            cv2.putText(frame, f"{speed_category}", (center_x, center_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, speed_color, 2)
            # Display speed from Kalman velocity
            cv2.putText(frame, f"KF:{speed_kalman:.1f} px/s", (center_x,center_y+25 ),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 128, 255), 1)


        # Draw bounding box
        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (int(l), int(t) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Draw trajectory history
        for i in range(1, len(trajectories[track_id])):
            cv2.line(frame, trajectories[track_id][i - 1], trajectories[track_id][i], color, 2)
            
        # Draw predicted next location
        

        for i in range(1, len(trajectories[track_id])):
            cv2.line(frame, trajectories[track_id][i - 1], trajectories[track_id][i], color, 2)
            cv2.arrowedLine(frame, trajectories[track_id][i - 1], trajectories[track_id][i], (0, 255, 0), 2, tipLength=0.5)
        #     cv2.rectangle(frame, (pl, pt), (pr, pb), (0, 0, 255), 2)
            # cv2.putText(frame, "Pred", (pl, pt - 5),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    #  Move this outside the track loop
    for track_id in list(trajectories.keys()):
        if track_id not in active_track_ids:
            del trajectories[track_id]
    cv2.putText(frame, f"Total Vehicles Passed: {total_count}", (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, f"Vehicles in Frame: {visible_count}", (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    # Traffic Flow Rate Calculation
    current_time = time.time()

    # Only log once per second
    if not vehicle_rate_log or current_time - vehicle_rate_log[-1][0] >= 1:
        vehicle_rate_log.append((current_time, total_count))

    # Compute smoothed dN/dt if enough data
    if len(vehicle_rate_log) >= 2:
        rates = []
        for i in range(len(vehicle_rate_log) - 1):
            t1, n1 = vehicle_rate_log[i]
            t2, n2 = vehicle_rate_log[i + 1]
            dt = t2 - t1
            dn = n2 - n1
            if dt > 0:
                rates.append(dn / dt)

        if rates:
            avg_rate = sum(rates) / len(rates)

            # Display the smoothed rate
            cv2.putText(frame, f"Flow Rate: {avg_rate:.2f} veh/s", (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 2)

            # Jam Detection
            if avg_rate < jam_threshold and avg_rate!=0:
                cv2.putText(frame, "Low flow", (30, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    cv2.line(frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (0, 255, 255), 2)
    out.write(frame)
    cv2.imshow("Tracking with Trajectories + Prediction", frame)
    if cv2.waitKey(1) == 27:  # ESC to quit
                    break

cap.release()
out.release()
cv2.destroyAllWindows()