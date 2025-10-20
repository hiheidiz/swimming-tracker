import cv2
import csv
import numpy as np

# --- SETTINGS ---
video_path = "100325_ALUMPOOL_SWIMMER_005_.mp4"  # Change to your video file
output_csv = "tracking_log_TEST.csv"
output_video = "005_TRACKED_TEST.mp4"
edge_margin = 5

def create_tracker():
    return cv2.TrackerCSRT_create()

# Initialize video and MultiTracker
cap = cv2.VideoCapture(video_path)
multi_tracker = cv2.legacy.MultiTracker_create()

# --- INIT ---
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
dt = 1 / fps if fps > 0 else 1/30
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

trackers = []         # list of tracker objects
object_ids = []       # parallel list of IDs for each tracker
positions = {}        # {obj_id: list of positions}
log_data = []
frame_idx = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    # --- Update trackers ---
    to_remove = []  # list of indices to delete
    for i, tracker in enumerate(trackers):
        ok, box = tracker.update(frame)
        if ok:
            x, y, w, h = [int(v) for v in box]
            cx, cy = x + w/2, y + h/2
            obj_id = object_ids[i]

            # --- Check if reached right edge ---
            if x + w >= width - edge_margin:
                print(f"Object {obj_id} reached right edge at frame {frame_idx}, removing.")
                to_remove.append(i)
                continue  # skip drawing/logging for this frame

            # --- Position & Velocity Logging ---
            if obj_id not in positions:
                positions[obj_id] = []

            positions[obj_id].append((cx, cy))
            if len(positions[obj_id]) >= 2:
                (x1, y1), (x0, y0) = positions[obj_id][-1], positions[obj_id][-2]

            # --- Draw Tracker ---
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"Obj {obj_id}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw short trajectory
            if len(positions[obj_id]) > 5:
                for j in range(-5, -1):
                    p1 = tuple(map(int, positions[obj_id][j]))
                    p2 = tuple(map(int, positions[obj_id][j+1]))
                    cv2.line(frame, p1, p2, (0, 200, 255), 2)

            # --- Log Data ---
            log_data.append([frame_idx, obj_id, cx, cy])

    # --- Remove trackers safely ---
    if to_remove:
        for idx in sorted(to_remove, reverse=True):
            del trackers[idx]
            del object_ids[idx]
        # No need to rebuild MultiTracker in this version; lists are synced

    # --- Display and Video Output ---
    cv2.putText(frame, "Press 'a' to add, 'q' to quit", (20,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    out.write(frame)
    cv2.imshow("Multi-Object Tracking", frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('a'):
        bbox = cv2.selectROI("Multi-Object Tracking", frame, fromCenter=False)
        if bbox != (0,0,0,0):
            tracker = create_tracker()
            tracker.init(frame, bbox)
            trackers.append(tracker)
            new_id = max(object_ids, default=-1) + 1
            object_ids.append(new_id)
            print(f"Added tracker #{new_id}")
        else:
            print("No ROI selected — skipping.")
    elif key == ord('q'):
        break

    frame_idx += 1

# --- Cleanup ---
cap.release()
out.release()
cv2.destroyAllWindows()

# --- Save CSV ---
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Frame", "ObjectID", "X", "Y"])
    writer.writerows(log_data)
