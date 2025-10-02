import cv2
import csv

cap = cv2.VideoCapture("fly.mp4")

# Read first frame
ret, frame = cap.read()
if not ret:
    print("Error: Could not read video")
    exit()

bbox = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)

# Create tracker (use legacy if needed)
tracker = cv2.legacy.TrackerCSRT_create()
tracker.init(frame, bbox)

# --- Setup VideoWriter ---
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # codec
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("tracked_swimmer.mp4", fourcc, fps, (width, height))

# --- Prepare CSV storage ---
positions = []  # list of (frame_number, cx, cy)
frame_number = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("📹 Reached end of video")
        break
    
    success, box = tracker.update(frame)
    
    if success:
        x, y, w, h = [int(v) for v in box]
        cx, cy = x + w // 2, y + h // 2
        cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
        positions.append((frame_number, cx, cy))
    else:
        # If tracking fails, record None
        positions.append((frame_number, None, None))
    
    out.write(frame)  # save each frame to output video
    cv2.imshow("Swimmer Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_number += 1

cap.release()
out.release()
cv2.destroyAllWindows()

# --- Write CSV ---
with open("tracked_positions.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["frame", "x", "y"])
    writer.writerows(positions)

print("✅ Video saved as tracked_swimmer.mp4")
print("✅ Positions saved as tracked_positions.csv")
