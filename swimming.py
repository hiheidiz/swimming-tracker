import cv2

cap = cv2.VideoCapture("swimming_video.mp4")

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

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    success, box = tracker.update(frame)
    
    if success:
        x, y, w, h = [int(v) for v in box]
        cx, cy = x + w // 2, y + h // 2
        cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
    
    out.write(frame)  # save each frame to output video
    cv2.imshow("Swimmer Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Video saved as tracked_swimmer.mp4")
