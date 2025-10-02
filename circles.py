import cv2
import numpy as np
from collections import deque

video_path = "circles.mov"
output_path = "circles_tracked.mp4"

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

min_radius = 8
max_radius = 20
history_length = 5  # number of frames to average movement

# Store previous positions for movement tracking
position_history = deque(maxlen=history_length)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # Detect circles
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=15,
        param1=50,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    current_positions = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for x, y, r in circles:
            if min_radius <= r <= max_radius:
                current_positions.append(np.array([x, y]))
    current_positions = np.array(current_positions)

    highlight_idx = -1
    if len(position_history) > 0 and len(current_positions) > 0:
        # Compute average movement over history
        avg_displacements = []
        for curr in current_positions:
            disp = 0
            for prev_positions in position_history:
                distances = np.linalg.norm(prev_positions - curr, axis=1)
                disp += distances.min()
            avg_displacements.append(disp / len(position_history))
        highlight_idx = int(np.argmax(avg_displacements))

    # Draw circles: red normally, blue for the fastest-moving
    for i, (x, y) in enumerate(current_positions):
        color = (0, 0, 255)  # red
        if i == highlight_idx:
            color = (255, 0, 0)  # blue
        cv2.circle(frame, (x, y), 10, color, 2)

    # Update history
    if len(current_positions) > 0:
        position_history.append(current_positions.copy())

    out.write(frame)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
