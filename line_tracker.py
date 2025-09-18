import cv2
import numpy as np

cap = cv2.VideoCapture("tracked_swimmer_works.mp4")

# --- Step 1: Select line manually (2 clicks) ---
points = []

def select_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Select Line", frame)

# Get first frame
ret, frame = cap.read()
if not ret:
    print("Error: cannot read video")
    exit()

cv2.imshow("Select Line", frame)
cv2.setMouseCallback("Select Line", select_point)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(points) < 2:
    print("Error: You must click two points on the line")
    exit()

# Reference line (user-selected)
x1_ref, y1_ref = points[0]
x2_ref, y2_ref = points[1]
ref_vector = np.array([x2_ref - x1_ref, y2_ref - y1_ref])

# --- Step 2: Track line across frames ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100,
                            minLineLength=100, maxLineGap=20)

    chosen_line = None
    min_dist = float("inf")

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Compute midpoint of detected line
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            # Compute midpoint of reference line
            mx_ref, my_ref = (x1_ref + x2_ref) / 2, (y1_ref + y2_ref) / 2
            # Distance between midpoints
            dist = np.sqrt((mx - mx_ref) ** 2 + (my - my_ref) ** 2)

            if dist < min_dist:
                min_dist = dist
                chosen_line = (x1, y1, x2, y2)

    if chosen_line:
        x1, y1, x2, y2 = chosen_line
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    cv2.imshow("Line Tracking", frame)
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
