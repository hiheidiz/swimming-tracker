#!/usr/bin/env python3
"""
tracker_final_continuous.py

Single-point tracking with manual start/stop capability.
Allows tracking an object that leaves and re-enters the frame by manually
stopping and restarting tracking.

Modifications:
- A: Use CSRT trackers for all points
- B: Apply Gaussian blur preprocessing to frames before tracker.update()
- D: Apply EWMA smoothing per tracked point at capture time (configurable alpha)
- E: Manual start/stop tracking capability for continuous tracking

Usage:
    python tracker_final_continuous.py --video path/to/video.mp4
    python tracker_final_continuous.py --video test.mp4 --out results.csv --smooth 5 --ewma-alpha 0.25 --blur-k 7

Controls:
    'a' - Start/resume tracking (select ROI when stopped)
    's' - Stop tracking (keep data, mark as lost)
    'q' - Quit and save results
"""
import cv2
import numpy as np
import argparse
import pandas as pd
import csv
from scipy.ndimage import median_filter
import os
import sys
from datetime import datetime

# ---------------------------
# Helpers / Utilities
# ---------------------------
def get_tracking_output_path(filename, subdir='outputs'):
    """
    Generate a timestamped output path in the tracking/outputs directory.
    
    Args:
        filename: Base filename (e.g., 'results.csv' or 'output.mp4')
        subdir: Subdirectory name (default: 'outputs')
    
    Returns:
        Full path to output file with timestamp
    """
    # Get script directory (should be tracking/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create outputs subdirectory inside tracking directory
    outputs_dir = os.path.join(script_dir, subdir)
    
    # Ensure outputs directory exists
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Split filename into name and extension
    name, ext = os.path.splitext(filename)
    
    # Create timestamped filename
    timestamped_filename = f"{name}_{timestamp}{ext}"
    
    return os.path.join(outputs_dir, timestamped_filename)


def create_csrt_tracker():
    """
    Create a CSRT tracker in a way that works across OpenCV versions.
    """
    # prefer legacy API if available (OpenCV >=4.5 typically)
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    elif hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    else:
        raise RuntimeError("CSRT tracker not available in your OpenCV build")


def ensure_odd_positive(k):
    k = int(k)
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    return k


# ---------------------------
# (unchanged) line detection & smoothing utilities (kept as-is)
# ---------------------------
def detect_vertical_line_in_region(frame, region_bbox, prev_x=None, max_jump=40, 
                                   weight_decay=0.01, bottom_bias=2.0, min_line_length=40):
    rx, ry, rw, rh = region_bbox
    roi = frame[ry:ry+rh, rx:rx+rw]
    b, g, r = cv2.split(roi)
    b_inv = 255 - b
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.addWeighted(b_inv, 0.6, gray, 0.4, 0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(enhanced)
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.8, beta=-20)
    enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
    edges = cv2.Canny(enhanced, 20, 100)

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                            threshold=40, minLineLength=min_line_length, maxLineGap=10)

    x_line = None
    if lines is not None:
        candidates = []
        weights_combined = []
        line_lengths = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) > 87:
                x_mid_local = (x1 + x2) / 2
                y_mid_local = (y1 + y2) / 2
                line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                x_mid_global = x_mid_local + rx
                margin = 5
                if not (rx - margin <= x_mid_global <= rx + rw + margin):
                    continue

                if prev_x is not None:
                    d = abs(x_mid_global - prev_x)
                    proximity_weight = np.exp(-weight_decay * d**2)
                else:
                    proximity_weight = 1.0

                normalized_y = y_mid_local / rh
                bottom_weight = 1 + bottom_bias * (normalized_y - 0.66)
                bottom_weight = max(bottom_weight, 0.1)
                length_weight = min(line_length / (rh * 0.5), 2.0)
                combined_weight = proximity_weight * bottom_weight * length_weight
                candidates.append(x_mid_global)
                weights_combined.append(combined_weight)
                line_lengths.append(line_length)

        if candidates:
            candidates = np.array(candidates)
            weights_combined = np.array(weights_combined)
            line_lengths = np.array(line_lengths)

            if np.sum(weights_combined) > 0:
                weighted_x = np.sum(candidates * weights_combined) / np.sum(weights_combined)

                if weighted_x < rx or weighted_x > rx + rw:
                    if prev_x is not None and rx <= prev_x <= rx + rw:
                        weighted_x = prev_x
                    else:
                        valid_mask = (candidates >= rx) & (candidates <= rx + rw)
                        if valid_mask.any():
                            best_idx = np.argmax(line_lengths * valid_mask)
                            weighted_x = candidates[best_idx]
                        else:
                            return None

                if prev_x is not None:
                    jump_size = abs(weighted_x - prev_x)
                    if jump_size > max_jump:
                        return prev_x
                    elif jump_size > max_jump * 0.5:
                        alpha = 0.3
                        weighted_x = alpha * weighted_x + (1 - alpha) * prev_x

                x_line = weighted_x

    if x_line is None:
        return None

    return x_line


def linear_interpolate_positions(times, xs, max_gap_seconds=0.2):
    xs = xs.copy()
    isnan = np.isnan(xs)
    if not isnan.any():
        return xs
    n = len(xs)
    i = 0
    while i < n:
        if np.isnan(xs[i]):
            j = i
            while j < n and np.isnan(xs[j]):
                j += 1
            if i > 0 and j < n:
                gap_duration = times[j] - times[i-1]
            elif i > 0 and j == n:
                gap_duration = times[-1] - times[i-1]
            elif i == 0 and j < n:
                gap_duration = times[j] - times[0]
            else:
                gap_duration = np.inf
            if gap_duration <= max_gap_seconds and i > 0 and j < n:
                xs[i:j] = np.linspace(xs[i-1], xs[j], j - i)
            i = j
        else:
            i += 1
    return xs


def interpolate_line_positions_bidirectional(line_positions, max_gap_frames=30):
    positions = line_positions.copy()
    n = len(positions)
    valid_mask = ~np.isnan(positions)
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return positions
    i = 0
    while i < n:
        if np.isnan(positions[i]):
            gap_start = i
            while i < n and np.isnan(positions[i]):
                i += 1
            gap_end = i
            has_before = gap_start > 0 and not np.isnan(positions[gap_start - 1])
            has_after = gap_end < n and not np.isnan(positions[gap_end])
            gap_size = gap_end - gap_start
            if has_before and has_after and gap_size <= max_gap_frames:
                before_val = positions[gap_start - 1]
                after_val = positions[gap_end]
                positions[gap_start:gap_end] = np.linspace(before_val, after_val, gap_size + 2)[1:-1]
            elif has_before and gap_size <= max_gap_frames // 2:
                positions[gap_start:gap_end] = positions[gap_start - 1]
            elif has_after and gap_size <= max_gap_frames // 2:
                positions[gap_start:gap_end] = positions[gap_end]
        else:
            i += 1
    return positions


def remove_line_outliers(line_positions, window=11, threshold=2.5):
    positions = line_positions.copy()
    valid_mask = ~np.isnan(positions)
    if valid_mask.sum() < window:
        return positions
    temp = positions.copy()
    temp[~valid_mask] = np.nanmedian(positions)
    filtered = median_filter(temp, size=window, mode='nearest')
    deviation = np.abs(positions - filtered)
    valid_deviations = deviation[valid_mask]
    if len(valid_deviations) == 0:
        return positions
    mad = np.median(valid_deviations)
    outlier_threshold = threshold * mad * 1.4826
    is_outlier = deviation > outlier_threshold
    positions[is_outlier & valid_mask] = np.nan
    return positions


def aggressive_smooth(x, gaussian_window=31, moving_avg_window=15):
    valid_mask = ~np.isnan(x)
    if not valid_mask.any():
        return x
    x_filled = x.copy()
    if np.isnan(x_filled).any():
        valid_indices = np.where(valid_mask)[0]
        valid_values = x[valid_mask]
        all_indices = np.arange(len(x))
        x_filled = np.interp(all_indices, valid_indices, valid_values)
    if gaussian_window > 1:
        x_gauss = cv2.GaussianBlur(x_filled.reshape(-1, 1), (gaussian_window, 1), 0).flatten()
    else:
        x_gauss = x_filled
    if moving_avg_window > 1:
        kernel = np.ones(moving_avg_window) / moving_avg_window
        xp = np.pad(x_gauss, (moving_avg_window//2, moving_avg_window//2), mode='edge')
        x_smooth = np.convolve(xp, kernel, mode='valid')
    else:
        x_smooth = x_gauss
    x_final = cv2.GaussianBlur(x_smooth.reshape(-1, 1), (11, 1), 0).flatten()
    x_final[~valid_mask] = np.nan
    return x_final


def three_point_smooth(positions):
    smoothed = positions.copy()
    n = len(positions)
    for i in range(1, n - 1):
        if np.isnan(positions[i-1]) or np.isnan(positions[i]) or np.isnan(positions[i+1]):
            continue
        smoothed[i] = 0.25 * positions[i-1] + 0.5 * positions[i] + 0.25 * positions[i+1]
    return smoothed


def compute_derivative(times, values):
    n = len(values)
    deriv = np.full(n, np.nan)
    for i in range(1, n-1):
        if np.isnan(values[i-1]) or np.isnan(values[i+1]):
            continue
        dt = times[i+1] - times[i-1]
        if dt == 0:
            continue
        deriv[i] = (values[i+1] - values[i-1]) / dt
    if n >= 2:
        if not np.isnan(values[1]) and not np.isnan(values[0]):
            deriv[0] = (values[1] - values[0]) / (times[1] - times[0])
        if not np.isnan(values[-1]) and not np.isnan(values[-2]):
            deriv[-1] = (values[-1] - values[-2]) / (times[-1] - times[-2])
    return deriv


# ---------------------------
# Main processing (modified to add blur + EWMA + CSRT + manual start/stop)
# ---------------------------
def process_video_continuous(video_path, out_csv=None, 
                             interpolate_max_gap=0.25, smoothing_iterations=3,
                             blur_kernel=5, ewma_alpha=0.2):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n{'='*70}")
    print("TRACKER FINAL CONTINUOUS - Single-Point Tracking (CSRT + Gaussian Preprocess + EWMA + Manual Start/Stop)")
    print(f"{'='*70}")
    print(f"[INFO] Video: {video_path}")
    print(f"[INFO] Resolution: {width}x{height}, FPS: {fps}, Frames: {frame_count}")
    print(f"[INFO] Gaussian blur kernel: {blur_kernel}, EWMA alpha: {ewma_alpha}")

    ret, first_frame = cap.read()
    if not ret:
        raise IOError("Could not read first frame")

    # Select region where reference line is
    print("\n[STEP 1] Select the region containing the reference line")
    print("          (draw a box around the area where the vertical line should be detected)")
    print("          Press SPACE or ENTER when done, ESC to cancel")
    line_region = cv2.selectROI("Select Line Region", first_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Line Region")
    if line_region[2] == 0 or line_region[3] == 0:
        raise ValueError("Line region selection cancelled or invalid")
    print(f"[INFO] Line region selected: x={line_region[0]}, y={line_region[1]}, w={line_region[2]}, h={line_region[3]}")

    # Setup output video (with timestamp in tracking directory)
    base = os.path.basename(video_path)
    out_name = f"tracked_{os.path.splitext(base)[0]}_continuous.mp4"
    out_path = get_tracking_output_path(out_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    print(f"[INFO] Writing tracked video to {out_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Tracking structures (single point, ObjectID 0)
    tracker = None
    is_tracking = False
    object_id = 0
    swimmer_data = {object_id: {'x_raw': [], 'y_raw': [], 'x_ewma': [], 'y_ewma': [], 'start_frame': None, 'frames': []}}

    # Line tracking
    line_x_list = []

    frame_idx = 0
    prev_line_x = None

    print("\n[INFO] Processing frames... Press 'a' to start/resume tracking (select ROI), 's' to stop, 'q' to quit")

    # ensure blur kernel odd positive
    blur_kernel = ensure_odd_positive(blur_kernel)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect vertical line on original frame
        x_ref = detect_vertical_line_in_region(frame, line_region, prev_x=prev_line_x,
                                               max_jump=40, weight_decay=0.01, bottom_bias=2.0)
        if x_ref is not None:
            prev_line_x = x_ref
        line_x_list.append(x_ref if x_ref is not None else np.nan)

        # Preprocess (Gaussian blur) for tracker updates
        denoised = cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), 0)

        # Update tracker if active
        if is_tracking and tracker is not None:
            try:
                success, box = tracker.update(denoised)
            except Exception:
                success = False
                box = None

            if success:
                x, y, w, h = [int(v) for v in box]
                cx, cy = x + w // 2, y + h // 2

                # Check if reached right edge (within 5 pixels)
                if cx >= width - 5:
                    print(f"[INFO] Point {object_id} reached right edge at frame {frame_idx}, stopping tracking")
                    is_tracking = False
                    tracker = None
                    continue

                # Initialize swimmer_data if needed
                if swimmer_data[object_id]['start_frame'] is None:
                    swimmer_data[object_id]['start_frame'] = frame_idx

                # Append raw
                swimmer_data[object_id]['x_raw'].append(float(cx))
                swimmer_data[object_id]['y_raw'].append(float(cy))
                swimmer_data[object_id]['frames'].append(frame_idx)

                # Compute EWMA
                if len(swimmer_data[object_id]['x_ewma']) == 0:
                    # initialize EWMA with first observed value
                    swimmer_data[object_id]['x_ewma'].append(float(cx))
                    swimmer_data[object_id]['y_ewma'].append(float(cy))
                else:
                    prev_x_ewma = swimmer_data[object_id]['x_ewma'][-1]
                    prev_y_ewma = swimmer_data[object_id]['y_ewma'][-1]
                    new_x_ewma = ewma_alpha * float(cx) + (1.0 - ewma_alpha) * prev_x_ewma
                    new_y_ewma = ewma_alpha * float(cy) + (1.0 - ewma_alpha) * prev_y_ewma
                    swimmer_data[object_id]['x_ewma'].append(new_x_ewma)
                    swimmer_data[object_id]['y_ewma'].append(new_y_ewma)

                # Draw tracker box and marker using raw box / center (keeps visualization consistent)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
                cv2.putText(frame, f"Point {object_id}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Show relative position if we have line (use EWMA smoothed center for display stability)
                latest_ewma_x = swimmer_data[object_id]['x_ewma'][-1] if len(swimmer_data[object_id]['x_ewma']) > 0 else float(cx)
                rel_pos = latest_ewma_x - x_ref if (x_ref is not None and not np.isnan(x_ref)) else None
                if rel_pos is not None:
                    text_color = (0, 255, 0) if rel_pos >= 0 else (0, 255, 255)
                    cv2.putText(frame, f"Dist: {int(rel_pos):+d}px", (int(latest_ewma_x) + 15, int(cy) - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                    line_color = (0, 255, 0) if rel_pos >= 0 else (0, 255, 255)
                    cv2.line(frame, (int(x_ref), int(cy)), (int(latest_ewma_x), int(cy)), line_color, 2)
            else:
                print(f"[INFO] Tracker {object_id} lost at frame {frame_idx}")
                is_tracking = False
                tracker = None

        # Draw line region
        rx, ry, rw, rh = line_region
        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (255, 0, 255), 2)

        # Draw reference line
        if x_ref is not None and not np.isnan(x_ref):
            cv2.line(frame, (int(x_ref), 0), (int(x_ref), height - 1), (0, 0, 255), 2)
            cv2.putText(frame, f"Ref Line: {int(x_ref)}px", (int(x_ref) + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Frame info
        status_text = "TRACKING" if is_tracking else "STOPPED"
        status_color = (0, 255, 0) if is_tracking else (0, 0, 255)
        cv2.putText(frame, f"Frame: {frame_idx}/{frame_count} | Time: {frame_idx/fps:.2f}s | Status: {status_text} | Points: {1 if is_tracking else 0}",
                   (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(frame, "Press 'a' to start/resume, 's' to stop, 'q' to quit", (10, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out_writer.write(frame)

        # Show preview (half-size)
        display_frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
        cv2.imshow("Tracking Preview", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            # Start/resume tracking: select on display_frame, scale bbox back, initialize with denoised frame
            if not is_tracking:
                bbox = cv2.selectROI("Tracking Preview", display_frame, fromCenter=False)
                cv2.destroyWindow("Tracking Preview")
                if bbox != (0, 0, 0, 0):
                    bbox_full = (int(bbox[0]*2), int(bbox[1]*2), int(bbox[2]*2), int(bbox[3]*2))
                    try:
                        tracker = create_csrt_tracker()
                    except RuntimeError as e:
                        print(f"[ERROR] Could not create CSRT tracker: {e}")
                        continue
                    # initialize tracker on denoised frame for stability
                    try:
                        tracker.init(denoised, bbox_full)
                    except Exception:
                        # fallback to original frame if denoised causes issues
                        tracker.init(frame, bbox_full)
                    is_tracking = True
                    # If this is a new section (not resuming), reset start_frame
                    if swimmer_data[object_id]['start_frame'] is None or len(swimmer_data[object_id]['frames']) == 0:
                        swimmer_data[object_id]['start_frame'] = frame_idx
                    print(f"[INFO] Started/resumed tracking point #{object_id} at frame {frame_idx}")
                # re-open preview window after selection
                cv2.imshow("Tracking Preview", display_frame)
        elif key == ord('s'):
            # Stop tracking (keep data, mark as lost)
            if is_tracking:
                is_tracking = False
                tracker = None
                print(f"[INFO] Stopped tracking point #{object_id} at frame {frame_idx}")
        elif key == ord('q'):
            print("\n[INFO] Stopped by user")
            break

        if frame_idx % 100 == 0:
            print(f"[INFO] Processed {frame_idx}/{frame_count} frames ({100*frame_idx/frame_count:.1f}%)")

        frame_idx += 1

    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()

    print(f"[INFO] Finished processing {frame_idx} frames")

    # --- Convert swimmer_data to use EWMA-smoothed series as the base 'x'/'y' for downstream processing ---
    # This keeps your downstream pipeline (which expects data['x'] and data['y']) intact while preserving raw traces.
    for obj_id in list(swimmer_data.keys()):
        data = swimmer_data[obj_id]
        # If EWMA arrays are shorter than raw (shouldn't be) or empty, fallback to raw
        if len(data.get('x_ewma', [])) == 0:
            # no EWMA values were recorded (very unusual) -> use raw
            data['x'] = data.get('x_raw', [])
            data['y'] = data.get('y_raw', [])
        else:
            data['x'] = data['x_ewma']
            data['y'] = data['y_ewma']

    # --- Post-processing (unchanged apart from using EWMA-based x/y as base) ---
    times = np.arange(frame_idx) / fps
    line_x = np.array(line_x_list, dtype=float)

    print(f"[INFO] Line detected in {(~np.isnan(line_x)).sum()}/{len(line_x)} frames initially")

    line_x_clean = remove_line_outliers(line_x, window=11, threshold=2.5)
    outliers_removed = (~np.isnan(line_x)).sum() - (~np.isnan(line_x_clean)).sum()
    if outliers_removed > 0:
        print(f"[INFO] Removed {outliers_removed} outlier line detections")

    max_gap_frames = int(fps * 1.5)
    line_x_interp = interpolate_line_positions_bidirectional(line_x_clean, max_gap_frames=max_gap_frames)
    gaps_filled = (~np.isnan(line_x_interp)).sum() - (~np.isnan(line_x_clean)).sum()
    if gaps_filled > 0:
        print(f"[INFO] Interpolated {gaps_filled} frames of missing line data")

    print(f"[INFO] Applying aggressive smoothing to line positions...")
    line_x_smooth = aggressive_smooth(line_x_interp, gaussian_window=51, moving_avg_window=25)

    for iteration in range(smoothing_iterations):
        line_x_smooth = three_point_smooth(line_x_smooth)

    print(f"[INFO] Applied 3-point smoothing ({smoothing_iterations} iterations) to line")

    valid_line_mask = ~np.isnan(line_x_smooth)
    ref_line_x = np.nanmedian(line_x_smooth) if valid_line_mask.any() else 0.0
    print(f"[INFO] Reference line median position: {ref_line_x:.1f}px")

    print(f"\n[INFO] Processing {len(swimmer_data)} tracked points...")

    csv_rows = []

    for obj_id in sorted(swimmer_data.keys()):
        data = swimmer_data[obj_id]
        start_frame = data['start_frame'] if data['start_frame'] is not None else 0
        print(f"  Point {obj_id}: {len(data['x'])} frames (started at frame {start_frame})")

        swimmer_x = np.full(frame_idx, np.nan)
        swimmer_y = np.full(frame_idx, np.nan)

        for i in range(len(data['x'])):
            frame_num = data['frames'][i] if i < len(data['frames']) else (start_frame + i)
            if frame_num < frame_idx:
                swimmer_x[frame_num] = data['x'][i]
                swimmer_y[frame_num] = data['y'][i]

        # Compute relative position (signed distance from line)
        rel_x = np.full_like(swimmer_x, np.nan)
        last_known_line = None
        for i in range(len(rel_x)):
            lx = line_x_smooth[i] if i < len(line_x_smooth) and not np.isnan(line_x_smooth[i]) else last_known_line
            if i < len(line_x_smooth) and not np.isnan(line_x_smooth[i]):
                last_known_line = line_x_smooth[i]
            if np.isnan(swimmer_x[i]) or lx is None or np.isnan(lx):
                rel_x[i] = np.nan
            else:
                rel_x[i] = swimmer_x[i] - lx

        swimmer_y_interp = linear_interpolate_positions(times, swimmer_y, max_gap_seconds=interpolate_max_gap)
        rel_x_interp = linear_interpolate_positions(times, rel_x, max_gap_seconds=interpolate_max_gap)

        print(f"[INFO] Applying aggressive smoothing to point {obj_id} positions...")
        swimmer_y_smooth = aggressive_smooth(swimmer_y_interp, gaussian_window=41, moving_avg_window=21)
        rel_x_smooth = aggressive_smooth(rel_x_interp, gaussian_window=41, moving_avg_window=21)

        for iteration in range(smoothing_iterations):
            swimmer_y_smooth = three_point_smooth(swimmer_y_smooth)
            rel_x_smooth = three_point_smooth(rel_x_smooth)

        print(f"[INFO] Applied 3-point smoothing ({smoothing_iterations} iterations) to point {obj_id}")

        for frame_num in range(start_frame, frame_idx):
            rel_x_val = rel_x[frame_num]
            y_val = swimmer_y[frame_num]
            rel_x_smooth_val = rel_x_smooth[frame_num]
            y_smooth_val = swimmer_y_smooth[frame_num]

            has_data = (not np.isnan(rel_x_val) or not np.isnan(y_val) or
                       not np.isnan(rel_x_smooth_val) or not np.isnan(y_smooth_val))

            if has_data:
                csv_rows.append({
                    'Frame': frame_num,
                    'ObjectID': obj_id,
                    'X': rel_x_val if not np.isnan(rel_x_val) else '',
                    'Y': y_val if not np.isnan(y_val) else '',
                    'X_smooth': rel_x_smooth_val if not np.isnan(rel_x_smooth_val) else '',
                    'Y_smooth': y_smooth_val if not np.isnan(y_smooth_val) else ''
                })

    df = pd.DataFrame(csv_rows)
    if len(df) > 0:
        df = df.sort_values(['ObjectID', 'Frame']).reset_index(drop=True)
        df = df[['Frame', 'ObjectID', 'X', 'Y', 'X_smooth', 'Y_smooth']]

    # Build stats dataframe (unchanged)
    all_data = {
        "time_s": times,
        "line_x_raw": line_x_list,
        "line_x_clean": line_x_clean[:len(times)],
        "line_x_interp": line_x_interp[:len(times)],
        "line_x": line_x_smooth[:len(times)]
    }

    for obj_id in sorted(swimmer_data.keys()):
        data = swimmer_data[obj_id]
        start_frame = data['start_frame'] if data['start_frame'] is not None else 0

        swimmer_x = np.full(frame_idx, np.nan)
        swimmer_y = np.full(frame_idx, np.nan)

        for i in range(len(data['x'])):
            frame_num = data['frames'][i] if i < len(data['frames']) else (start_frame + i)
            if frame_num < frame_idx:
                swimmer_x[frame_num] = data['x'][i]
                swimmer_y[frame_num] = data['y'][i]

        rel_x = np.full_like(swimmer_x, np.nan)
        last_known_line = None
        for i in range(len(rel_x)):
            lx = line_x_smooth[i] if i < len(line_x_smooth) and not np.isnan(line_x_smooth[i]) else last_known_line
            if i < len(line_x_smooth) and not np.isnan(line_x_smooth[i]):
                last_known_line = line_x_smooth[i]
            if np.isnan(swimmer_x[i]) or lx is None or np.isnan(lx):
                rel_x[i] = np.nan
            else:
                rel_x[i] = swimmer_x[i] - lx

        rel_x_interp = linear_interpolate_positions(times, rel_x, max_gap_seconds=interpolate_max_gap)
        swimmer_y_interp = linear_interpolate_positions(times, swimmer_y, max_gap_seconds=interpolate_max_gap)

        rel_x_sm = aggressive_smooth(rel_x_interp, gaussian_window=41, moving_avg_window=21)
        swimmer_y_smooth = aggressive_smooth(swimmer_y_interp, gaussian_window=41, moving_avg_window=21)
        swimmer_x_smooth = aggressive_smooth(linear_interpolate_positions(times, swimmer_x, max_gap_seconds=interpolate_max_gap),
                                            gaussian_window=41, moving_avg_window=21)

        for iteration in range(smoothing_iterations):
            rel_x_sm = three_point_smooth(rel_x_sm)
            swimmer_y_smooth = three_point_smooth(swimmer_y_smooth)
            swimmer_x_smooth = three_point_smooth(swimmer_x_smooth)

        vx = compute_derivative(times, rel_x_sm)
        ax = compute_derivative(times, vx)

        all_data[f"point_{obj_id}_x"] = swimmer_x
        all_data[f"point_{obj_id}_y"] = swimmer_y
        all_data[f"point_{obj_id}_x_smooth"] = swimmer_x_smooth
        all_data[f"point_{obj_id}_y_smooth"] = swimmer_y_smooth
        all_data[f"point_{obj_id}_start_frame"] = start_frame
        all_data[f"point_{obj_id}_rel_x_px"] = rel_x
        all_data[f"point_{obj_id}_rel_x_interp_px"] = rel_x_interp
        all_data[f"point_{obj_id}_rel_x_smooth_px"] = rel_x_sm
        all_data[f"point_{obj_id}_v_x_px_s"] = vx
        all_data[f"point_{obj_id}_a_x_px_s2"] = ax

    df_stats = pd.DataFrame(all_data)

    if out_csv:
        df.to_csv(out_csv, index=False)
        print(f"[INFO] Results written to {out_csv}")

    for obj_id in sorted(swimmer_data.keys()):
        point_x = df_stats[f"point_{obj_id}_x"].values
        valid_point = point_x[~np.isnan(point_x)]

        rel_x_sm = df_stats[f"point_{obj_id}_rel_x_smooth_px"].values
        vx = df_stats[f"point_{obj_id}_v_x_px_s"].values

        valid_rel_x = rel_x_sm[~np.isnan(rel_x_sm)]
        valid_vx = vx[~np.isnan(vx)]

        start_frame = swimmer_data[obj_id]['start_frame'] if swimmer_data[obj_id]['start_frame'] is not None else 0
        start_time = start_frame / fps

        print(f"\n[STATS] Point {obj_id}:")
        print(f"  Started: frame {start_frame} (t={start_time:.2f}s)")
        print(f"  Detected: {len(valid_point)}/{len(point_x)} frames ({100*len(valid_point)/len(point_x):.1f}%)")

        if len(valid_rel_x) > 0:
            print(f"  Position Statistics (distance from line):")
            print(f"    Mean: {np.mean(valid_rel_x):+.2f} px (+ = right of line, - = left of line)")
            print(f"    Std: {np.std(valid_rel_x):.2f} px")
            print(f"    Min: {np.min(valid_rel_x):+.2f} px")
            print(f"    Max: {np.max(valid_rel_x):+.2f} px")
            crossed_line = (valid_rel_x < 0).sum()
            if crossed_line > 0:
                print(f"    Frames left of line: {crossed_line} ({100*crossed_line/len(valid_rel_x):.1f}%)")

        if len(valid_vx) > 0:
            print(f"  Velocity Statistics:")
            print(f"    Mean: {np.mean(valid_vx):.2f} px/s")
            print(f"    Std: {np.std(valid_vx):.2f} px/s")
            print(f"    Min: {np.min(valid_vx):.2f} px/s")
            print(f"    Max: {np.max(valid_vx):.2f} px/s")

    print(f"\n{'='*70}")
    print("✅ PROCESSING COMPLETE!")
    print(f"{'='*70}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Single-point tracking with manual start/stop capability for continuous tracking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--video', required=True, help='Path to input video file')
    parser.add_argument('--out', default=None,
                       help='CSV output path (default: timestamped file in tracking/ directory)')
    parser.add_argument('--smooth', type=int, default=3,
                       help='Number of 3-point smoothing iterations (default: 3)')
    parser.add_argument('--interp-gap', type=float, default=0.25,
                       help='Max gap to interpolate in seconds (default: 0.25)')
    parser.add_argument('--ewma-alpha', type=float, default=0.2,
                       help='EWMA smoothing alpha (0 < alpha < 1). Default 0.2')
    parser.add_argument('--blur-k', type=int, default=5,
                       help='Gaussian blur kernel size (odd integer). Default 5')

    args = parser.parse_args()

    # validate ewma alpha
    if not (0.0 < args.ewma_alpha <= 1.0):
        print("[WARN] ewma-alpha out of range, resetting to 0.2")
        args.ewma_alpha = 0.2

    # Set default output path with timestamp if not provided
    if args.out is None:
        args.out = get_tracking_output_path('tracker_final_continuous_results.csv')

    df = process_video_continuous(
        args.video,
        out_csv=args.out,
        interpolate_max_gap=args.interp_gap,
        smoothing_iterations=args.smooth,
        blur_kernel=args.blur_k,
        ewma_alpha=args.ewma_alpha
    )

    print("\n[PREVIEW] First few rows of data:")
    if df is not None and not df.empty:
        print(df.head(10))
    else:
        print("[INFO] No tracking data produced.")


if __name__ == '__main__':
    main()

