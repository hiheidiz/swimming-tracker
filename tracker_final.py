#!/usr/bin/env python3
"""
tracker_final.py

Combines smooth_combined_tracker.py with multi-point selection capability.
Based on smooth_combined_tracker.py with added ability to track multiple body parts.

Features:
- Select multiple tracking points (body parts) during video playback (press 'a')
- Each point represents a different part of the swimmer's body
- Enhanced line tracking with blue channel optimization
- Compute distance from each point to reference line
- Custom 3-point smoothing: 1/4 previous + 1/2 current + 1/4 next position
- Comprehensive CSV output with all data

Usage:
    python tracker_final.py --video path/to/video.mp4
"""

import cv2
import numpy as np
import argparse
import pandas as pd
import csv
from scipy.ndimage import median_filter


def detect_vertical_line_in_region(frame, region_bbox, prev_x=None, max_jump=40, 
                                   weight_decay=0.01, bottom_bias=2.0, min_line_length=40):
    """
    Enhanced vertical line detection within a specified region using Hough Lines.
    Uses blue channel information for better contrast (Line B≈23, Water B≈65).
    
    Args:
        frame: BGR frame
        region_bbox: (x, y, w, h) defining the region to search for the line
        prev_x: previous x position (global coordinates) for temporal consistency
        max_jump: max allowed px jump per frame
        weight_decay: how strongly to penalize lines far from previous x
        bottom_bias: how strongly to favor lower portions of the image
        min_line_length: minimum line length to consider
    
    Returns:
        x_line (float) in global coordinates or None if not found
    """
    rx, ry, rw, rh = region_bbox
    
    # Extract region of interest
    roi = frame[ry:ry+rh, rx:rx+rw]
    
    # Enhanced contrast using blue channel information
    # Line has B≈23, water has B≈65, so emphasize low blue values
    b, g, r = cv2.split(roi)
    
    # Create a mask that emphasizes low blue values (where the line is)
    # Invert blue channel so line (B=23) becomes brighter
    b_inv = 255 - b
    
    # Combine channels with emphasis on blue contrast
    # Use inverted blue channel and regular grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Weighted combination: 60% inverted blue, 40% regular gray
    enhanced = cv2.addWeighted(b_inv, 0.6, gray, 0.4, 0)
    
    # Further enhance contrast with CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(enhanced)
    
    # Additional contrast boost
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.8, beta=-20)
    
    # Apply bilateral filter to reduce noise while preserving edges
    enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
    
    # Edge detection with adjusted thresholds for enhanced image
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
            
            # More strict vertical requirement (87-93 degrees or -93 to -87)
            if abs(angle) > 87:  # very vertical
                x_mid_local = (x1 + x2) / 2
                y_mid_local = (y1 + y2) / 2
                line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                # Convert to global coordinates
                x_mid_global = x_mid_local + rx
                
                # Safety check: ensure line is within region bounds (with small margin)
                margin = 5
                if not (rx - margin <= x_mid_global <= rx + rw + margin):
                    continue
                
                # --- proximity weight (temporal) ---
                if prev_x is not None:
                    d = abs(x_mid_global - prev_x)
                    proximity_weight = np.exp(-weight_decay * d**2)
                else:
                    proximity_weight = 1.0

                # --- bottom bias weight (spatial) ---
                normalized_y = y_mid_local / rh  # 0 at top, 1 at bottom of ROI
                bottom_weight = 1 + bottom_bias * (normalized_y - 0.66)  # boost bottom third
                bottom_weight = max(bottom_weight, 0.1)  # avoid negative weights
                
                # --- length weight (prefer longer lines) ---
                # Normalize by ROI height
                length_weight = min(line_length / (rh * 0.5), 2.0)  # cap at 2x
                
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
                
                # Additional safety: check if weighted position is reasonable
                # Should be within the region bounds
                if weighted_x < rx or weighted_x > rx + rw:
                    # Fall back to previous if available
                    if prev_x is not None and rx <= prev_x <= rx + rw:
                        weighted_x = prev_x
                    else:
                        # Use the longest line within bounds
                        valid_mask = (candidates >= rx) & (candidates <= rx + rw)
                        if valid_mask.any():
                            best_idx = np.argmax(line_lengths * valid_mask)
                            weighted_x = candidates[best_idx]
                        else:
                            return None
                
                # More aggressive jump rejection
                if prev_x is not None:
                    jump_size = abs(weighted_x - prev_x)
                    if jump_size > max_jump:
                        # If jump is too large, don't update
                        return prev_x
                    elif jump_size > max_jump * 0.5:
                        # If jump is moderate, dampen it
                        alpha = 0.3  # Use 30% new value, 70% old value
                        weighted_x = alpha * weighted_x + (1 - alpha) * prev_x
                
                x_line = weighted_x

    # if no line found, return None (will be handled by interpolation later)
    if x_line is None:
        return None

    return x_line


def linear_interpolate_positions(times, xs, max_gap_seconds=0.2):
    """
    Fill short NaN gaps in xs by linear interpolation if gap <= max_gap_seconds.
    """
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
    """
    Interpolate missing line positions using both past and future values.
    For gaps, linearly interpolate between last known and next known position.
    
    Args:
        line_positions: array of line x positions with NaN for missing
        max_gap_frames: maximum gap size to interpolate
    
    Returns:
        interpolated array
    """
    positions = line_positions.copy()
    n = len(positions)
    
    # Find all valid (non-NaN) positions
    valid_mask = ~np.isnan(positions)
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        return positions
    
    # For each gap, interpolate if within max_gap_frames
    i = 0
    while i < n:
        if np.isnan(positions[i]):
            # Find start of gap
            gap_start = i
            
            # Find end of gap
            while i < n and np.isnan(positions[i]):
                i += 1
            gap_end = i
            
            # Check if we can interpolate
            has_before = gap_start > 0 and not np.isnan(positions[gap_start - 1])
            has_after = gap_end < n and not np.isnan(positions[gap_end])
            gap_size = gap_end - gap_start
            
            if has_before and has_after and gap_size <= max_gap_frames:
                # Interpolate between before and after
                before_val = positions[gap_start - 1]
                after_val = positions[gap_end]
                positions[gap_start:gap_end] = np.linspace(before_val, after_val, gap_size + 2)[1:-1]
            elif has_before and gap_size <= max_gap_frames // 2:
                # Only have before value, hold it for small gaps
                positions[gap_start:gap_end] = positions[gap_start - 1]
            elif has_after and gap_size <= max_gap_frames // 2:
                # Only have after value, use it for small gaps
                positions[gap_start:gap_end] = positions[gap_end]
        else:
            i += 1
    
    return positions


def remove_line_outliers(line_positions, window=11, threshold=2.5):
    """
    Remove outlier line positions using a median filter approach.
    Replaces outliers with NaN so they can be interpolated.
    
    Args:
        line_positions: array of line x positions
        window: window size for median filter (should be odd)
        threshold: number of standard deviations to consider as outlier
    
    Returns:
        cleaned array with outliers replaced by NaN
    """
    positions = line_positions.copy()
    valid_mask = ~np.isnan(positions)
    
    if valid_mask.sum() < window:
        return positions
    
    # Create a temporary array for median filtering
    temp = positions.copy()
    temp[~valid_mask] = np.nanmedian(positions)  # Fill NaNs with median for filtering
    
    # Apply median filter
    filtered = median_filter(temp, size=window, mode='nearest')
    
    # Calculate deviation from median
    deviation = np.abs(positions - filtered)
    
    # Only check valid positions
    valid_deviations = deviation[valid_mask]
    if len(valid_deviations) == 0:
        return positions
    
    # Calculate threshold based on median absolute deviation
    mad = np.median(valid_deviations)
    outlier_threshold = threshold * mad * 1.4826  # 1.4826 is scale factor for normal distribution
    
    # Mark outliers as NaN
    is_outlier = deviation > outlier_threshold
    positions[is_outlier & valid_mask] = np.nan
    
    return positions


def aggressive_smooth(x, gaussian_window=31, moving_avg_window=15):
    """
    Apply aggressive multi-stage smoothing for very smooth curves.
    
    Args:
        x: input signal (may contain NaNs)
        gaussian_window: window size for Gaussian blur (must be odd)
        moving_avg_window: window size for moving average
    
    Returns:
        smoothed signal
    """
    # Stage 1: Fill NaNs with interpolation for smoothing
    valid_mask = ~np.isnan(x)
    if not valid_mask.any():
        return x
    
    x_filled = x.copy()
    if np.isnan(x_filled).any():
        valid_indices = np.where(valid_mask)[0]
        valid_values = x[valid_mask]
        all_indices = np.arange(len(x))
        x_filled = np.interp(all_indices, valid_indices, valid_values)
    
    # Stage 2: Gaussian blur smoothing
    if gaussian_window > 1:
        x_gauss = cv2.GaussianBlur(x_filled.reshape(-1, 1), (gaussian_window, 1), 0).flatten()
    else:
        x_gauss = x_filled
    
    # Stage 3: Moving average smoothing
    if moving_avg_window > 1:
        kernel = np.ones(moving_avg_window) / moving_avg_window
        xp = np.pad(x_gauss, (moving_avg_window//2, moving_avg_window//2), mode='edge')
        x_smooth = np.convolve(xp, kernel, mode='valid')
    else:
        x_smooth = x_gauss
    
    # Stage 4: Additional light Gaussian for extra smoothness
    x_final = cv2.GaussianBlur(x_smooth.reshape(-1, 1), (11, 1), 0).flatten()
    
    # Restore NaNs where original data was NaN
    x_final[~valid_mask] = np.nan
    
    return x_final


def three_point_smooth(positions):
    """
    Apply 3-point weighted smoothing: 1/4 previous + 1/2 current + 1/4 next
    
    Args:
        positions: numpy array of positions (can contain NaN)
    
    Returns:
        smoothed positions
    """
    smoothed = positions.copy()
    n = len(positions)
    
    for i in range(1, n - 1):
        # Skip if any of the three points are NaN
        if np.isnan(positions[i-1]) or np.isnan(positions[i]) or np.isnan(positions[i+1]):
            continue
        
        # Apply the smoothing formula
        smoothed[i] = 0.25 * positions[i-1] + 0.5 * positions[i] + 0.25 * positions[i+1]
    
    return smoothed


def compute_derivative(times, values):
    """Compute numerical derivative using central differences."""
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


def process_video_multitrack(video_path, out_csv=None, 
                             interpolate_max_gap=0.25, smoothing_iterations=3):
    """
    Process video with multi-swimmer tracking and line tracking.
    Based on smooth_combined_tracker.py with multi-tracker capability.
    """
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
    print("TRACKER FINAL - Multi-Point Body Part Tracking")
    print(f"{'='*70}")
    print(f"[INFO] Video: {video_path}")
    print(f"[INFO] Resolution: {width}x{height}, FPS: {fps}, Frames: {frame_count}")
    
    # Read first frame for selection
    ret, first_frame = cap.read()
    if not ret:
        raise IOError("Could not read first frame")
    
    # Step 1: Select region where the reference line is
    print("\n[STEP 1] Select the region containing the reference line")
    print("          (draw a box around the area where the vertical line should be detected)")
    print("          Press SPACE or ENTER when done, ESC to cancel")
    
    line_region = cv2.selectROI("Select Line Region", first_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Line Region")
    
    if line_region[2] == 0 or line_region[3] == 0:
        raise ValueError("Line region selection cancelled or invalid")
    
    print(f"[INFO] Line region selected: x={line_region[0]}, y={line_region[1]}, w={line_region[2]}, h={line_region[3]}")
    
    # Setup output video
    import os
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    base = os.path.basename(video_path)
    out_name = f"tracked_{os.path.splitext(base)[0]}.mp4"
    out_path = os.path.join(os.path.dirname(video_path) if os.path.dirname(video_path) else '.', out_name)
    out_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    print(f"[INFO] Writing tracked video to {out_path}")
    
    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Initialize tracking structures (multi-tracker style)
    trackers = []
    object_ids = []
    swimmer_data = {}  # {obj_id: {'x': [], 'y': [], 'start_frame': int}}
    
    # Line tracking
    line_x_list = []
    
    frame_idx = 0
    prev_line_x = None
    
    print("\n[INFO] Processing frames... Press 'a' to add tracking point (body part), 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect vertical line in the specified region
        x_ref = detect_vertical_line_in_region(frame, line_region, prev_x=prev_line_x,
                                               max_jump=40, weight_decay=0.01, bottom_bias=2.0)
        if x_ref is not None:
            prev_line_x = x_ref
        line_x_list.append(x_ref if x_ref is not None else np.nan)
        
        # Update all swimmer trackers
        to_remove = []
        for i, tracker in enumerate(trackers):
            success, box = tracker.update(frame)
            
            if success:
                x, y, w, h = [int(v) for v in box]
                cx, cy = x + w // 2, y + h // 2
                obj_id = object_ids[i]
                
                # Check if reached right edge (within 5 pixels)
                if cx >= width - 5:
                    print(f"[INFO] Point {obj_id} reached right edge at frame {frame_idx}, removing")
                    to_remove.append(i)
                    continue
                
                # Store position
                if obj_id not in swimmer_data:
                    swimmer_data[obj_id] = {'x': [], 'y': [], 'start_frame': frame_idx}
                
                swimmer_data[obj_id]['x'].append(float(cx))
                swimmer_data[obj_id]['y'].append(float(cy))
                
                # Draw tracker box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
                cv2.putText(frame, f"Point {obj_id}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Show relative position if we have line
                if x_ref is not None and not np.isnan(x_ref):
                    rel_pos = cx - x_ref
                    # Color code based on which side of line: green for right (+), yellow for left (-)
                    text_color = (0, 255, 0) if rel_pos >= 0 else (0, 255, 255)  # Green if right, yellow if left
                    cv2.putText(frame, f"Dist: {int(rel_pos):+d}px", (cx + 15, cy - 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                    # Draw line from reference to swimmer (color coded)
                    line_color = (0, 255, 0) if rel_pos >= 0 else (0, 255, 255)
                    cv2.line(frame, (int(x_ref), cy), (cx, cy), line_color, 2)
            else:
                to_remove.append(i)
        
        # Remove failed trackers
        for idx in sorted(to_remove, reverse=True):
            removed_id = object_ids[idx]
            print(f"[INFO] Tracker {removed_id} lost at frame {frame_idx}")
            del trackers[idx]
            del object_ids[idx]
        
        # Draw line region box (semi-transparent)
        rx, ry, rw, rh = line_region
        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (255, 0, 255), 2)
        
        # Draw detected reference line
        if x_ref is not None and not np.isnan(x_ref):
            cv2.line(frame, (int(x_ref), 0), (int(x_ref), height - 1), (0, 0, 255), 2)
            cv2.putText(frame, f"Ref Line: {int(x_ref)}px", (int(x_ref) + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw frame info
        cv2.putText(frame, f"Frame: {frame_idx}/{frame_count} | Time: {frame_idx/fps:.2f}s | Points: {len(trackers)}", 
                   (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'a' to add point, 'q' to quit", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        out_writer.write(frame)
        
        # Show preview
        display_frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
        cv2.imshow("Tracking Preview", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            # Add new tracker
            bbox = cv2.selectROI("Tracking Preview", display_frame, fromCenter=False)
            if bbox != (0, 0, 0, 0):
                # Scale bbox back to full resolution
                bbox_full = (bbox[0]*2, bbox[1]*2, bbox[2]*2, bbox[3]*2)
                tracker = cv2.legacy.TrackerCSRT_create()
                tracker.init(frame, bbox_full)
                trackers.append(tracker)
                new_id = max(object_ids, default=-1) + 1
                object_ids.append(new_id)
                print(f"[INFO] Added point #{new_id} at frame {frame_idx}")
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
    
    # --- Post-processing ---
    times = np.arange(frame_idx) / fps
    line_x = np.array(line_x_list, dtype=float)
    
    print(f"[INFO] Line detected in {(~np.isnan(line_x)).sum()}/{len(line_x)} frames initially")
    
    # Step 1: Remove outliers from line positions
    line_x_clean = remove_line_outliers(line_x, window=11, threshold=2.5)
    outliers_removed = (~np.isnan(line_x)).sum() - (~np.isnan(line_x_clean)).sum()
    if outliers_removed > 0:
        print(f"[INFO] Removed {outliers_removed} outlier line detections")
    
    # Step 2: Interpolate gaps bidirectionally (averaging between past and future)
    max_gap_frames = int(fps * 1.5)  # Allow up to 1.5 seconds of gap
    line_x_interp = interpolate_line_positions_bidirectional(line_x_clean, max_gap_frames=max_gap_frames)
    gaps_filled = (~np.isnan(line_x_interp)).sum() - (~np.isnan(line_x_clean)).sum()
    if gaps_filled > 0:
        print(f"[INFO] Interpolated {gaps_filled} frames of missing line data")
    
    # Step 3: Apply AGGRESSIVE temporal smoothing to interpolated line positions
    print(f"[INFO] Applying aggressive smoothing to line positions...")
    line_x_smooth = aggressive_smooth(line_x_interp, gaussian_window=51, moving_avg_window=25)
    
    # Step 4: Apply 3-point smoothing
    for iteration in range(smoothing_iterations):
        line_x_smooth = three_point_smooth(line_x_smooth)
    
    print(f"[INFO] Applied 3-point smoothing ({smoothing_iterations} iterations) to line")
    
    valid_line_mask = ~np.isnan(line_x_smooth)
    ref_line_x = np.nanmedian(line_x_smooth) if valid_line_mask.any() else 0.0
    print(f"[INFO] Reference line median position: {ref_line_x:.1f}px")
    
    # Process each tracked point
    print(f"\n[INFO] Processing {len(swimmer_data)} tracked points...")
    
    # Build dataframe in long format: Frame, ObjectID, X, Y, X_smooth, Y_smooth
    csv_rows = []
    
    for obj_id in sorted(swimmer_data.keys()):
        data = swimmer_data[obj_id]
        start_frame = data['start_frame']
        print(f"  Point {obj_id}: {len(data['x'])} frames (started at frame {start_frame})")
        
        # Pad arrays to match frame count, starting from start_frame
        swimmer_x = np.full(frame_idx, np.nan)
        swimmer_y = np.full(frame_idx, np.nan)
        
        for i in range(len(data['x'])):
            frame_num = start_frame + i
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
                # Signed distance: positive when right of line, negative when left
                rel_x[i] = swimmer_x[i] - lx
        
        # Interpolate short gaps
        swimmer_y_interp = linear_interpolate_positions(times, swimmer_y, max_gap_seconds=interpolate_max_gap)
        rel_x_interp = linear_interpolate_positions(times, rel_x, max_gap_seconds=interpolate_max_gap)
        
        # Apply AGGRESSIVE smoothing to the interpolated point position
        print(f"[INFO] Applying aggressive smoothing to point {obj_id} positions...")
        swimmer_y_smooth = aggressive_smooth(swimmer_y_interp, gaussian_window=41, moving_avg_window=21)
        rel_x_smooth = aggressive_smooth(rel_x_interp, gaussian_window=41, moving_avg_window=21)
        
        # Apply 3-point smoothing
        for iteration in range(smoothing_iterations):
            swimmer_y_smooth = three_point_smooth(swimmer_y_smooth)
            rel_x_smooth = three_point_smooth(rel_x_smooth)
        
        print(f"[INFO] Applied 3-point smoothing ({smoothing_iterations} iterations) to point {obj_id}")
        
        # Build rows for this object (include all frames from start_frame onwards)
        for frame_num in range(start_frame, frame_idx):
            rel_x_val = rel_x[frame_num]  # Distance from line (raw)
            y_val = swimmer_y[frame_num]
            rel_x_smooth_val = rel_x_smooth[frame_num]  # Distance from line (smoothed)
            y_smooth_val = swimmer_y_smooth[frame_num]
            
            # Add row with data (empty string for NaN values)
            csv_rows.append({
                'Frame': frame_num,
                'ObjectID': obj_id,
                'X': rel_x_val if not np.isnan(rel_x_val) else '',
                'Y': y_val if not np.isnan(y_val) else '',
                'X_smooth': rel_x_smooth_val if not np.isnan(rel_x_smooth_val) else '',
                'Y_smooth': y_smooth_val if not np.isnan(y_smooth_val) else ''
            })
    
    # Create DataFrame from rows and sort by ObjectID, Frame
    df = pd.DataFrame(csv_rows)
    if len(df) > 0:
        df = df.sort_values(['ObjectID', 'Frame']).reset_index(drop=True)
        # Ensure column order: Frame, ObjectID, X, Y, X_smooth, Y_smooth
        df = df[['Frame', 'ObjectID', 'X', 'Y', 'X_smooth', 'Y_smooth']]
    
    # For statistics, keep internal wide format for calculations
    all_data = {
        "time_s": times,
        "line_x_raw": line_x_list,
        "line_x_clean": line_x_clean[:len(times)],
        "line_x_interp": line_x_interp[:len(times)],
        "line_x": line_x_smooth[:len(times)]
    }
    
    for obj_id in sorted(swimmer_data.keys()):
        data = swimmer_data[obj_id]
        start_frame = data['start_frame']
        
        # Pad arrays to match frame count, starting from start_frame
        swimmer_x = np.full(frame_idx, np.nan)
        swimmer_y = np.full(frame_idx, np.nan)
        
        for i in range(len(data['x'])):
            frame_num = start_frame + i
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
                # Signed distance: positive when right of line, negative when left
                rel_x[i] = swimmer_x[i] - lx
        
        # Interpolate short gaps
        rel_x_interp = linear_interpolate_positions(times, rel_x, max_gap_seconds=interpolate_max_gap)
        swimmer_y_interp = linear_interpolate_positions(times, swimmer_y, max_gap_seconds=interpolate_max_gap)
        
        # Apply AGGRESSIVE smoothing to the interpolated point position
        rel_x_sm = aggressive_smooth(rel_x_interp, gaussian_window=41, moving_avg_window=21)
        swimmer_y_smooth = aggressive_smooth(swimmer_y_interp, gaussian_window=41, moving_avg_window=21)
        swimmer_x_smooth = aggressive_smooth(linear_interpolate_positions(times, swimmer_x, max_gap_seconds=interpolate_max_gap), 
                                            gaussian_window=41, moving_avg_window=21)
        
        # Apply 3-point smoothing
        for iteration in range(smoothing_iterations):
            rel_x_sm = three_point_smooth(rel_x_sm)
            swimmer_y_smooth = three_point_smooth(swimmer_y_smooth)
            swimmer_x_smooth = three_point_smooth(swimmer_x_smooth)
        
        # Compute velocity and acceleration for statistics
        vx = compute_derivative(times, rel_x_sm)
        ax = compute_derivative(times, vx)
        
        # Add to dataframe for statistics
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
    
    # Print statistics for each tracked point (using df_stats for calculations)
    for obj_id in sorted(swimmer_data.keys()):
        point_x = df_stats[f"point_{obj_id}_x"].values
        valid_point = point_x[~np.isnan(point_x)]
        
        rel_x_sm = df_stats[f"point_{obj_id}_rel_x_smooth_px"].values
        vx = df_stats[f"point_{obj_id}_v_x_px_s"].values
        
        valid_rel_x = rel_x_sm[~np.isnan(rel_x_sm)]
        valid_vx = vx[~np.isnan(vx)]
        
        start_frame = swimmer_data[obj_id]['start_frame']
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
        description="Multi-point tracking for tracking different body parts of a swimmer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tracker_final.py --video test.mp4
  python tracker_final.py --video test.mp4 --out results.csv --smooth 5
  
Instructions:
  1. Select the region where the reference line should be detected
  2. Press 'a' during playback to add tracking points (different body parts)
  3. Press 'q' to finish and process results
  
Note: Each "point" represents a different part of the swimmer's body (hand, shoulder, head, etc.)
      All tracking data is saved to CSV for analysis.
        """
    )
    parser.add_argument('--video', required=True, help='Path to input video file')
    parser.add_argument('--out', default='tracker_final_results.csv', 
                       help='CSV output path (default: tracker_final_results.csv)')
    parser.add_argument('--smooth', type=int, default=3, 
                       help='Number of 3-point smoothing iterations (default: 3)')
    parser.add_argument('--interp-gap', type=float, default=0.25, 
                       help='Max gap to interpolate in seconds (default: 0.25)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("TRACKER FINAL - MULTI-POINT BODY PART TRACKING")
    print("=" * 70)
    
    df = process_video_multitrack(
        args.video, 
        out_csv=args.out, 
        interpolate_max_gap=args.interp_gap,
        smoothing_iterations=args.smooth
    )
    
    print("\n[PREVIEW] First few rows of data:")
    print(df.head(10))


if __name__ == '__main__':
    main()
