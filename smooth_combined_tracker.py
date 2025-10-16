#!/usr/bin/env python3
"""
swimming_select.py

Interactive swimmer tracking with manual selection of line region and tracking point.

Usage:
    python swimming_select.py --video path/to/video.mp4

What it does:
- Lets you select a region where the reference line should be detected (to avoid ambiguity with multiple lines)
- Lets you select a point/region on the swimmer to track
- Uses enhanced Hough Line detection within the selected region
- Tracks swimmer using CSRT tracker
- Computes swimmer position relative to the reference line
- Saves tracking video, CSV data, and analysis plots

Dependencies:
    pip install opencv-python numpy matplotlib pandas scipy
"""

import cv2
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
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


def smooth_signal(x, window=5):
    """Simple moving average smoothing."""
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    xp = np.pad(x, (window//2, window//2), mode='edge')
    sm = np.convolve(xp, kernel, mode='valid')
    return sm


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


def process_video_with_selection(video_path, out_csv=None, plot=True, 
                                 interpolate_max_gap=0.25, smoothing_window=25):
    """
    Process video with manual selection of line region and tracking point.
    
    Line Tracking Safeguards:
    1. Region-based detection: Only searches for line within user-selected region
    2. Strict verticality: Requires lines to be 87-93 degrees (very vertical)
    3. Length weighting: Prefers longer, more prominent lines
    4. Temporal consistency: Heavily weights lines near previous position
    5. Jump rejection: Rejects or dampens large position jumps between frames
    6. Bounds checking: Ensures detected line stays within selected region
    7. Outlier removal: Uses median filtering to remove spurious detections
    8. Bidirectional interpolation: Fills gaps by averaging between past and future
    9. Gaussian smoothing: Final temporal smoothing for stable output
    
    This multi-stage approach prevents the line from jumping between different
    vertical lines in the video and handles temporary occlusions gracefully.
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

    print(f"\n[INFO] Video: {video_path}")
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
    
    # Step 2: Select point/region to track on swimmer
    print("\n[STEP 2] Select the swimmer (or a point on the swimmer) to track")
    print("          Press SPACE or ENTER when done, ESC to cancel")
    
    swimmer_bbox = cv2.selectROI("Select Swimmer to Track", first_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Swimmer to Track")
    
    if swimmer_bbox[2] == 0 or swimmer_bbox[3] == 0:
        raise ValueError("Swimmer selection cancelled or invalid")
    
    print(f"[INFO] Swimmer selected: x={swimmer_bbox[0]}, y={swimmer_bbox[1]}, w={swimmer_bbox[2]}, h={swimmer_bbox[3]}")
    
    # Initialize CSRT tracker
    tracker = cv2.legacy.TrackerCSRT_create()
    tracker.init(first_frame, swimmer_bbox)
    
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
    
    # Storage for tracking data
    times = []
    swimmer_x = []
    swimmer_y = []
    line_x_list = []
    
    frame_idx = 0
    prev_line_x = None
    
    print("\n[INFO] Processing frames... (press 'q' to quit early)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        t = frame_idx / fps
        times.append(t)
        
        # Detect vertical line in the specified region
        x_ref = detect_vertical_line_in_region(frame, line_region, prev_x=prev_line_x,
                                               max_jump=40, weight_decay=0.01, bottom_bias=2.0)
        if x_ref is not None:
            prev_line_x = x_ref
        line_x_list.append(x_ref if x_ref is not None else np.nan)
        
        # Update swimmer tracker
        success, box = tracker.update(frame)
        
        if success:
            x, y, w, h = [int(v) for v in box]
            cx, cy = x + w // 2, y + h // 2
            swimmer_x.append(float(cx))
            swimmer_y.append(float(cy))
        else:
            swimmer_x.append(np.nan)
            swimmer_y.append(np.nan)
        
        # Draw overlay
        overlay = frame.copy()
        
        # Draw line region box (semi-transparent)
        rx, ry, rw, rh = line_region
        cv2.rectangle(overlay, (rx, ry), (rx + rw, ry + rh), (255, 0, 255), 2)
        cv2.putText(overlay, "Line Region", (rx, ry - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Draw detected reference line
        if x_ref is not None and not np.isnan(x_ref):
            cv2.line(overlay, (int(x_ref), 0), (int(x_ref), height - 1), (0, 0, 255), 2)
            cv2.putText(overlay, f"Ref Line: {int(x_ref)}px", (int(x_ref) + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw swimmer tracking
        if success:
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(overlay, (cx, cy), 6, (0, 255, 0), -1)
            
            # Show relative position if we have line
            if x_ref is not None and not np.isnan(x_ref):
                rel_pos = cx - x_ref
                # Color code based on which side of line: green for right (+), yellow for left (-)
                text_color = (0, 255, 0) if rel_pos >= 0 else (0, 255, 255)  # Green if right, yellow if left
                cv2.putText(overlay, f"Dist: {int(rel_pos):+d}px", (cx + 15, cy - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                # Draw line from reference to swimmer (color coded)
                line_color = (0, 255, 0) if rel_pos >= 0 else (0, 255, 255)
                cv2.line(overlay, (int(x_ref), cy), (cx, cy), line_color, 2)
        else:
            cv2.putText(overlay, "TRACKING LOST", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        # Draw frame info
        cv2.putText(overlay, f"Frame: {frame_idx}/{frame_count} | Time: {t:.2f}s", 
                   (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        out_writer.write(overlay)
        
        # Show preview
        display_frame = cv2.resize(overlay, (overlay.shape[1]//2, overlay.shape[0]//2))
        cv2.imshow("Tracking Preview", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
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
    times = np.array(times)
    swimmer_x = np.array(swimmer_x, dtype=float)
    swimmer_y = np.array(swimmer_y, dtype=float)
    line_x = np.array(line_x_list, dtype=float)
    
    print(f"[INFO] Line detected in {(~np.isnan(line_x)).sum()}/{len(line_x)} frames initially")
    
    # Step 1: Remove outliers from line positions
    line_x_clean = remove_line_outliers(line_x, window=11, threshold=2.5)
    outliers_removed = (~np.isnan(line_x)).sum() - (~np.isnan(line_x_clean)).sum()
    if outliers_removed > 0:
        print(f"[INFO] Removed {outliers_removed} outlier line detections")
    
    # Step 2: Interpolate gaps bidirectionally (averaging between past and future)
    # This handles cases where line disappears temporarily
    max_gap_frames = int(fps * 1.5)  # Allow up to 1.5 seconds of gap
    line_x_interp = interpolate_line_positions_bidirectional(line_x_clean, max_gap_frames=max_gap_frames)
    gaps_filled = (~np.isnan(line_x_interp)).sum() - (~np.isnan(line_x_clean)).sum()
    if gaps_filled > 0:
        print(f"[INFO] Interpolated {gaps_filled} frames of missing line data")
    
    # Step 3: Apply AGGRESSIVE temporal smoothing to interpolated line positions
    # Using much larger kernel for very smooth line tracking
    if len(line_x_interp) > 5:
        print(f"[INFO] Applying aggressive smoothing to line positions...")
        # Use aggressive multi-stage smoothing with large kernels
        line_x = aggressive_smooth(line_x_interp, gaussian_window=51, moving_avg_window=25)
    else:
        line_x = line_x_interp
    
    valid_line_mask = ~np.isnan(line_x)
    ref_line_x = np.nanmedian(line_x) if valid_line_mask.any() else 0.0
    print(f"[INFO] Reference line median position: {ref_line_x:.1f}px")
    
    # Compute relative position (signed distance from line)
    # Positive = swimmer to the right of line
    # Negative = swimmer to the left of line (crossed the boundary)
    rel_x = np.full_like(swimmer_x, np.nan)
    last_known_line = None
    for i in range(len(rel_x)):
        lx = line_x[i] if not np.isnan(line_x[i]) else last_known_line
        if not np.isnan(line_x[i]):
            last_known_line = line_x[i]
        if np.isnan(swimmer_x[i]) or lx is None or np.isnan(lx):
            rel_x[i] = np.nan
        else:
            # Signed distance: positive when right of line, negative when left
            rel_x[i] = swimmer_x[i] - lx
    
    # Interpolate short gaps
    rel_x_interp = linear_interpolate_positions(times, rel_x, max_gap_seconds=interpolate_max_gap)
    swimmer_y_interp = linear_interpolate_positions(times, swimmer_y, max_gap_seconds=interpolate_max_gap)
    
    # Apply AGGRESSIVE smoothing to the interpolated swimmer position
    print(f"[INFO] Applying aggressive smoothing to swimmer positions...")
    rel_x_sm = aggressive_smooth(rel_x_interp, gaussian_window=41, moving_avg_window=21)
    
    # Also smooth Y position for consistency
    swimmer_y_smooth = aggressive_smooth(swimmer_y_interp, gaussian_window=41, moving_avg_window=21)
    
    # Compute velocity and acceleration
    vx = compute_derivative(times, rel_x_sm)
    ax = compute_derivative(times, vx)
    
    # Create output dataframe with all processing stages
    df = pd.DataFrame({
        "time_s": times,
        "swimmer_x": swimmer_x,
        "swimmer_y": swimmer_y,
        "swimmer_y_smooth": swimmer_y_smooth,  # Smoothed Y position
        "line_x_raw": line_x_list,  # Original raw detections
        "line_x_clean": line_x_clean,  # After outlier removal
        "line_x_interp": line_x_interp,  # After bidirectional interpolation
        "line_x": line_x,  # Final smoothed version (heavily smoothed)
        "rel_x_px": rel_x,
        "rel_x_interp_px": rel_x_interp,
        "rel_x_smooth_px": rel_x_sm,  # Heavily smoothed swimmer position
        "v_x_px_s": vx,
        "a_x_px_s2": ax
    })
    
    if out_csv:
        df.to_csv(out_csv, index=False)
        print(f"[INFO] Results written to {out_csv}")
    
    # Print statistics
    valid_swimmer = swimmer_x[~np.isnan(swimmer_x)]
    valid_line = line_x[~np.isnan(line_x)]
    valid_vx = vx[~np.isnan(vx)]
    valid_rel_x = rel_x_sm[~np.isnan(rel_x_sm)]
    
    print(f"\n[STATS] Tracking Statistics:")
    print(f"  Swimmer detected: {len(valid_swimmer)}/{len(swimmer_x)} frames ({100*len(valid_swimmer)/len(swimmer_x):.1f}%)")
    print(f"  Line detected: {len(valid_line)}/{len(line_x)} frames ({100*len(valid_line)/len(line_x):.1f}%)")
    
    if len(valid_rel_x) > 0:
        print(f"\n[STATS] Position Statistics (distance from line):")
        print(f"  Mean: {np.mean(valid_rel_x):+.2f} px (+ = right of line, - = left of line)")
        print(f"  Std: {np.std(valid_rel_x):.2f} px")
        print(f"  Min: {np.min(valid_rel_x):+.2f} px")
        print(f"  Max: {np.max(valid_rel_x):+.2f} px")
        crossed_line = (valid_rel_x < 0).sum()
        if crossed_line > 0:
            print(f"  Frames with swimmer left of line: {crossed_line} ({100*crossed_line/len(valid_rel_x):.1f}%)")
    
    if len(valid_vx) > 0:
        print(f"\n[STATS] Velocity Statistics:")
        print(f"  Mean: {np.mean(valid_vx):.2f} px/s")
        print(f"  Std: {np.std(valid_vx):.2f} px/s")
        print(f"  Min: {np.min(valid_vx):.2f} px/s")
        print(f"  Max: {np.max(valid_vx):.2f} px/s")
    
    if plot:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        # Line position plot - simplified
        axes[0].plot(times, df['line_x'], '-', linewidth=2, label='Line Position (smoothed)', color='red')
        axes[0].axhline(y=ref_line_x, color='red', linestyle='--', alpha=0.5, 
                       label=f'Median: {ref_line_x:.1f}px')
        axes[0].set_ylabel('Line Position (px)', fontsize=12)
        axes[0].set_title('Reference Line Tracking', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Velocity plot
        axes[1].plot(times, df['v_x_px_s'], label='Velocity', color='green', linewidth=2)
        axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1].set_ylabel('Velocity (px/s)', fontsize=12)
        axes[1].set_title('Swimmer Velocity', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Acceleration plot
        axes[2].plot(times, df['a_x_px_s2'], label='Acceleration', color='purple', linewidth=2)
        axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[2].set_xlabel('Time (s)', fontsize=12)
        axes[2].set_ylabel('Acceleration (px/s²)', fontsize=12)
        axes[2].set_title('Swimmer Acceleration', fontsize=14, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        import os
        base = os.path.basename(video_path)
        plot_name = f"analysis_{os.path.splitext(base)[0]}.png"
        plot_path = os.path.join(os.path.dirname(video_path) if os.path.dirname(video_path) else '.', plot_name)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Analysis plot saved to {plot_path}")
        
        plt.show()
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Interactive swimmer tracking with manual selection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python swimming_select.py --video complex_test.mp4
  python swimming_select.py --video fly.mp4 --out my_results.csv --no-plot
  
Instructions:
  1. First, you'll select a region where the reference line should be detected
  2. Then, you'll select the swimmer (or a point on the swimmer) to track
  3. Press SPACE or ENTER to confirm each selection
  4. Press 'q' during processing to stop early
        """
    )
    parser.add_argument('--video', required=True, help='Path to input video file')
    parser.add_argument('--out', default='selected_tracking_results.csv', 
                       help='CSV output path (default: selected_tracking_results.csv)')
    parser.add_argument('--no-plot', dest='plot', action='store_false', 
                       help='Disable plotting')
    parser.add_argument('--smooth-window', type=int, default=25, 
                       help='Smoothing window size for additional smoothing (default: 25, higher = smoother)')
    parser.add_argument('--interp-gap', type=float, default=0.25, 
                       help='Max gap to interpolate in seconds (default: 0.25)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("INTERACTIVE SWIMMER TRACKING WITH MANUAL SELECTION")
    print("=" * 70)
    
    df = process_video_with_selection(
        args.video, 
        out_csv=args.out, 
        plot=args.plot,
        interpolate_max_gap=args.interp_gap,
        smoothing_window=args.smooth_window
    )
    
    print("\n" + "=" * 70)
    print("✅ PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"\n[PREVIEW] First few rows of data:")
    print(df.head(10))


if __name__ == '__main__':
    main()

