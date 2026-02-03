#!/usr/bin/env python3
"""
Multi-point tracking with independent start/stop for each point.
Press '0'-'9' to toggle tracking for each point, 'q' to quit.
"""
import cv2
import numpy as np
import argparse
import pandas as pd
import os
from datetime import datetime

def get_tracking_output_path(filename, subdir='outputs'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(script_dir, subdir)
    os.makedirs(outputs_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    name, ext = os.path.splitext(filename)
    return os.path.join(outputs_dir, f"{name}_{timestamp}{ext}")

def create_csrt_tracker():
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    elif hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    else:
        raise RuntimeError("CSRT tracker not available")

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
                if gap_duration <= max_gap_seconds:
                    xs[i:j] = np.linspace(xs[i-1], xs[j], j - i)
            i = j
        else:
            i += 1
    return xs

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

def process_video_multi_start_stop(video_path, out_csv=None, interpolate_max_gap=0.25, 
                                   smoothing_iterations=3, blur_kernel=5, ewma_alpha=0.2, max_points=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, first_frame = cap.read()
    if not ret:
        raise IOError("Could not read first frame")

    base = os.path.basename(video_path)
    out_name = f"tracked_{os.path.splitext(base)[0]}_multi_start_stop.mp4"
    out_path = get_tracking_output_path(out_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    trackers = {}
    is_tracking = {}
    swimmer_data = {}
    frame_idx = 0

    blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        denoised = cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), 0)

        for point_id in list(trackers.keys()):
            if is_tracking.get(point_id, False) and trackers[point_id] is not None:
                try:
                    success, box = trackers[point_id].update(denoised)
                except Exception:
                    success = False

                if success:
                    x, y, w, h = [int(v) for v in box]
                    cx, cy = x + w // 2, y + h // 2

                    if cx >= width - 5:
                        is_tracking[point_id] = False
                        trackers[point_id] = None
                        continue

                    if point_id not in swimmer_data:
                        swimmer_data[point_id] = {'x_raw': [], 'y_raw': [], 'x_ewma': [], 'y_ewma': [], 'start_frame': None, 'frames': []}

                    if swimmer_data[point_id]['start_frame'] is None:
                        swimmer_data[point_id]['start_frame'] = frame_idx

                    swimmer_data[point_id]['x_raw'].append(float(cx))
                    swimmer_data[point_id]['y_raw'].append(float(cy))
                    swimmer_data[point_id]['frames'].append(frame_idx)

                    if len(swimmer_data[point_id]['x_ewma']) == 0:
                        swimmer_data[point_id]['x_ewma'].append(float(cx))
                        swimmer_data[point_id]['y_ewma'].append(float(cy))
                    else:
                        prev_x_ewma = swimmer_data[point_id]['x_ewma'][-1]
                        prev_y_ewma = swimmer_data[point_id]['y_ewma'][-1]
                        new_x_ewma = ewma_alpha * float(cx) + (1.0 - ewma_alpha) * prev_x_ewma
                        new_y_ewma = ewma_alpha * float(cy) + (1.0 - ewma_alpha) * prev_y_ewma
                        swimmer_data[point_id]['x_ewma'].append(new_x_ewma)
                        swimmer_data[point_id]['y_ewma'].append(new_y_ewma)

                    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
                             (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 255), (128, 255, 0)]
                    color = colors[point_id % len(colors)]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.circle(frame, (cx, cy), 6, color, -1)
                    cv2.putText(frame, f"Point {point_id}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    is_tracking[point_id] = False
                    trackers[point_id] = None

        active_points = sum(1 for v in is_tracking.values() if v)
        status_color = (0, 255, 0) if active_points > 0 else (0, 0, 255)
        cv2.putText(frame, f"Frame: {frame_idx}/{frame_count} | Active: {active_points}", (10, height - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(frame, "Press '0'-'9' to toggle points, 'q' to quit", (10, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out_writer.write(frame)
        display_frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
        cv2.imshow("Tracking Preview", display_frame)

        key = cv2.waitKey(1) & 0xFF
        
        if ord('0') <= key <= ord('0') + max_points - 1:
            point_id = key - ord('0')
            
            if is_tracking.get(point_id, False):
                is_tracking[point_id] = False
                trackers[point_id] = None
            else:
                bbox = cv2.selectROI("Tracking Preview", display_frame, fromCenter=False)
                cv2.destroyWindow("Tracking Preview")
                if bbox != (0, 0, 0, 0):
                    bbox_full = (int(bbox[0]*2), int(bbox[1]*2), int(bbox[2]*2), int(bbox[3]*2))
                    tracker = create_csrt_tracker()
                    try:
                        tracker.init(denoised, bbox_full)
                    except Exception:
                        tracker.init(frame, bbox_full)
                    
                    trackers[point_id] = tracker
                    is_tracking[point_id] = True
                    
                    if point_id not in swimmer_data:
                        swimmer_data[point_id] = {'x_raw': [], 'y_raw': [], 'x_ewma': [], 'y_ewma': [], 'start_frame': None, 'frames': []}
                    
                cv2.imshow("Tracking Preview", display_frame)
        
        elif key == ord('q'):
            break

        frame_idx += 1

    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()

    for obj_id in list(swimmer_data.keys()):
        data = swimmer_data[obj_id]
        if len(data.get('x_ewma', [])) == 0:
            data['x'] = data.get('x_raw', [])
            data['y'] = data.get('y_raw', [])
        else:
            data['x'] = data['x_ewma']
            data['y'] = data['y_ewma']

    times = np.arange(frame_idx) / fps
    csv_rows = []

    for obj_id in sorted(swimmer_data.keys()):
        data = swimmer_data[obj_id]
        start_frame = data['start_frame'] or 0

        swimmer_x = np.full(frame_idx, np.nan)
        swimmer_y = np.full(frame_idx, np.nan)

        for i in range(len(data['x'])):
            frame_num = data['frames'][i] if i < len(data['frames']) else (start_frame + i)
            if frame_num < frame_idx:
                swimmer_x[frame_num] = data['x'][i]
                swimmer_y[frame_num] = data['y'][i]

        swimmer_y_interp = linear_interpolate_positions(times, swimmer_y, max_gap_seconds=interpolate_max_gap)
        swimmer_x_interp = linear_interpolate_positions(times, swimmer_x, max_gap_seconds=interpolate_max_gap)

        swimmer_y_smooth = aggressive_smooth(swimmer_y_interp, gaussian_window=41, moving_avg_window=21)
        swimmer_x_smooth = aggressive_smooth(swimmer_x_interp, gaussian_window=41, moving_avg_window=21)

        for iteration in range(smoothing_iterations):
            swimmer_y_smooth = three_point_smooth(swimmer_y_smooth)
            swimmer_x_smooth = three_point_smooth(swimmer_x_smooth)

        for frame_num in range(start_frame, frame_idx):
            x_val = swimmer_x[frame_num]
            y_val = swimmer_y[frame_num]
            x_smooth_val = swimmer_x_smooth[frame_num]
            y_smooth_val = swimmer_y_smooth[frame_num]

            if not (np.isnan(x_val) and np.isnan(y_val) and np.isnan(x_smooth_val) and np.isnan(y_smooth_val)):
                csv_rows.append({
                    'Frame': frame_num,
                    'ObjectID': obj_id,
                    'X': x_val if not np.isnan(x_val) else '',
                    'Y': y_val if not np.isnan(y_val) else '',
                    'X_smooth': x_smooth_val if not np.isnan(x_smooth_val) else '',
                    'Y_smooth': y_smooth_val if not np.isnan(y_smooth_val) else ''
                })

    df = pd.DataFrame(csv_rows)
    if len(df) > 0:
        df = df.sort_values(['ObjectID', 'Frame']).reset_index(drop=True)
        df = df[['Frame', 'ObjectID', 'X', 'Y', 'X_smooth', 'Y_smooth']]

    if out_csv:
        df.to_csv(out_csv, index=False)

    return df


def main():
    parser = argparse.ArgumentParser(description="Multi-point tracking with number key controls.")
    parser.add_argument('--video', required=True, help='Path to input video file')
    parser.add_argument('--out', default=None, help='CSV output path (default: timestamped in tracking/outputs/)')
    parser.add_argument('--smooth', type=int, default=3, help='Smoothing iterations (default: 3)')
    parser.add_argument('--interp-gap', type=float, default=0.25, help='Max gap to interpolate in seconds (default: 0.25)')
    parser.add_argument('--ewma-alpha', type=float, default=0.2, help='EWMA alpha (default: 0.2)')
    parser.add_argument('--blur-k', type=int, default=5, help='Blur kernel size (default: 5)')
    parser.add_argument('--max-points', type=int, default=10, help='Max points to track (default: 10)')

    args = parser.parse_args()

    if args.out is None:
        args.out = get_tracking_output_path('tracker_multi_start_stop_results.csv')

    process_video_multi_start_stop(
        args.video,
        out_csv=args.out,
        interpolate_max_gap=args.interp_gap,
        smoothing_iterations=args.smooth,
        blur_kernel=args.blur_k,
        ewma_alpha=args.ewma_alpha,
        max_points=args.max_points
    )


if __name__ == '__main__':
    main()
