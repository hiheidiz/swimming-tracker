#!/usr/bin/env python3
"""
animate_graph.py

Create an animated video combining the original video with live velocity graphs.

Usage:
    python animate_graph.py --video SHOW_TEST_VIDEO.mp4 --csv insta_tracking_results.csv

Dependencies:
    pip install opencv-python numpy matplotlib pandas
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from matplotlib.backends.backend_agg import FigureCanvasAgg
import argparse
import os

def create_graph_frame(times, velocities, current_idx, fig_size=(8, 6)):
    """
    Create a matplotlib figure showing velocity up to current time with a marker.
    
    Returns:
        numpy array (image) of the graph
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=fig_size, gridspec_kw={'height_ratios': [2, 1]})
    
    current_time = times[current_idx]
    
    # Plot 1: Velocity graph with history and current position
    ax1.plot(times[:current_idx+1], velocities[:current_idx+1], 
             color='purple', linewidth=2, label='Velocity')
    ax1.plot(times[current_idx+1:], velocities[current_idx+1:], 
             color='lightgray', linewidth=1, alpha=0.3, label='Future')
    
    # Mark current position
    if not np.isnan(velocities[current_idx]):
        ax1.plot(current_time, velocities[current_idx], 'ro', markersize=12, 
                label=f'Now: {velocities[current_idx]:.1f} px/s')
    
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_ylabel('Velocity (px/s)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Swimmer Velocity Over Time (t = {current_time:.2f}s)', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([times[0], times[-1]])
    
    # Set y-limits based on data
    valid_vel = velocities[~np.isnan(velocities)]
    if len(valid_vel) > 0:
        v_min, v_max = np.min(valid_vel), np.max(valid_vel)
        margin = (v_max - v_min) * 0.1
        ax1.set_ylim([v_min - margin, v_max + margin])
    
    # Plot 2: Position graph with history
    # Calculate normalized position by integrating velocity
    positions = np.nancumsum(velocities * np.diff(times, prepend=times[0]))
    
    ax2.plot(times[:current_idx+1], positions[:current_idx+1], 
             color='blue', linewidth=2, label='Position')
    ax2.plot(times[current_idx+1:], positions[current_idx+1:], 
             color='lightgray', linewidth=1, alpha=0.3)
    
    # Mark current position
    if not np.isnan(positions[current_idx]):
        ax2.plot(current_time, positions[current_idx], 'bo', markersize=12,
                label=f'Now: {positions[current_idx]:.1f} px')
    
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Position (px)', fontsize=12, fontweight='bold')
    ax2.set_title('Cumulative Position', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([times[0], times[-1]])
    
    plt.tight_layout()
    
    # Convert to numpy array
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    graph_img = np.asarray(buf)
    
    plt.close(fig)
    
    # Convert RGBA to BGR for OpenCV
    graph_img = cv2.cvtColor(graph_img, cv2.COLOR_RGBA2BGR)
    
    return graph_img


def combine_video_and_graph(video_path, csv_path, output_path, 
                            layout='overlay', graph_size=(800, 600)):
    """
    Create a video combining original video with animated velocity graph.
    
    Args:
        video_path: path to original video
        csv_path: path to CSV with tracking data
        output_path: path for output video
        layout: 'overlay', 'side-by-side' or 'top-bottom'
        graph_size: (width, height) for graph (ignored for overlay)
    """
    # Load CSV data
    print(f"[INFO] Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    times = df['time_s'].values
    
    # Try to get velocity data from different possible column names
    if 'v_x_px_s' in df.columns:
        velocities = df['v_x_px_s'].values
        print(f"[INFO] Using velocity column: 'v_x_px_s'")
    elif 'velocity' in df.columns:
        velocities = df['velocity'].values
        print(f"[INFO] Using velocity column: 'velocity'")
    else:
        raise ValueError(f"Could not find velocity column. Available columns: {list(df.columns)}")
    
    print(f"[INFO] Found {len(velocities)} velocity data points")
    
    # Load video
    print(f"[INFO] Loading video from {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[INFO] Video: {video_width}x{video_height}, {fps} FPS, {frame_count} frames")
    print(f"[INFO] Data: {len(df)} data points")
    
    # Calculate output dimensions and graph size
    if layout == 'overlay':
        output_width = video_width
        output_height = video_height
        # Graph is 1/4 of video width, height maintains aspect ratio
        graph_width = video_width // 4
        graph_height = int(graph_width * 0.75)  # 4:3 aspect ratio
        print(f"[INFO] Overlay graph size: {graph_width}x{graph_height}")
    elif layout == 'side-by-side':
        graph_width, graph_height = graph_size
        output_width = video_width + graph_width
        output_height = max(video_height, graph_height)
    else:  # top-bottom
        graph_width, graph_height = graph_size
        output_width = max(video_width, graph_width)
        output_height = video_height + graph_height
    
    print(f"[INFO] Output: {output_width}x{output_height}")
    
    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    
    print(f"[INFO] Creating animated video: {output_path}")
    print(f"[INFO] Layout: {layout}")
    
    frame_idx = 0
    
    while frame_idx < min(frame_count, len(df)):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create graph for current frame
        graph_img = create_graph_frame(times, velocities, frame_idx, 
                                       fig_size=(graph_width/100, graph_height/100))
        
        # Resize graph to exact dimensions
        graph_img = cv2.resize(graph_img, (graph_width, graph_height))
        
        # Create canvas
        if layout == 'overlay':
            # Start with the video frame
            canvas = frame.copy()
            
            # Calculate position for bottom-right corner with margin
            margin = 20
            x_pos = video_width - graph_width - margin
            y_pos = video_height - graph_height - margin
            
            # Add semi-transparent background for graph
            overlay_bg = canvas[y_pos:y_pos+graph_height, x_pos:x_pos+graph_width].copy()
            # Blend graph with background (80% graph, 20% background)
            blended = cv2.addWeighted(graph_img, 0.9, overlay_bg, 0.1, 0)
            
            # Place the blended graph on canvas
            canvas[y_pos:y_pos+graph_height, x_pos:x_pos+graph_width] = blended
            
            # Add border around graph
            cv2.rectangle(canvas, (x_pos-2, y_pos-2), 
                         (x_pos+graph_width+2, y_pos+graph_height+2), 
                         (255, 255, 255), 2)
        
        elif layout == 'side-by-side':
            canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            # Place video on left
            canvas[:video_height, :video_width] = frame
            # Place graph on right
            canvas[:graph_height, video_width:video_width+graph_width] = graph_img
        
        else:  # top-bottom
            canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            # Place video on top
            canvas[:video_height, :video_width] = frame
            # Place graph on bottom
            canvas[video_height:video_height+graph_height, :graph_width] = graph_img
        
        # Add frame counter
        cv2.putText(canvas, f"Frame: {frame_idx}/{frame_count} | Time: {times[frame_idx]:.2f}s", 
                   (10, output_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(canvas)
        
        if frame_idx % 50 == 0:
            print(f"[INFO] Processed {frame_idx}/{frame_count} frames ({100*frame_idx/frame_count:.1f}%)")
        
        frame_idx += 1
    
    cap.release()
    out.release()
    
    print(f"\n[SUCCESS] Animated video created: {output_path}")
    print(f"[INFO] Total frames: {frame_idx}")


def create_graph_only_video(csv_path, output_path, duration=None, fps=30, size=(1920, 1080)):
    """
    Create a video of just the animated graph (no original video).
    """
    # Load CSV data
    print(f"[INFO] Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    times = df['time_s'].values
    velocities = df['v_x_px_s'].values
    
    # Calculate number of frames
    if duration is None:
        duration = times[-1]
    
    frame_count = int(duration * fps)
    
    print(f"[INFO] Creating graph-only video: {output_path}")
    print(f"[INFO] Duration: {duration:.2f}s, FPS: {fps}, Frames: {frame_count}")
    print(f"[INFO] Size: {size[0]}x{size[1]}")
    
    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, size)
    
    for frame_idx in range(frame_count):
        current_time = frame_idx / fps
        
        # Find closest data point
        data_idx = np.argmin(np.abs(times - current_time))
        
        # Create graph
        graph_img = create_graph_frame(times, velocities, data_idx, 
                                       fig_size=(size[0]/100, size[1]/100))
        
        # Resize to exact size
        graph_img = cv2.resize(graph_img, size)
        
        out.write(graph_img)
        
        if frame_idx % 50 == 0:
            print(f"[INFO] Processed {frame_idx}/{frame_count} frames ({100*frame_idx/frame_count:.1f}%)")
    
    out.release()
    
    print(f"\n[SUCCESS] Graph video created: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create animated video with velocity graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Overlay layout (default - graph in bottom-right corner)
  python animate_graph.py --video SHOW_TEST_VIDEO.mp4 --csv selected_tracking_results.csv
  
  # Side-by-side layout (video + graph)
  python animate_graph.py --video SHOW_TEST_VIDEO.mp4 --csv selected_tracking_results.csv --layout side-by-side
  
  # Top-bottom layout
  python animate_graph.py --video SHOW_TEST_VIDEO.mp4 --csv selected_tracking_results.csv --layout top-bottom
  
  # Graph only (no video)
  python animate_graph.py --csv selected_tracking_results.csv --graph-only
  
  # Custom output name
  python animate_graph.py --video SHOW_TEST_VIDEO.mp4 --csv selected_tracking_results.csv --output my_animated.mp4
        """
    )
    parser.add_argument('--video', help='Path to original video file (required unless --graph-only)')
    parser.add_argument('--csv', required=True, help='Path to CSV file with tracking data')
    parser.add_argument('--output', help='Output video path (auto-generated if not provided)')
    parser.add_argument('--layout', choices=['overlay', 'side-by-side', 'top-bottom'], default='overlay',
                       help='Layout for video+graph (default: overlay - bottom-right corner)')
    parser.add_argument('--graph-size', type=int, nargs=2, default=[800, 600],
                       metavar=('WIDTH', 'HEIGHT'),
                       help='Graph size in pixels (default: 800 600)')
    parser.add_argument('--graph-only', action='store_true',
                       help='Create graph-only video (no original video)')
    parser.add_argument('--fps', type=int, default=30,
                       help='FPS for graph-only video (default: 30)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.graph_only and not args.video:
        parser.error("--video is required unless --graph-only is specified")
    
    # Generate output path if not provided
    if not args.output:
        csv_base = os.path.splitext(os.path.basename(args.csv))[0]
        if args.graph_only:
            args.output = f"graph_animation_{csv_base}.mp4"
        else:
            video_base = os.path.splitext(os.path.basename(args.video))[0]
            if args.layout == 'overlay':
                args.output = f"animated_{video_base}.mp4"
            else:
                args.output = f"animated_{video_base}_{args.layout.replace('-', '_')}.mp4"
    
    print("=" * 70)
    print("ANIMATED VELOCITY GRAPH VIDEO CREATOR")
    print("=" * 70)
    
    if args.graph_only:
        create_graph_only_video(
            csv_path=args.csv,
            output_path=args.output,
            fps=args.fps,
            size=tuple(args.graph_size) if len(args.graph_size) == 2 else (1920, 1080)
        )
    else:
        combine_video_and_graph(
            video_path=args.video,
            csv_path=args.csv,
            output_path=args.output,
            layout=args.layout,
            graph_size=tuple(args.graph_size)
        )
    
    print("\n" + "=" * 70)
    print("✅ DONE!")
    print("=" * 70)


if __name__ == '__main__':
    main()

