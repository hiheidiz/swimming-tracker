#!/usr/bin/env python3
"""
Plot tracking data in meters based on video width being 8.5 feet.
Video is recorded at 20x slow motion.
Creates two graphs:
1. Position over time (x and y in meters)
2. X-velocity over time (in m/s)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV data
df = pd.read_csv('SHOW_tracker_final_results.csv')

# Video dimensions
VIDEO_WIDTH_FEET = 8.5
VIDEO_WIDTH_METERS = VIDEO_WIDTH_FEET * 0.3048  # Convert feet to meters
SLOWMO_FACTOR = 20  # Video is at 20x slow motion

# Determine video width in pixels
# We'll use the maximum x position from all points
video_width_px = 0
for point_id in range(4):
    x_col = f'point_{point_id}_x'
    if x_col in df.columns:
        max_x = df[x_col].max()
        if not np.isnan(max_x):
            video_width_px = max(video_width_px, max_x)

print(f"Video width: {VIDEO_WIDTH_FEET} feet = {VIDEO_WIDTH_METERS:.4f} meters")
print(f"Video width in pixels: {video_width_px:.0f} px")
print(f"Slow motion factor: {SLOWMO_FACTOR}x")

# Conversion factor: meters per pixel
meters_per_pixel = VIDEO_WIDTH_METERS / video_width_px
print(f"Conversion factor: {meters_per_pixel:.6f} meters/pixel")

# Prepare data for plotting
# Adjust time for slow motion (divide by slowmo factor to get real time)
times = df['time_s'].values / SLOWMO_FACTOR
point_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, orange, green, red
point_labels = ['Point 0', 'Point 1', 'Point 2', 'Point 3']

# Create figure with 2 subplots
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle(f'Swimming Analysis (Video: {VIDEO_WIDTH_FEET} ft = {VIDEO_WIDTH_METERS:.3f} m, {SLOWMO_FACTOR}x Slow Motion)', 
             fontsize=16, weight='bold')

# Plot 1: Position over time (x and y in meters)
ax1 = axes[0]

for point_id in range(4):
    # Get column names
    x_col = f'point_{point_id}_rel_x_smooth_px'  # Distance from reference line
    y_col = f'point_{point_id}_y_smooth'
    start_frame_col = f'point_{point_id}_start_frame'
    
    if x_col in df.columns and y_col in df.columns:
        # Convert to meters
        x_meters = df[x_col].values * meters_per_pixel
        y_meters = df[y_col].values * meters_per_pixel
        
        # Get start time
        start_frame = df[start_frame_col].iloc[0]
        if not np.isnan(start_frame):
            start_time = times[int(start_frame)]
        else:
            start_time = 0
        
        # Filter out NaN values for plotting
        valid_mask = ~(np.isnan(x_meters) | np.isnan(y_meters))
        
        if valid_mask.any():
            # Plot x position
            ax1.plot(times[valid_mask], x_meters[valid_mask], 
                    color=point_colors[point_id], linewidth=2, 
                    label=f'{point_labels[point_id]} (X)', linestyle='-')
            
            # Plot y position
            ax1.plot(times[valid_mask], y_meters[valid_mask], 
                    color=point_colors[point_id], linewidth=2, 
                    label=f'{point_labels[point_id]} (Y)', linestyle='--', alpha=0.7)
            
            # Mark start time
            ax1.axvline(x=start_time, color=point_colors[point_id], 
                       linestyle=':', alpha=0.3, linewidth=1)

ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
ax1.set_xlabel('Real Time (s)', fontsize=12)
ax1.set_ylabel('Position (m)', fontsize=12)
ax1.set_title('Position Over Time (Distance from Reference Line for X)', fontsize=14, fontweight='bold')
ax1.legend(loc='best', fontsize=9, ncol=2)
ax1.grid(True, alpha=0.3)

# Plot 2: X-velocity over time (in m/s)
ax2 = axes[1]

for point_id in range(4):
    # Get velocity column
    vx_col = f'point_{point_id}_v_x_px_s'
    start_frame_col = f'point_{point_id}_start_frame'
    
    if vx_col in df.columns:
        # Convert to m/s and adjust for slow motion (multiply by slowmo factor)
        vx_ms = df[vx_col].values * meters_per_pixel * SLOWMO_FACTOR
        
        # Get start time
        start_frame = df[start_frame_col].iloc[0]
        if not np.isnan(start_frame):
            start_time = times[int(start_frame)]
        else:
            start_time = 0
        
        # Filter out NaN values
        valid_mask = ~np.isnan(vx_ms)
        
        if valid_mask.any():
            ax2.plot(times[valid_mask], vx_ms[valid_mask], 
                    color=point_colors[point_id], linewidth=2, 
                    label=point_labels[point_id])
            
            # Mark start time
            ax2.axvline(x=start_time, color=point_colors[point_id], 
                       linestyle=':', alpha=0.3, linewidth=1)

ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
ax2.set_xlabel('Real Time (s)', fontsize=12)
ax2.set_ylabel('X-Velocity (m/s)', fontsize=12)
ax2.set_title('X-Direction Velocity Over Time (Real Speed)', fontsize=14, fontweight='bold')
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Save the figure
output_file = 'analysis_meters_SHOW.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✅ Plot saved to: {output_file}")

# plt.show()  # Comment out to avoid blocking

# Print some statistics
print("\n" + "="*70)
print("STATISTICS (in meters and m/s)")
print("="*70)

for point_id in range(4):
    x_col = f'point_{point_id}_rel_x_smooth_px'
    vx_col = f'point_{point_id}_v_x_px_s'
    start_frame_col = f'point_{point_id}_start_frame'
    
    if x_col in df.columns and vx_col in df.columns:
        x_meters = df[x_col].values * meters_per_pixel
        vx_ms = df[vx_col].values * meters_per_pixel * SLOWMO_FACTOR
        
        valid_x = x_meters[~np.isnan(x_meters)]
        valid_vx = vx_ms[~np.isnan(vx_ms)]
        
        start_frame = df[start_frame_col].iloc[0]
        
        if len(valid_x) > 0:
            print(f"\nPoint {point_id}:")
            print(f"  Started at frame: {int(start_frame)}")
            print(f"  X Position (distance from line):")
            print(f"    Mean: {np.mean(valid_x):+.4f} m")
            print(f"    Range: {np.min(valid_x):+.4f} to {np.max(valid_x):+.4f} m")
            print(f"    Total displacement: {valid_x[-1] - valid_x[0]:.4f} m")
        
        if len(valid_vx) > 0:
            print(f"  X Velocity:")
            print(f"    Mean: {np.mean(valid_vx):.4f} m/s")
            print(f"    Max: {np.max(valid_vx):.4f} m/s")
            print(f"    Min: {np.min(valid_vx):.4f} m/s")

