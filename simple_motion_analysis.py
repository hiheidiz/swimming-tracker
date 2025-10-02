import csv
import numpy as np
import matplotlib.pyplot as plt

def simple_motion_analysis(csv_file="tracked_positions.csv", window_size=5):
    """
    Simple motion analysis showing position, velocity, and acceleration over time.
    """
    # Read CSV data
    frames = []
    x_positions = []
    y_positions = []
    
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            frames.append(int(row['frame']))
            # Handle None values
            if row['x'] == 'None' or row['x'] == '':
                x_positions.append(None)
            else:
                x_positions.append(float(row['x']))
            if row['y'] == 'None' or row['y'] == '':
                y_positions.append(None)
            else:
                y_positions.append(float(row['y']))
    
    frames = np.array(frames)
    x_positions = np.array(x_positions)
    y_positions = np.array(y_positions)
    
    # Calculate average positions over window_size frames
    averaged_frames = []
    averaged_x_positions = []
    averaged_y_positions = []
    
    for i in range(0, len(frames), window_size):
        # Get the window of frames
        end_idx = min(i + window_size, len(frames))
        window_frames = frames[i:end_idx]
        window_x = x_positions[i:end_idx]
        window_y = y_positions[i:end_idx]
        
        # Only include windows that have at least one valid x position
        valid_x_mask = ~np.isnan(window_x)
        if np.any(valid_x_mask):
            # Use the middle frame number as representative
            avg_frame = np.mean(window_frames)
            # Average only valid x positions
            avg_x = np.nanmean(window_x)
            avg_y = np.nanmean(window_y)
            
            averaged_frames.append(avg_frame)
            averaged_x_positions.append(avg_x)
            averaged_y_positions.append(avg_y)
    
    averaged_frames = np.array(averaged_frames)
    averaged_x_positions = np.array(averaged_x_positions)
    averaged_y_positions = np.array(averaged_y_positions)
    
    # Calculate x-velocity (pixels per frame)
    x_velocities = np.diff(averaged_x_positions)
    velocity_frames = averaged_frames[1:]
    
    # Calculate x-acceleration (change in velocity per frame)
    x_accelerations = np.diff(x_velocities)
    acceleration_frames = velocity_frames[1:]
    
    # Create the three main plots
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Position over time
    plt.subplot(1, 3, 1)
    plt.plot(averaged_frames, averaged_x_positions, 'b-o', markersize=4, linewidth=2)
    plt.xlabel('Frame Number')
    plt.ylabel('X Position (pixels)')
    plt.title('Position Over Time')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Velocity over time
    plt.subplot(1, 3, 2)
    plt.plot(velocity_frames, x_velocities, 'r-o', markersize=4, linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Frame Number')
    plt.ylabel('X-Velocity (pixels/frame)')
    plt.title('Velocity Over Time')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Acceleration over time
    plt.subplot(1, 3, 3)
    plt.plot(acceleration_frames, x_accelerations, 'purple', marker='o', markersize=4, linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Frame Number')
    plt.ylabel('X-Acceleration (pixels/frame²)')
    plt.title('Acceleration Over Time')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simple_motion_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print basic statistics
    print("📊 MOTION ANALYSIS SUMMARY")
    print("=" * 40)
    print(f"Data points: {len(averaged_frames)}")
    print(f"Mean velocity: {np.mean(x_velocities):.2f} pixels/frame")
    print(f"Mean acceleration: {np.mean(x_accelerations):.2f} pixels/frame²")
    print(f"Max velocity: {np.max(x_velocities):.2f} pixels/frame")
    print(f"Max acceleration: {np.max(x_accelerations):.2f} pixels/frame²")
    print(f"✅ Graph saved as: simple_motion_analysis.png")

if __name__ == "__main__":
    simple_motion_analysis()
