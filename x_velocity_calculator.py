import csv
import numpy as np
import matplotlib.pyplot as plt

def calculate_x_velocity_with_averaging(csv_file, window_size=5):
    """
    Calculate velocity using only x-axis values with frame averaging.
    
    Args:
        csv_file (str): Path to the CSV file with tracking data
        window_size (int): Number of frames to average over (default: 5)
    
    Returns:
        tuple: (averaged_frames, averaged_x_positions, x_velocities)
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
    
    print(f"📊 Original data: {len(frames)} frames")
    print(f"   Valid x positions: {np.sum(~np.isnan(x_positions))}")
    print(f"   Valid y positions: {np.sum(~np.isnan(y_positions))}")
    
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
            avg_y = np.nanmean(window_y)  # Also calculate y for reference
            
            averaged_frames.append(avg_frame)
            averaged_x_positions.append(avg_x)
            averaged_y_positions.append(avg_y)
    
    averaged_frames = np.array(averaged_frames)
    averaged_x_positions = np.array(averaged_x_positions)
    averaged_y_positions = np.array(averaged_y_positions)
    
    print(f"📊 After averaging (window={window_size}): {len(averaged_frames)} data points")
    
    # Calculate x-velocity (pixels per frame)
    x_velocities = np.diff(averaged_x_positions)
    velocity_frames = averaged_frames[1:]  # One less element due to diff
    
    # Calculate x-acceleration (change in velocity per frame)
    x_accelerations = np.diff(x_velocities)
    acceleration_frames = velocity_frames[1:]  # One less element due to diff
    
    return averaged_frames, averaged_x_positions, x_velocities, averaged_y_positions, velocity_frames, x_accelerations, acceleration_frames

def plot_x_velocity_analysis(frames, x_positions, x_velocities, y_positions, velocity_frames, x_accelerations, acceleration_frames, window_size):
    """
    Plot the x-velocity analysis results.
    """
    plt.figure(figsize=(18, 15))
    
    # Plot 1: X-position over time
    plt.subplot(3, 3, 1)
    plt.plot(frames, x_positions, 'b-o', markersize=4, linewidth=2)
    plt.xlabel('Frame Number')
    plt.ylabel('X Position (pixels)')
    plt.title('X-Position Over Time (Averaged)')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: X-velocity over time
    plt.subplot(3, 3, 2)
    plt.plot(velocity_frames, x_velocities, 'r-o', markersize=4, linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Frame Number')
    plt.ylabel('X-Velocity (pixels/frame)')
    plt.title('X-Velocity Over Time')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: X-acceleration over time
    plt.subplot(3, 3, 3)
    plt.plot(acceleration_frames, x_accelerations, 'purple', marker='o', markersize=4, linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Frame Number')
    plt.ylabel('X-Acceleration (pixels/frame²)')
    plt.title('X-Acceleration Over Time')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Y-position for reference
    plt.subplot(3, 3, 4)
    plt.plot(frames, y_positions, 'g-o', markersize=4, linewidth=2)
    plt.xlabel('Frame Number')
    plt.ylabel('Y Position (pixels)')
    plt.title('Y-Position Over Time (Averaged)')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: X-velocity distribution
    plt.subplot(3, 3, 5)
    plt.hist(x_velocities, bins=15, alpha=0.7, color='red', edgecolor='black')
    plt.xlabel('X-Velocity (pixels/frame)')
    plt.ylabel('Frequency')
    plt.title('X-Velocity Distribution')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: X-acceleration distribution
    plt.subplot(3, 3, 6)
    plt.hist(x_accelerations, bins=15, alpha=0.7, color='purple', edgecolor='black')
    plt.xlabel('X-Acceleration (pixels/frame²)')
    plt.ylabel('Frequency')
    plt.title('X-Acceleration Distribution')
    plt.grid(True, alpha=0.3)
    
    # Plot 7: Movement direction analysis
    plt.subplot(3, 3, 7)
    positive_vel = x_velocities > 0
    negative_vel = x_velocities < 0
    zero_vel = x_velocities == 0
    
    plt.scatter(velocity_frames[positive_vel], x_velocities[positive_vel], 
                c='red', alpha=0.7, label='Moving Right', s=30)
    plt.scatter(velocity_frames[negative_vel], x_velocities[negative_vel], 
                c='blue', alpha=0.7, label='Moving Left', s=30)
    plt.scatter(velocity_frames[zero_vel], x_velocities[zero_vel], 
                c='green', alpha=0.7, label='No X Movement', s=30)
    
    plt.xlabel('Frame Number')
    plt.ylabel('X-Velocity (pixels/frame)')
    plt.title('Movement Direction Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 8: Acceleration analysis
    plt.subplot(3, 3, 8)
    positive_acc = x_accelerations > 0
    negative_acc = x_accelerations < 0
    zero_acc = x_accelerations == 0
    
    plt.scatter(acceleration_frames[positive_acc], x_accelerations[positive_acc], 
                c='red', alpha=0.7, label='Accelerating', s=30)
    plt.scatter(acceleration_frames[negative_acc], x_accelerations[negative_acc], 
                c='blue', alpha=0.7, label='Decelerating', s=30)
    plt.scatter(acceleration_frames[zero_acc], x_accelerations[zero_acc], 
                c='green', alpha=0.7, label='Constant Speed', s=30)
    
    plt.xlabel('Frame Number')
    plt.ylabel('X-Acceleration (pixels/frame²)')
    plt.title('Acceleration Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 9: Speed zones based on x-velocity
    plt.subplot(3, 3, 9)
    abs_velocities = np.abs(x_velocities)
    slow_threshold = np.percentile(abs_velocities, 33)
    fast_threshold = np.percentile(abs_velocities, 67)
    
    slow_mask = abs_velocities < slow_threshold
    medium_mask = (abs_velocities >= slow_threshold) & (abs_velocities < fast_threshold)
    fast_mask = abs_velocities >= fast_threshold
    
    plt.scatter(velocity_frames[slow_mask], x_velocities[slow_mask], 
                c='green', alpha=0.6, label=f'Slow (< {slow_threshold:.1f})', s=30)
    plt.scatter(velocity_frames[medium_mask], x_velocities[medium_mask], 
                c='orange', alpha=0.6, label=f'Medium ({slow_threshold:.1f}-{fast_threshold:.1f})', s=30)
    plt.scatter(velocity_frames[fast_mask], x_velocities[fast_mask], 
                c='red', alpha=0.6, label=f'Fast (> {fast_threshold:.1f})', s=30)
    
    plt.xlabel('Frame Number')
    plt.ylabel('X-Velocity (pixels/frame)')
    plt.title('X-Velocity Speed Zones')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('x_velocity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_x_velocity_data(frames, x_positions, x_velocities, y_positions, x_accelerations, velocity_frames, acceleration_frames, output_file='x_velocity_data.csv'):
    """
    Save x-velocity and acceleration data to CSV file.
    """
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['frame', 'x_position', 'y_position', 'x_velocity', 'x_acceleration'])
        
        # Write all data points
        for i in range(len(frames)):
            frame = frames[i]
            x_pos = x_positions[i]
            y_pos = y_positions[i]
            
            # Find corresponding velocity
            vel_idx = np.where(velocity_frames == frame)[0]
            velocity = x_velocities[vel_idx[0]] if len(vel_idx) > 0 else 'N/A'
            
            # Find corresponding acceleration
            acc_idx = np.where(acceleration_frames == frame)[0]
            acceleration = x_accelerations[acc_idx[0]] if len(acc_idx) > 0 else 'N/A'
            
            writer.writerow([frame, x_pos, y_pos, velocity, acceleration])
    
    print(f"✅ X-velocity and acceleration data saved to {output_file}")

def print_x_velocity_statistics(x_velocities, x_accelerations):
    """
    Print comprehensive x-velocity statistics.
    """
    print("\n📊 X-VELOCITY STATISTICS")
    print("=" * 50)
    print(f"Mean x-velocity: {np.mean(x_velocities):.2f} pixels/frame")
    print(f"Median x-velocity: {np.median(x_velocities):.2f} pixels/frame")
    print(f"Standard deviation: {np.std(x_velocities):.2f} pixels/frame")
    print(f"Min x-velocity: {np.min(x_velocities):.2f} pixels/frame")
    print(f"Max x-velocity: {np.max(x_velocities):.2f} pixels/frame")
    print(f"25th percentile: {np.percentile(x_velocities, 25):.2f} pixels/frame")
    print(f"75th percentile: {np.percentile(x_velocities, 75):.2f} pixels/frame")
    
    # Movement direction analysis
    positive_movement = np.sum(x_velocities > 0)
    negative_movement = np.sum(x_velocities < 0)
    no_movement = np.sum(x_velocities == 0)
    total_movements = len(x_velocities)
    
    print(f"\n🏃 MOVEMENT DIRECTION ANALYSIS")
    print("=" * 50)
    print(f"Moving right (positive): {positive_movement} frames ({positive_movement/total_movements*100:.1f}%)")
    print(f"Moving left (negative): {negative_movement} frames ({negative_movement/total_movements*100:.1f}%)")
    print(f"No x-movement: {no_movement} frames ({no_movement/total_movements*100:.1f}%)")
    
    # Speed analysis
    abs_velocities = np.abs(x_velocities)
    slow_threshold = np.percentile(abs_velocities, 33)
    fast_threshold = np.percentile(abs_velocities, 67)
    
    slow_frames = np.sum(abs_velocities < slow_threshold)
    medium_frames = np.sum((abs_velocities >= slow_threshold) & (abs_velocities < fast_threshold))
    fast_frames = np.sum(abs_velocities >= fast_threshold)
    
    print(f"\n⚡ SPEED ANALYSIS (X-axis)")
    print("=" * 50)
    print(f"Slow movement: {slow_frames} frames ({slow_frames/total_movements*100:.1f}%)")
    print(f"Medium movement: {medium_frames} frames ({medium_frames/total_movements*100:.1f}%)")
    print(f"Fast movement: {fast_frames} frames ({fast_frames/total_movements*100:.1f}%)")
    
    # Acceleration statistics
    print(f"\n⚡ X-ACCELERATION STATISTICS")
    print("=" * 50)
    print(f"Mean x-acceleration: {np.mean(x_accelerations):.2f} pixels/frame²")
    print(f"Median x-acceleration: {np.median(x_accelerations):.2f} pixels/frame²")
    print(f"Standard deviation: {np.std(x_accelerations):.2f} pixels/frame²")
    print(f"Min x-acceleration: {np.min(x_accelerations):.2f} pixels/frame²")
    print(f"Max x-acceleration: {np.max(x_accelerations):.2f} pixels/frame²")
    print(f"25th percentile: {np.percentile(x_accelerations, 25):.2f} pixels/frame²")
    print(f"75th percentile: {np.percentile(x_accelerations, 75):.2f} pixels/frame²")
    
    # Acceleration analysis
    positive_acc = np.sum(x_accelerations > 0)
    negative_acc = np.sum(x_accelerations < 0)
    zero_acc = np.sum(x_accelerations == 0)
    total_acc = len(x_accelerations)
    
    print(f"\n🚀 ACCELERATION ANALYSIS")
    print("=" * 50)
    print(f"Accelerating (positive): {positive_acc} frames ({positive_acc/total_acc*100:.1f}%)")
    print(f"Decelerating (negative): {negative_acc} frames ({negative_acc/total_acc*100:.1f}%)")
    print(f"Constant speed: {zero_acc} frames ({zero_acc/total_acc*100:.1f}%)")

def main():
    """
    Main function to run the x-velocity analysis.
    """
    csv_file = "tracked_positions.csv"
    window_size = 5  # Average over 5 frames
    
    print("🏃 X-AXIS VELOCITY ANALYSIS")
    print("=" * 50)
    print(f"Analyzing data from: {csv_file}")
    print(f"Averaging window size: {window_size} frames")
    
    # Calculate x-velocities and accelerations
    frames, x_positions, x_velocities, y_positions, velocity_frames, x_accelerations, acceleration_frames = calculate_x_velocity_with_averaging(
        csv_file, window_size
    )
    
    # Print statistics
    print_x_velocity_statistics(x_velocities, x_accelerations)
    
    # Save data
    save_x_velocity_data(frames, x_positions, x_velocities, y_positions, x_accelerations, velocity_frames, acceleration_frames)
    
    # Create plots
    print(f"\n📈 Creating x-velocity and acceleration analysis plots...")
    plot_x_velocity_analysis(frames, x_positions, x_velocities, y_positions, velocity_frames, x_accelerations, acceleration_frames, window_size)
    
    print(f"\n✅ X-velocity and acceleration analysis complete!")
    print(f"📁 Files created:")
    print(f"   - x_velocity_data.csv (x-velocity and acceleration data)")
    print(f"   - x_velocity_analysis.png (comprehensive analysis plots)")

if __name__ == "__main__":
    main()
