import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

def calculate_velocity_with_moving_average(csv_file, window_size=10):
    """
    Calculate swimmer velocity from CSV tracking data with moving average smoothing.
    
    Args:
        csv_file (str): Path to the CSV file with tracking data
        window_size (int): Size of the moving average window (default: 10)
    
    Returns:
        tuple: (frames, velocities, smoothed_velocities)
    """
    # Read CSV data
    frames = []
    x_positions = []
    y_positions = []
    
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            frames.append(int(row['frame']))
            x_positions.append(float(row['x']))
            y_positions.append(float(row['y']))
    
    frames = np.array(frames)
    x_positions = np.array(x_positions)
    y_positions = np.array(y_positions)
    
    # Calculate instantaneous velocity (pixels per frame)
    # Velocity = sqrt((dx/dt)^2 + (dy/dt)^2)
    dx = np.diff(x_positions)
    dy = np.diff(y_positions)
    instantaneous_velocity = np.sqrt(dx**2 + dy**2)
    
    # Apply moving average to smooth the velocity
    # Use uniform_filter1d for better edge handling
    smoothed_velocity = uniform_filter1d(instantaneous_velocity, size=window_size, mode='nearest')
    
    # Adjust frames array to match velocity array (one less element due to diff)
    velocity_frames = frames[1:]
    
    return velocity_frames, instantaneous_velocity, smoothed_velocity

def plot_velocity_analysis(frames, instantaneous_velocity, smoothed_velocity, window_size):
    """
    Plot the velocity analysis results.
    """
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Instantaneous vs Smoothed Velocity
    plt.subplot(2, 2, 1)
    plt.plot(frames, instantaneous_velocity, alpha=0.3, label='Instantaneous', color='lightblue')
    plt.plot(frames, smoothed_velocity, label=f'Moving Average (window={window_size})', color='red', linewidth=2)
    plt.xlabel('Frame Number')
    plt.ylabel('Velocity (pixels/frame)')
    plt.title('Swimmer Velocity Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Velocity distribution
    plt.subplot(2, 2, 2)
    plt.hist(smoothed_velocity, bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Velocity (pixels/frame)')
    plt.ylabel('Frequency')
    plt.title('Velocity Distribution')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Velocity statistics over time (rolling stats)
    plt.subplot(2, 2, 3)
    rolling_mean = uniform_filter1d(smoothed_velocity, size=20, mode='nearest')
    rolling_std = np.sqrt(uniform_filter1d((smoothed_velocity - rolling_mean)**2, size=20, mode='nearest'))
    
    plt.plot(frames, rolling_mean, label='Rolling Mean', color='blue')
    plt.fill_between(frames, rolling_mean - rolling_std, rolling_mean + rolling_std, 
                     alpha=0.3, color='blue', label='±1 Std Dev')
    plt.xlabel('Frame Number')
    plt.ylabel('Velocity (pixels/frame)')
    plt.title('Rolling Statistics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Speed zones
    plt.subplot(2, 2, 4)
    # Define speed zones based on percentiles
    slow_threshold = np.percentile(smoothed_velocity, 33)
    fast_threshold = np.percentile(smoothed_velocity, 67)
    
    slow_mask = smoothed_velocity < slow_threshold
    medium_mask = (smoothed_velocity >= slow_threshold) & (smoothed_velocity < fast_threshold)
    fast_mask = smoothed_velocity >= fast_threshold
    
    plt.scatter(frames[slow_mask], smoothed_velocity[slow_mask], 
                c='green', alpha=0.6, label=f'Slow (< {slow_threshold:.1f})', s=20)
    plt.scatter(frames[medium_mask], smoothed_velocity[medium_mask], 
                c='orange', alpha=0.6, label=f'Medium ({slow_threshold:.1f}-{fast_threshold:.1f})', s=20)
    plt.scatter(frames[fast_mask], smoothed_velocity[fast_mask], 
                c='red', alpha=0.6, label=f'Fast (> {fast_threshold:.1f})', s=20)
    
    plt.xlabel('Frame Number')
    plt.ylabel('Velocity (pixels/frame)')
    plt.title('Speed Zones')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('velocity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_velocity_data(frames, instantaneous_velocity, smoothed_velocity, output_file='velocity_data.csv'):
    """
    Save velocity data to CSV file.
    """
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['frame', 'instantaneous_velocity', 'smoothed_velocity'])
        
        for i in range(len(frames)):
            writer.writerow([frames[i], instantaneous_velocity[i], smoothed_velocity[i]])
    
    print(f"✅ Velocity data saved to {output_file}")

def print_velocity_statistics(smoothed_velocity):
    """
    Print comprehensive velocity statistics.
    """
    print("\n📊 VELOCITY STATISTICS")
    print("=" * 50)
    print(f"Mean velocity: {np.mean(smoothed_velocity):.2f} pixels/frame")
    print(f"Median velocity: {np.median(smoothed_velocity):.2f} pixels/frame")
    print(f"Standard deviation: {np.std(smoothed_velocity):.2f} pixels/frame")
    print(f"Min velocity: {np.min(smoothed_velocity):.2f} pixels/frame")
    print(f"Max velocity: {np.max(smoothed_velocity):.2f} pixels/frame")
    print(f"25th percentile: {np.percentile(smoothed_velocity, 25):.2f} pixels/frame")
    print(f"75th percentile: {np.percentile(smoothed_velocity, 75):.2f} pixels/frame")
    
    # Calculate time spent in different speed zones
    slow_threshold = np.percentile(smoothed_velocity, 33)
    fast_threshold = np.percentile(smoothed_velocity, 67)
    
    slow_frames = np.sum(smoothed_velocity < slow_threshold)
    medium_frames = np.sum((smoothed_velocity >= slow_threshold) & (smoothed_velocity < fast_threshold))
    fast_frames = np.sum(smoothed_velocity >= fast_threshold)
    total_frames = len(smoothed_velocity)
    
    print(f"\n🏊 SPEED ZONE ANALYSIS")
    print("=" * 50)
    print(f"Slow movement: {slow_frames} frames ({slow_frames/total_frames*100:.1f}%)")
    print(f"Medium movement: {medium_frames} frames ({medium_frames/total_frames*100:.1f}%)")
    print(f"Fast movement: {fast_frames} frames ({fast_frames/total_frames*100:.1f}%)")

def main():
    """
    Main function to run the velocity analysis.
    """
    csv_file = "tracked_positions.csv"
    window_size = 10  # Moving average window size
    
    print("🏊 SWIMMER VELOCITY ANALYSIS")
    print("=" * 50)
    print(f"Analyzing data from: {csv_file}")
    print(f"Moving average window size: {window_size} frames")
    
    # Calculate velocities
    frames, instantaneous_velocity, smoothed_velocity = calculate_velocity_with_moving_average(
        csv_file, window_size
    )
    
    # Print statistics
    print_velocity_statistics(smoothed_velocity)
    
    # Save data
    save_velocity_data(frames, instantaneous_velocity, smoothed_velocity)
    
    # Create plots
    print(f"\n📈 Creating velocity analysis plots...")
    plot_velocity_analysis(frames, instantaneous_velocity, smoothed_velocity, window_size)
    
    print(f"\n✅ Analysis complete!")
    print(f"📁 Files created:")
    print(f"   - velocity_data.csv (velocity data)")
    print(f"   - velocity_analysis.png (analysis plots)")

if __name__ == "__main__":
    main()
