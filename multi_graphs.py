import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- SETTINGS ---
csv_path = "tracking_log_2.csv"
output_folder = "object_graphs_filtered_2"
fps = 30  # adjust to match your video FPS
velocity_threshold_std = 3.0  # number of std deviations for outlier detection

os.makedirs(output_folder, exist_ok=True)

# --- READ DATA ---
df = pd.read_csv(csv_path)
df["Time"] = df["Frame"] / fps

# --- FUNCTION: Remove Velocity Outliers ---
def remove_velocity_outliers(series):
    # Compute z-scores and replace large deviations with NaN
    median = np.nanmedian(series)
    mad = np.nanmedian(np.abs(series - median))  # median absolute deviation
    if mad == 0:
        return series  # no variation
    z = 0.6745 * (series - median) / mad
    cleaned = series.copy()
    cleaned[np.abs(z) > velocity_threshold_std] = np.nan
    return cleaned

# --- UNIQUE OBJECTS ---
object_ids = df["ObjectID"].unique()
print(f"Found {len(object_ids)} objects: {object_ids}")

# --- PLOT PER OBJECT ---
for obj_id in object_ids:
    data = df[df["ObjectID"] == obj_id].sort_values("Frame").copy()
    time = data["Time"]

    # Clean velocity data
    data["Vx_clean"] = remove_velocity_outliers(data["Vx"])
    data["Vy_clean"] = remove_velocity_outliers(data["Vy"])

    # Recompute acceleration based on cleaned velocity (optional)
    data["Ax_clean"] = data["Vx_clean"].diff() * fps
    data["Ay_clean"] = data["Vy_clean"].diff() * fps

    # Insert NaNs into X/Y when corresponding Vx/Vy are invalid
    data.loc[data["Vx_clean"].isna(), ["X", "Ax_clean"]] = np.nan
    data.loc[data["Vy_clean"].isna(), ["Y", "Ay_clean"]] = np.nan

    # --- Create Figure ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Object {obj_id} Tracking (Outliers Removed)", fontsize=16, weight='bold')

    # --- X plots ---
    axes[0, 0].plot(time, data["X"], label='X Position', color='blue')
    axes[0, 0].set_title("X Position vs Time")

    axes[0, 1].plot(time, data["Vx_clean"], label='X Velocity', color='orange')
    axes[0, 1].set_title("X Velocity vs Time")

    axes[0, 2].plot(time, data["Ax_clean"], label='X Acceleration', color='red')
    axes[0, 2].set_title("X Acceleration vs Time")

    # --- Y plots ---
    axes[1, 0].plot(time, data["Y"], label='Y Position', color='green')
    axes[1, 0].set_title("Y Position vs Time")

    axes[1, 1].plot(time, data["Vy_clean"], label='Y Velocity', color='purple')
    axes[1, 1].set_title("Y Velocity vs Time")

    axes[1, 2].plot(time, data["Ay_clean"], label='Y Acceleration', color='brown')
    axes[1, 2].set_title("Y Acceleration vs Time")

    # Format plots
    for ax in axes.flatten():
        ax.set_xlabel("Time (s)")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- Save ---
    save_path = os.path.join(output_folder, f"object_{obj_id}_graphs_filtered.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"✅ Saved filtered graphs for Object {obj_id} → {save_path}")
