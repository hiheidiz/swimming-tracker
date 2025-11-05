import sys
import os
import cv2
import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# --------------------------
# Helper smoothing functions
# --------------------------
def smooth_array(arr):
    """Apply 1/4 prev + 1/2 current + 1/4 next smoothing elementwise.
    Keeps endpoints unchanged."""
    arr = np.asarray(arr, dtype=float)
    if arr.size < 3:
        return arr.copy()
    out = arr.copy()
    out[1:-1] = 0.25 * arr[:-2] + 0.5 * arr[1:-1] + 0.25 * arr[2:]
    return out

# --------------------------
# Main application
# --------------------------
class PointsProcessor(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Points Processor — View Tracked Points")
        self.setGeometry(80, 80, 1200, 800)

        # State
        self.cap = None
        self.video_path = None
        self.csv_path = None
        self.fps = 30.0
        self.frame_idx = 0
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.playing = False

        # Data from CSV
        self.df_raw = None
        self.df_proc = None
        self.points_by_frame = {}  # frame -> list of (object_id, x, y)
        self.object_names = {}  # mapping from object_id to custom name

        # Overlay graph state
        self.overlay_graph_type = "X_smooth"  # Current graph type to display
        self.overlay_visible_objects = set()  # Set of object IDs to show in overlay

        # UI
        self.init_ui()

    def init_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # Top control row
        row = QtWidgets.QHBoxLayout()
        self.load_video_btn = QtWidgets.QPushButton("Load Video")
        self.load_video_btn.clicked.connect(self.load_video)
        row.addWidget(self.load_video_btn)

        self.load_csv_btn = QtWidgets.QPushButton("Load CSV")
        self.load_csv_btn.clicked.connect(self.load_csv)
        row.addWidget(self.load_csv_btn)

        self.play_btn = QtWidgets.QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        row.addWidget(self.play_btn)

        # Overlay graph controls
        row.addWidget(QtWidgets.QLabel("Overlay Graph:"))
        self.overlay_graph_combo = QtWidgets.QComboBox()
        self.overlay_graph_combo.addItems(["Position", "Velocity", "Acceleration"])
        self.overlay_graph_combo.currentIndexChanged.connect(self.on_overlay_graph_changed)
        row.addWidget(self.overlay_graph_combo)

        # Object checkboxes container
        self.object_checkboxes_widget = QtWidgets.QWidget()
        self.object_checkboxes_layout = QtWidgets.QHBoxLayout(self.object_checkboxes_widget)
        self.object_checkboxes_layout.setContentsMargins(0, 0, 0, 0)
        self.object_checkboxes_layout.addWidget(QtWidgets.QLabel("Objects:"))
        self.object_checkboxes = {}  # object_id -> checkbox
        row.addWidget(self.object_checkboxes_widget)

        self.status_label = QtWidgets.QLabel("No video or CSV loaded")
        row.addWidget(self.status_label)
        row.addStretch()
        layout.addLayout(row)

        # Central content: video on left, plots on right
        content_row = QtWidgets.QHBoxLayout()

        # Video display (left) with overlay graph
        video_widget = QtWidgets.QWidget()
        video_widget.setFixedSize(960, 540)
        video_layout_widget = QtWidgets.QVBoxLayout(video_widget)
        video_layout_widget.setContentsMargins(0, 0, 0, 0)

        self.video_label = QtWidgets.QLabel()
        self.video_label.setFixedSize(960, 540)  # approximate; will scale frames into this label
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        video_layout_widget.addWidget(self.video_label)

        # Overlay graph on top of video
        self.overlay_fig = Figure(figsize=(9.6, 5.4))
        self.overlay_fig.patch.set_facecolor('none')
        self.overlay_fig.patch.set_alpha(0.0)
        self.overlay_ax = self.overlay_fig.add_subplot(111)
        self.overlay_ax.set_facecolor('none')
        self.overlay_ax.patch.set_alpha(0.0)
        self.overlay_ax.spines['top'].set_visible(False)
        self.overlay_ax.spines['right'].set_visible(False)
        self.overlay_ax.spines['bottom'].set_color('white')
        self.overlay_ax.spines['left'].set_color('white')
        self.overlay_ax.tick_params(colors='white', labelsize=8)
        self.overlay_ax.xaxis.label.set_color('white')
        self.overlay_ax.yaxis.label.set_color('white')
        self.overlay_ax.grid(True, linestyle='--', alpha=0.3, color='white')
        self.overlay_fig.tight_layout(pad=0.5)

        self.overlay_canvas = FigureCanvas(self.overlay_fig)
        self.overlay_canvas.setParent(video_widget)
        self.overlay_canvas.setFixedSize(960, 540)
        self.overlay_canvas.setStyleSheet("background-color: transparent;")
        self.overlay_canvas.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.overlay_canvas.raise_()

        # Initially hide overlay (will show when CSV is loaded)
        self.overlay_canvas.setVisible(False)

        content_row.addWidget(video_widget, stretch=2, alignment=QtCore.Qt.AlignLeft)

        # Plots panel (right)
        plots_panel = QtWidgets.QWidget()
        plots_layout = QtWidgets.QVBoxLayout(plots_panel)
        plots_layout.setContentsMargins(4, 0, 0, 0)
        plots_layout.setSpacing(8)

        # Embedded matplotlib canvases for three plots
        self.plot_canvases = {}
        self.plot_axes = {}
        self.plot_figs = {}

        for key, title in [("X_smooth", "Smoothed X Position (pixels)"),
                           ("Velocity_smooth", "Smoothed Velocity (pixels/frame)"),
                           ("Acceleration_smooth", "Smoothed Acceleration (pixels/frame^2)")]:
            fig = Figure(figsize=(5, 1.6))
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            ax.set_title(title, fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.4)
            self.plot_canvases[key] = canvas
            self.plot_axes[key] = ax
            self.plot_figs[key] = fig
            plots_layout.addWidget(canvas)

        content_row.addWidget(plots_panel, stretch=1)
        layout.addLayout(content_row)

        # Bottom row: frame slider and frame label
        bottom = QtWidgets.QHBoxLayout()
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.sliderReleased.connect(self.slider_seek)
        bottom.addWidget(self.frame_slider)
        self.frame_display = QtWidgets.QLabel("Frame: 0")
        bottom.addWidget(self.frame_display)
        layout.addLayout(bottom)

        # Track click handler ids for embedded figures
        self.figures = {}
        # Keep references to detailed windows so they are not garbage collected
        self.detail_windows = []

    # --------------------------
    # Video control methods
    # --------------------------
    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select video file", "", "Video Files (*.mp4 *.mov *.avi *.mkv)")
        if not path:
            return
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", f"Could not open video: {path}")
            return
        self.video_path = path
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_idx = 0

        # Setup slider
        self.frame_slider.setEnabled(True)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(self.frame_count-1)
        self.frame_slider.setValue(0)
        self.frame_display.setText(f"Frame: {self.frame_idx} / {self.frame_count-1}")

        # Update status
        if self.csv_path:
            self.status_label.setText(f"Video: {os.path.basename(path)} | CSV: {os.path.basename(self.csv_path)}")
        else:
            self.status_label.setText(f"Video: {os.path.basename(path)}")

        # Update overlay graph if CSV is loaded
        if self.df_proc is not None:
            self.update_overlay_graph()

        self.show_frame_at(0)

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV file", "", "CSV Files (*.csv)")
        if not path:
            return

        try:
            df = pd.read_csv(path)

            # Check required columns
            required_cols = ["Frame", "ObjectID", "X", "Y"]
            if not all(col in df.columns for col in required_cols):
                QMessageBox.critical(self, "Error",
                    f"CSV must contain columns: {', '.join(required_cols)}\n"
                    f"Found columns: {', '.join(df.columns)}")
                return

            self.csv_path = path
            self.df_raw = df[required_cols].copy()

            # Check if CSV already has processed columns
            has_smooth = "X_smooth" in df.columns and "Y_smooth" in df.columns
            has_velocity = "Velocity_smooth" in df.columns
            has_accel = "Acceleration_smooth" in df.columns

            if has_smooth and has_velocity and has_accel:
                # Already processed, use as-is
                self.df_proc = df.copy()
                print("Using pre-processed CSV")
            else:
                # Need to process
                print("Processing raw CSV...")
                self.process_csv()

            # Build points_by_frame dictionary for quick lookup
            self.points_by_frame = {}
            for _, row in self.df_raw.iterrows():
                frame = int(row["Frame"])
                oid = int(row["ObjectID"])
                x = float(row["X"])
                y = float(row["Y"])
                if frame not in self.points_by_frame:
                    self.points_by_frame[frame] = []
                self.points_by_frame[frame].append((oid, (x, y)))

            # Extract object names if available (from a Name column or use defaults)
            if "Name" in df.columns:
                for _, row in df.iterrows():
                    oid = int(row["ObjectID"])
                    name = str(row["Name"]).strip()
                    if name:
                        self.object_names[oid] = name
            else:
                # Use default names
                unique_oids = self.df_raw["ObjectID"].unique()
                for oid in unique_oids:
                    if int(oid) not in self.object_names:
                        self.object_names[int(oid)] = f"Obj {int(oid)}"

            # Create checkboxes for objects
            self.create_object_checkboxes()

            # Show and initialize overlay graph
            self.overlay_canvas.setVisible(True)
            self.update_overlay_graph()

            # Update status
            if self.video_path:
                self.status_label.setText(f"Video: {os.path.basename(self.video_path)} | CSV: {os.path.basename(path)}")
            else:
                self.status_label.setText(f"CSV: {os.path.basename(path)}")

            # Update plots
            if self.df_proc is not None:
                self.update_plots()

            # Redraw current frame if video is loaded
            if self.cap:
                self.show_frame_at(self.frame_idx)

            QMessageBox.information(self, "Success", f"Loaded CSV with {len(self.df_raw)} data points")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load CSV: {str(e)}")
            import traceback
            traceback.print_exc()

    def process_csv(self):
        """Process raw CSV to compute smoothed values and derivatives."""
        if self.df_raw is None:
            return

        # Group by object, compute smoothed X/Y using the 1/4-1/2-1/4 filter
        smoothed_rows = []
        for oid, group in self.df_raw.groupby("ObjectID"):
            g = group.sort_values("Frame").reset_index(drop=True)
            xs = g["X"].to_numpy()
            ys = g["Y"].to_numpy()
            xs_s = smooth_array(xs)
            ys_s = smooth_array(ys)
            g["X_smooth"] = xs_s
            g["Y_smooth"] = ys_s
            smoothed_rows.append(g)

        df_smoothed = pd.concat(smoothed_rows)

        # Compute velocity from smoothed X and then smooth velocity with same filter
        proc_groups = []
        for oid, group in df_smoothed.groupby("ObjectID"):
            g = group.sort_values("Frame").reset_index(drop=True)
            x_s = g["X_smooth"].to_numpy()
            # velocity in pixels per frame
            if len(x_s) < 2:
                vel = np.zeros_like(x_s)
            else:
                vel = np.gradient(x_s)
            vel_s = smooth_array(vel)
            if len(vel_s) < 2:
                accel = np.zeros_like(vel_s)
            else:
                accel = np.gradient(vel_s)
            accel_s = smooth_array(accel)

            g["Velocity_smooth"] = vel_s
            g["Acceleration_smooth"] = accel_s
            proc_groups.append(g)

        self.df_proc = pd.concat(proc_groups)

    def show_frame_at(self, frame_no):
        if not self.cap:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
        ret, frame = self.cap.read()
        if not ret:
            return
        self.frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

        # Display frame without drawing points
        self.display_frame(frame)
        self.frame_slider.setValue(self.frame_idx)
        self.frame_display.setText(f"Frame: {self.frame_idx} / {self.frame_count-1}")

        # Update overlay graph
        self.update_overlay_graph()

    def display_frame(self, frame, overlay_mask=None, pnts=None):
        # convert BGR to RGB and display
        disp = frame.copy()
        if overlay_mask is not None:
            disp = cv2.add(disp, overlay_mask)
        # Note: points are no longer drawn on the video display
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        # scale to label size keeping aspect
        label_w = self.video_label.width()
        label_h = self.video_label.height()
        scale = min(label_w / w, label_h / h)
        new_w, new_h = int(w*scale), int(h*scale)
        rgb = cv2.resize(rgb, (new_w, new_h))
        qimg = QtGui.QImage(rgb.data, new_w, new_h, 3*new_w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pix)

    def toggle_play(self):
        if not self.cap:
            return
        if not self.playing:
            # start playing
            self.playing = True
            self.play_btn.setText("Pause")
            interval_ms = int(1000 / (self.fps if self.fps>0 else 30))
            self.timer.start(interval_ms)
        else:
            self.playing = False
            self.play_btn.setText("Play")
            self.timer.stop()

    def next_frame(self):
        if not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.playing = False
            self.play_btn.setText("Play")
            return

        # Display frame without drawing points
        self.display_frame(frame)

        # advance internal counters
        self.frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        self.frame_slider.setValue(self.frame_idx)
        self.frame_display.setText(f"Frame: {self.frame_idx} / {self.frame_count-1}")

        # Update overlay graph
        self.update_overlay_graph()

    # --------------------------
    # Overlay graph methods
    # --------------------------
    def create_object_checkboxes(self):
        """Create checkboxes for each object."""
        # Clear existing checkboxes
        for cb in self.object_checkboxes.values():
            cb.setParent(None)
            cb.deleteLater()
        self.object_checkboxes.clear()

        if self.df_raw is None:
            return

        # Get unique object IDs
        unique_oids = sorted(self.df_raw["ObjectID"].unique())

        for oid in unique_oids:
            oid_int = int(oid)
            obj_name = self.object_names.get(oid_int, f"Obj {oid_int}")
            cb = QtWidgets.QCheckBox(obj_name)
            cb.setChecked(True)  # All checked by default
            self.overlay_visible_objects.add(oid_int)

            def make_handler(target_oid):
                def handler(state):
                    if state == QtCore.Qt.Checked:
                        self.overlay_visible_objects.add(target_oid)
                    else:
                        self.overlay_visible_objects.discard(target_oid)
                    self.update_overlay_graph()
                return handler

            cb.stateChanged.connect(make_handler(oid_int))
            self.object_checkboxes[oid_int] = cb
            self.object_checkboxes_layout.addWidget(cb)

        self.object_checkboxes_layout.addStretch()

    def on_overlay_graph_changed(self, index):
        """Handle overlay graph type dropdown change."""
        graph_types = {
            0: "X_smooth",
            1: "Velocity_smooth",
            2: "Acceleration_smooth"
        }
        self.overlay_graph_type = graph_types.get(index, "X_smooth")
        self.update_overlay_graph()

    def update_overlay_graph(self):
        """Update the overlay graph with data up to current frame."""
        if self.df_proc is None or self.frame_idx < 0:
            self.overlay_ax.clear()
            self.overlay_canvas.draw()
            return

        # Get total frame count from video (or max frame from data if video not loaded)
        if hasattr(self, 'frame_count') and self.frame_count > 0:
            max_frame = self.frame_count - 1
        else:
            # Use max frame from data if video not loaded
            max_frame = int(self.df_proc["Frame"].max())

        # Get data up to current frame
        df_filtered = self.df_proc[self.df_proc["Frame"] <= self.frame_idx].copy()

        # Clear previous plot
        self.overlay_ax.clear()

        # Get column name and label
        col_map = {
            "X_smooth": ("X_smooth", "Smoothed X Position (pixels)"),
            "Velocity_smooth": ("Velocity_smooth", "Smoothed Velocity (pixels/frame)"),
            "Acceleration_smooth": ("Acceleration_smooth", "Smoothed Acceleration (pixels/frame^2)")
        }

        col_name, ylabel = col_map.get(self.overlay_graph_type, ("X_smooth", "Smoothed X Position (pixels)"))

        # Calculate y-axis range from all data (for visible objects only)
        y_min = float('inf')
        y_max = float('-inf')
        for oid, group in self.df_proc.groupby("ObjectID"):
            oid_int = int(oid)
            if oid_int in self.overlay_visible_objects:
                y_vals = group[col_name].values
                if len(y_vals) > 0:
                    y_min = min(y_min, y_vals.min())
                    y_max = max(y_max, y_vals.max())

        # If no visible objects or no data, use default range
        if y_min == float('inf') or y_max == float('-inf'):
            y_min = 0
            y_max = 100
        else:
            # Add margin (5% above and below)
            y_range = y_max - y_min
            if y_range == 0:
                y_range = abs(y_max) if y_max != 0 else 100
            margin = y_range * 0.05
            y_min = y_min - margin
            y_max = y_max + margin

        # Plot data for visible objects only (only up to current frame)
        for oid, group in df_filtered.groupby("ObjectID"):
            oid_int = int(oid)
            if oid_int in self.overlay_visible_objects:
                obj_name = self.object_names.get(oid_int, f"Obj {oid_int}")
                self.overlay_ax.plot(group["Frame"], group[col_name],
                                   linestyle='-', marker='.', markersize=4,
                                   label=obj_name, linewidth=2)

        # Set x-axis limits to show full frame range (fixed)
        self.overlay_ax.set_xlim(0, max_frame)

        # Set y-axis limits to show full data range with margin (fixed)
        self.overlay_ax.set_ylim(y_min, y_max)

        # Set labels and styling
        self.overlay_ax.set_xlabel("Frame", color='white', fontsize=9)
        self.overlay_ax.set_ylabel(ylabel, color='white', fontsize=9)
        self.overlay_ax.spines['top'].set_visible(False)
        self.overlay_ax.spines['right'].set_visible(False)
        self.overlay_ax.spines['bottom'].set_color('white')
        self.overlay_ax.spines['left'].set_color('white')
        self.overlay_ax.tick_params(colors='white', labelsize=8)
        self.overlay_ax.grid(True, linestyle='--', alpha=0.3, color='white')

        # Add legend if there are visible objects
        if self.overlay_visible_objects:
            self.overlay_ax.legend(loc="upper right", fontsize='small',
                                 facecolor='black', edgecolor='white',
                                 labelcolor='white', framealpha=0.7)

        self.overlay_fig.tight_layout(pad=0.5)
        self.overlay_canvas.draw()

    # --------------------------
    # Slider seek
    # --------------------------
    def slider_seek(self):
        if not self.cap:
            return
        val = self.frame_slider.value()
        self.show_frame_at(val)
        # stop playback
        if self.playing:
            self.toggle_play()

    # --------------------------
    # Plotting + click-to-seek
    # --------------------------
    def update_plots(self):
        """Update all embedded plots with processed data."""
        if self.df_proc is None:
            return

        plots = [
            ("X_smooth", "Smoothed X Position (pixels)", "position_overlay.png"),
            ("Velocity_smooth", "Smoothed Velocity (pixels/frame)", "velocity_overlay.png"),
            ("Acceleration_smooth", "Smoothed Acceleration (pixels/frame^2)", "accel_overlay.png"),
        ]

        for col, ylabel, fname in plots:
            if col not in self.plot_axes:
                continue
            ax = self.plot_axes[col]
            fig = self.plot_figs[col]
            canvas = self.plot_canvases[col]

            ax.clear()
            for oid, group in self.df_proc.groupby("ObjectID"):
                # Use custom name if available, otherwise default to Obj {oid}
                obj_name = self.object_names.get(int(oid), f"Obj {oid}")
                ax.plot(group["Frame"], group[col], marker='.', linestyle='-', label=obj_name, markersize=3)
            ax.set_xlabel("Frame")
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel + " vs Frame")
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend(loc="upper right", fontsize='x-small')
            fig.tight_layout()

            # Save PNGs
            fig.savefig(fname, dpi=200)
            print(f"Saved figure to {fname}")

            # Rebind click event to seek video to nearest frame for this canvas
            if fname in self.figures:
                old_fig, old_cid = self.figures[fname]
                try:
                    old_fig.canvas.mpl_disconnect(old_cid)
                except Exception:
                    pass

            def on_click(event, local_df=self.df_proc.copy(), ax=ax, col_name=col, y_label=ylabel):
                if event.inaxes != ax:
                    return
                # Double-click opens detailed interactive window
                if getattr(event, 'dblclick', False):
                    self.open_detailed_plot(local_df, col_name, y_label)
                    return
                click_x = event.xdata
                nearest_frame = None
                nearest_dist = float('inf')
                if click_x is None:
                    return
                for _, g in local_df.groupby("ObjectID"):
                    frames = g["Frame"].to_numpy()
                    if frames.size == 0:
                        continue
                    idx = np.argmin(np.abs(frames - click_x))
                    dist = abs(frames[idx] - click_x)
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest_frame = int(frames[idx])
                if nearest_frame is not None:
                    print(f"Graph clicked: seeking to frame {nearest_frame}")
                    self.show_frame_at(nearest_frame)

            cid = canvas.mpl_connect('button_press_event', on_click)
            self.figures[fname] = (fig, cid)

            canvas.draw()

    def open_detailed_plot(self, df, column_name, y_label):
        # Create a dialog containing a matplotlib canvas with toolbar and checkboxes to toggle objects
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"{y_label} — Detailed View")
        dlg.resize(1000, 600)

        main_layout = QtWidgets.QHBoxLayout(dlg)

        # Left: plot + toolbar
        left = QtWidgets.QVBoxLayout()
        fig = Figure(figsize=(6, 4))
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, dlg)
        ax = fig.add_subplot(111)
        ax.set_xlabel("Frame")
        ax.set_ylabel(y_label)
        ax.set_title(y_label + " vs Frame")
        ax.grid(True, linestyle='--', alpha=0.4)

        # Plot all object lines initially
        oid_to_line = {}
        for oid, group in df.groupby("ObjectID"):
            # Use custom name if available, otherwise default to Obj {oid}
            obj_name = self.object_names.get(int(oid), f"Obj {oid}")
            line, = ax.plot(group["Frame"], group[column_name], linestyle='-', marker='.', markersize=3, label=obj_name)
            oid_to_line[int(oid)] = line
        ax.legend(loc="upper right", fontsize='x-small')
        fig.tight_layout()
        left.addWidget(toolbar)
        left.addWidget(canvas)

        # Right: object toggle checkboxes (scrollable)
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        checkbox_widgets = []
        for oid in sorted(oid_to_line.keys()):
            # Use custom name if available, otherwise default to Obj {oid}
            obj_name = self.object_names.get(oid, f"Obj {oid}")
            cb = QtWidgets.QCheckBox(obj_name)
            cb.setChecked(True)
            def make_handler(target_oid):
                def handler(state):
                    line = oid_to_line.get(target_oid)
                    if line is None:
                        return
                    line.set_visible(state == QtCore.Qt.Checked)
                    canvas.draw_idle()
                return handler
            cb.stateChanged.connect(make_handler(oid))
            right_layout.addWidget(cb)
            checkbox_widgets.append(cb)
        right_layout.addStretch()

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(right_panel)

        main_layout.addLayout(left, stretch=4)
        main_layout.addWidget(scroll, stretch=1)

        # Keep reference and show
        self.detail_windows.append(dlg)
        canvas.draw()

        # Click-to-seek in detailed window as well
        def on_click(event, local_df=df.copy(), local_ax=ax):
            if event.inaxes != local_ax:
                return
            # Ignore double-click here; double-click is already used to open this window
            click_x = event.xdata
            if click_x is None:
                return
            nearest_frame = None
            nearest_dist = float('inf')
            for _, g in local_df.groupby("ObjectID"):
                frames = g["Frame"].to_numpy()
                if frames.size == 0:
                    continue
                idx = np.argmin(np.abs(frames - click_x))
                dist = abs(frames[idx] - click_x)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_frame = int(frames[idx])
            if nearest_frame is not None:
                print(f"Detailed graph clicked: seeking to frame {nearest_frame}")
                self.show_frame_at(nearest_frame)

        canvas.mpl_connect('button_press_event', on_click)
        dlg.show()

# --------------------------
# Run the app
# --------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    mainw = PointsProcessor()
    mainw.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
