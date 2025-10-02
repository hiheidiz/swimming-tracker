import cv2
import csv
import numpy as np

def improved_swimmer_tracking():
    """
    Improved swimmer tracking with better error handling and feedback.
    """
    cap = cv2.VideoCapture("swimming_video.mp4")
    
    if not cap.isOpened():
        print("❌ Error: Could not open video file")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Video Info:")
    print(f"   Total frames: {total_frames}")
    print(f"   FPS: {fps}")
    print(f"   Resolution: {width}x{height}")
    print(f"   Duration: {total_frames/fps:.1f} seconds")
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Could not read first frame")
        cap.release()
        return
    
    print(f"\n🎯 Please select the swimmer to track:")
    print(f"   - Click and drag to draw a rectangle around the swimmer")
    print(f"   - Press SPACE or ENTER to confirm")
    print(f"   - Press 'c' to cancel")
    
    bbox = cv2.selectROI("Select Swimmer to Track", frame, fromCenter=False, showCrosshair=True)
    
    if bbox[2] == 0 or bbox[3] == 0:  # width or height is 0
        print("❌ No valid region selected. Exiting.")
        cap.release()
        return
    
    print(f"✅ Selected region: {bbox}")
    
    # Create tracker with fallback options
    tracker = None
    tracker_name = "CSRT"
    
    try:
        tracker = cv2.TrackerCSRT_create()
        print(f"✅ Using {tracker_name} tracker")
    except:
        try:
            tracker = cv2.TrackerKCF_create()
            tracker_name = "KCF"
            print(f"✅ Using {tracker_name} tracker (CSRT not available)")
        except:
            print("❌ No suitable tracker available")
            cap.release()
            return
    
    # Initialize tracker
    success = tracker.init(frame, bbox)
    if not success:
        print("❌ Failed to initialize tracker")
        cap.release()
        return
    
    # Setup VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("tracked_swimmer_improved.mp4", fourcc, fps, (width, height))
    
    # Tracking variables
    positions = []
    frame_number = 0
    tracking_failures = 0
    consecutive_failures = 0
    max_consecutive_failures = 10  # Stop if tracking fails for 10 consecutive frames
    
    print(f"\n🏊 Starting tracking...")
    print(f"   Press 'q' to quit early")
    print(f"   Press 'r' to reinitialize tracker at current position")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"\n📹 Reached end of video (frame {frame_number}/{total_frames})")
            break
        
        # Update tracker
        success, box = tracker.update(frame)
        
        if success:
            x, y, w, h = [int(v) for v in box]
            cx, cy = x + w // 2, y + h // 2
            
            # Draw tracking info
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
            cv2.putText(frame, f"Frame: {frame_number}/{total_frames}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Tracking: {tracker_name}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            positions.append((frame_number, cx, cy))
            consecutive_failures = 0
            
        else:
            # Tracking failed
            tracking_failures += 1
            consecutive_failures += 1
            
            cv2.putText(frame, f"Frame: {frame_number}/{total_frames}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "TRACKING LOST!", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Failures: {consecutive_failures}/{max_consecutive_failures}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            positions.append((frame_number, None, None))
            
            # Stop if too many consecutive failures
            if consecutive_failures >= max_consecutive_failures:
                print(f"\n⚠️  Tracking lost for {consecutive_failures} consecutive frames. Stopping.")
                break
        
        # Write frame to output video
        out.write(frame)
        
        # Show frame
        cv2.imshow("Swimmer Tracking - Improved", frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(f"\n⏹️  User stopped tracking at frame {frame_number}")
            break
        elif key == ord('r') and success:
            # Reinitialize tracker at current position
            new_bbox = (x, y, w, h)
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, new_bbox)
            print(f"🔄 Tracker reinitialized at frame {frame_number}")
        
        frame_number += 1
        
        # Progress update every 50 frames
        if frame_number % 50 == 0:
            progress = (frame_number / total_frames) * 100
            print(f"📊 Progress: {frame_number}/{total_frames} frames ({progress:.1f}%)")
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Save results
    with open("tracked_positions_improved.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame", "x", "y"])
        writer.writerows(positions)
    
    # Print summary
    successful_tracks = sum(1 for pos in positions if pos[1] is not None)
    total_tracks = len(positions)
    success_rate = (successful_tracks / total_tracks) * 100 if total_tracks > 0 else 0
    
    print(f"\n📊 TRACKING SUMMARY")
    print(f"=" * 50)
    print(f"Total frames processed: {frame_number}")
    print(f"Successful tracks: {successful_tracks}")
    print(f"Failed tracks: {tracking_failures}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"✅ Video saved as: tracked_swimmer_improved.mp4")
    print(f"✅ Positions saved as: tracked_positions_improved.csv")
    
    if success_rate < 50:
        print(f"\n⚠️  Low success rate! Consider:")
        print(f"   - Selecting a more distinctive region")
        print(f"   - Using a different tracker")
        print(f"   - Checking video quality")

if __name__ == "__main__":
    improved_swimmer_tracking()
