import cv2
import csv
import numpy as np

def robust_swimmer_tracking():
    """
    Robust swimmer tracking with multiple tracker fallbacks and better error handling.
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
    
    print(f"✅ First frame read successfully")
    print(f"   Frame shape: {frame.shape}")
    
    # Let user select ROI
    print(f"\n🎯 Please select the swimmer to track:")
    print(f"   - Click and drag to draw a rectangle around the swimmer")
    print(f"   - Press SPACE or ENTER to confirm")
    print(f"   - Press 'c' to cancel")
    
    bbox = cv2.selectROI("Select Swimmer to Track", frame, fromCenter=False, showCrosshair=True)
    
    if bbox[2] == 0 or bbox[3] == 0:
        print("❌ No valid region selected. Exiting.")
        cap.release()
        return
    
    print(f"✅ Selected region: {bbox}")
    
    # Try different trackers in order of preference
    tracker_options = [
        ("CSRT", lambda: cv2.TrackerCSRT_create()),
        ("KCF", lambda: cv2.TrackerKCF_create()),
        ("MOSSE", lambda: cv2.TrackerMOSSE_create()),
        ("MIL", lambda: cv2.TrackerMIL_create()),
    ]
    
    tracker = None
    tracker_name = None
    
    for name, tracker_func in tracker_options:
        try:
            print(f"🔄 Trying {name} tracker...")
            tracker = tracker_func()
            
            # Test initialization
            success = tracker.init(frame, bbox)
            if success:
                tracker_name = name
                print(f"✅ Successfully initialized {name} tracker")
                break
            else:
                print(f"❌ {name} tracker failed to initialize")
                tracker = None
                
        except Exception as e:
            print(f"❌ {name} tracker not available: {e}")
            tracker = None
    
    if tracker is None:
        print("❌ No suitable tracker could be initialized")
        cap.release()
        return
    
    # Setup VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("tracked_swimmer_robust.mp4", fourcc, fps, (width, height))
    
    # Tracking variables
    positions = []
    frame_number = 0
    tracking_failures = 0
    consecutive_failures = 0
    max_consecutive_failures = 15  # Allow more failures before stopping
    
    print(f"\n🏊 Starting tracking with {tracker_name} tracker...")
    print(f"   Press 'q' to quit early")
    print(f"   Press 'r' to reinitialize tracker")
    print(f"   Press 's' to save current frame")
    
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
            
            # Validate tracking result
            if w > 0 and h > 0 and x >= 0 and y >= 0 and x + w <= width and y + h <= height:
                # Draw tracking info
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
                cv2.putText(frame, f"Frame: {frame_number}/{total_frames}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Tracking: {tracker_name}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Pos: ({cx}, {cy})", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                positions.append((frame_number, cx, cy))
                consecutive_failures = 0
            else:
                # Invalid tracking result
                success = False
                print(f"⚠️  Frame {frame_number}: Invalid tracking result - Box: {x, y, w, h}")
        
        if not success:
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
        cv2.imshow("Robust Swimmer Tracking", frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(f"\n⏹️  User stopped tracking at frame {frame_number}")
            break
        elif key == ord('r') and success:
            # Reinitialize tracker at current position
            new_bbox = (x, y, w, h)
            try:
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, new_bbox)
                print(f"🔄 Tracker reinitialized at frame {frame_number}")
            except:
                print(f"❌ Failed to reinitialize tracker")
        elif key == ord('s'):
            cv2.imwrite(f"debug_frame_{frame_number}.jpg", frame)
            print(f"💾 Saved debug frame {frame_number}")
        
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
    with open("tracked_positions_robust.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame", "x", "y"])
        writer.writerows(positions)
    
    # Print summary
    successful_tracks = sum(1 for pos in positions if pos[1] is not None)
    total_tracks = len(positions)
    success_rate = (successful_tracks / total_tracks) * 100 if total_tracks > 0 else 0
    
    print(f"\n📊 TRACKING SUMMARY")
    print(f"=" * 50)
    print(f"Tracker used: {tracker_name}")
    print(f"Total frames processed: {frame_number}")
    print(f"Successful tracks: {successful_tracks}")
    print(f"Failed tracks: {tracking_failures}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"✅ Video saved as: tracked_swimmer_robust.mp4")
    print(f"✅ Positions saved as: tracked_positions_robust.csv")
    
    if success_rate < 50:
        print(f"\n⚠️  Low success rate! Consider:")
        print(f"   - Selecting a more distinctive region")
        print(f"   - Using a different tracker")
        print(f"   - Checking video quality")
        print(f"   - The swimmer might be moving too fast or changing appearance")

if __name__ == "__main__":
    robust_swimmer_tracking()
