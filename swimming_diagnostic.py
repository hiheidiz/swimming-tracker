import cv2
import csv

def diagnose_tracking_issue():
    """
    Diagnostic version to understand why tracking stops early.
    """
    cap = cv2.VideoCapture("swimming_video.mp4")
    
    if not cap.isOpened():
        print("❌ Error: Could not open video file")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📹 Video has {total_frames} frames at {fps} FPS")
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Could not read first frame")
        cap.release()
        return
    
    print(f"✅ First frame read successfully")
    print(f"   Frame shape: {frame.shape}")
    
    # Let user select ROI
    print(f"\n🎯 Select the swimmer to track...")
    bbox = cv2.selectROI("Select Swimmer", frame, fromCenter=False, showCrosshair=True)
    
    if bbox[2] == 0 or bbox[3] == 0:
        print("❌ No valid region selected")
        cap.release()
        return
    
    print(f"✅ Selected bbox: {bbox}")
    
    # Create tracker
    tracker = cv2.TrackerCSRT_create()
    success = tracker.init(frame, bbox)
    
    if not success:
        print("❌ Failed to initialize tracker")
        cap.release()
        return
    
    print(f"✅ Tracker initialized successfully")
    
    # Track with detailed logging
    frame_number = 0
    positions = []
    failure_count = 0
    
    print(f"\n🏊 Starting detailed tracking...")
    print(f"   Press 'q' to quit")
    print(f"   Press 's' to save current frame")
    
    while frame_number < 50:  # Only track first 50 frames for diagnosis
        ret, frame = cap.read()
        if not ret:
            print(f"❌ Could not read frame {frame_number}")
            break
        
        success, box = tracker.update(frame)
        
        if success:
            x, y, w, h = [int(v) for v in box]
            cx, cy = x + w // 2, y + h // 2
            
            # Draw tracking
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            
            # Add text info
            cv2.putText(frame, f"Frame: {frame_number}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Pos: ({cx}, {cy})", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Box: {bbox}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            positions.append((frame_number, cx, cy))
            print(f"✅ Frame {frame_number}: Success - Pos({cx}, {cy}), Box{x, y, w, h}")
            
        else:
            failure_count += 1
            cv2.putText(frame, f"Frame: {frame_number}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "TRACKING FAILED!", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Failures: {failure_count}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            positions.append((frame_number, None, None))
            print(f"❌ Frame {frame_number}: Tracking failed (failure #{failure_count})")
            
            # If we have too many failures, stop
            if failure_count >= 5:
                print(f"⚠️  Too many failures ({failure_count}), stopping diagnosis")
                break
        
        cv2.imshow("Tracking Diagnosis", frame)
        
        key = cv2.waitKey(30) & 0xFF  # Wait 30ms between frames
        if key == ord('q'):
            print(f"⏹️  User quit at frame {frame_number}")
            break
        elif key == ord('s'):
            cv2.imwrite(f"debug_frame_{frame_number}.jpg", frame)
            print(f"💾 Saved debug frame {frame_number}")
        
        frame_number += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Save diagnostic data
    with open("tracking_diagnostic.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame", "x", "y", "success"])
        for i, pos in enumerate(positions):
            success = "Yes" if pos[1] is not None else "No"
            writer.writerow([pos[0], pos[1], pos[2], success])
    
    # Summary
    successful = sum(1 for pos in positions if pos[1] is not None)
    total = len(positions)
    
    print(f"\n📊 DIAGNOSTIC SUMMARY")
    print(f"=" * 40)
    print(f"Frames analyzed: {total}")
    print(f"Successful tracks: {successful}")
    print(f"Failed tracks: {total - successful}")
    print(f"Success rate: {(successful/total)*100:.1f}%")
    print(f"✅ Diagnostic data saved to: tracking_diagnostic.csv")
    
    if successful < total * 0.8:
        print(f"\n⚠️  ISSUES DETECTED:")
        print(f"   - Low tracking success rate")
        print(f"   - Possible causes:")
        print(f"     * Poor ROI selection")
        print(f"     * Object moves too fast")
        print(f"     * Object changes appearance")
        print(f"     * Video quality issues")

if __name__ == "__main__":
    diagnose_tracking_issue()
