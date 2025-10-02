import cv2
import csv
import sys
import os

def flexible_tracking(video_file=None):
    """
    Flexible tracking script that can work with any video file.
    """
    # Determine which video file to use
    if video_file is None:
        # Check for command line argument
        if len(sys.argv) > 1:
            video_file = sys.argv[1]
        else:
            # Default to swimming_video.mp4, but check if fly.mp4 exists
            if os.path.exists("fly.mp4"):
                print("🪰 Found fly.mp4 - using it for tracking")
                video_file = "fly.mp4"
            elif os.path.exists("swimming_video.mp4"):
                print("🏊 Using swimming_video.mp4 for tracking")
                video_file = "swimming_video.mp4"
            else:
                print("❌ No video file found. Please specify a video file.")
                return
    
    print(f"📹 Tracking video: {video_file}")
    
    cap = cv2.VideoCapture(video_file)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file: {video_file}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"📊 Video Info:")
    print(f"   Total frames: {total_frames}")
    print(f"   FPS: {fps}")
    print(f"   Resolution: {width}x{height}")
    print(f"   Duration: {duration:.1f} seconds")
    
    if total_frames < 10:
        print(f"⚠️  Warning: Very short video ({total_frames} frames)")
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Could not read first frame")
        cap.release()
        return
    
    print(f"\n🎯 Please select the object to track:")
    print(f"   - Click and drag to draw a rectangle around the object")
    print(f"   - Press SPACE or ENTER to confirm")
    print(f"   - Press 'c' to cancel")
    
    bbox = cv2.selectROI("Select Object to Track", frame, fromCenter=False, showCrosshair=True)
    
    if bbox[2] == 0 or bbox[3] == 0:
        print("❌ No valid region selected. Exiting.")
        cap.release()
        return
    
    print(f"✅ Selected region: {bbox}")
    
    # Create tracker
    try:
        tracker = cv2.legacy.TrackerCSRT_create()
        print("✅ Using CSRT tracker")
    except:
        try:
            tracker = cv2.TrackerCSRT_create()
            print("✅ Using CSRT tracker (non-legacy)")
        except:
            print("❌ Could not create tracker")
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
    output_file = f"tracked_{os.path.splitext(os.path.basename(video_file))[0]}.mp4"
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Tracking variables
    positions = []
    frame_number = 0
    successful_tracks = 0
    
    print(f"\n🏃 Starting tracking...")
    print(f"   Press 'q' to quit early")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"\n📹 Reached end of video (frame {frame_number}/{total_frames})")
            break
        
        success, box = tracker.update(frame)
        
        if success:
            x, y, w, h = [int(v) for v in box]
            cx, cy = x + w // 2, y + h // 2
            
            # Draw tracking info
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
            cv2.putText(frame, f"Frame: {frame_number}/{total_frames}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Pos: ({cx}, {cy})", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            positions.append((frame_number, cx, cy))
            successful_tracks += 1
        else:
            cv2.putText(frame, f"Frame: {frame_number}/{total_frames}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "TRACKING LOST!", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            positions.append((frame_number, None, None))
        
        out.write(frame)
        cv2.imshow("Flexible Tracking", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"\n⏹️  User stopped tracking at frame {frame_number}")
            break
        
        frame_number += 1
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Save positions to CSV
    csv_file = f"tracked_positions_{os.path.splitext(os.path.basename(video_file))[0]}.csv"
    with open(csv_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame", "x", "y"])
        writer.writerows(positions)
    
    # Print summary
    success_rate = (successful_tracks / frame_number) * 100 if frame_number > 0 else 0
    
    print(f"\n📊 TRACKING SUMMARY")
    print(f"=" * 50)
    print(f"Video: {video_file}")
    print(f"Total frames processed: {frame_number}")
    print(f"Successful tracks: {successful_tracks}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"✅ Video saved as: {output_file}")
    print(f"✅ Positions saved as: {csv_file}")
    
    if total_frames < 10:
        print(f"\n💡 Note: This was a very short video ({total_frames} frames)")
        print(f"   The tracking completed normally - it just finished quickly!")

if __name__ == "__main__":
    flexible_tracking()
