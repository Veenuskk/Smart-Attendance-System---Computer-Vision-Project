import cv2
import os

# Path to your video file
video_path = "Renjith.mp4"

# Folder where frames will be saved
output_folder = "Renjith"

# Create folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Read the video
cap = cv2.VideoCapture(video_path)

frame_count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Save each frame as an image
    frame_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
    cv2.imwrite(frame_filename, frame)
    
    frame_count += 1

cap.release()

print(f"Total frames saved: {frame_count}")