import cv2
import os
import time

# Settings
VIDEO_FOLDER = "test_videos"
DURATION = 10   # seconds
FPS = 20

os.makedirs(VIDEO_FOLDER, exist_ok=True)

# Open camera
cap = cv2.VideoCapture(0)

# Get frame size
width = int(cap.get(3))
height = int(cap.get(4))

# Generate filename using time
timestamp = time.strftime("%Y%m%d_%H%M%S")
filename = os.path.join(VIDEO_FOLDER, f"video_{timestamp}.mp4")

# Video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(filename, fourcc, FPS, (width, height))

print("Recording started...")

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    out.write(frame)

    cv2.imshow("Recording...", frame)

    # Stop after DURATION seconds
    if time.time() - start_time > DURATION:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Saved video: {filename}")