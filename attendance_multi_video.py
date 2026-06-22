import cv2
import numpy as np
import pickle
import pandas as pd
import os
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity

# Load embeddings
with open("embeddings/embeddings.pkl", "rb") as f:
    data = pickle.load(f)

known_embeddings = np.array(data["embeddings"])
known_labels = np.array(data["labels"])

students = list(set(known_labels))

# Video folder
VIDEO_FOLDER = "test_videos"

video_files = sorted(os.listdir(VIDEO_FOLDER))

# Attendance structure
intervals = len(video_files)
attendance = {student: [0]*intervals for student in students}

# Models
detector = MTCNN()
embedder = FaceNet()

THRESHOLD = 0.6
FRAME_SKIP = 10

# Loop through each video (each = interval)
for idx, video_name in enumerate(video_files):

    print(f"\nProcessing {video_name} as Interval {idx+1}")

    cap = cv2.VideoCapture(os.path.join(VIDEO_FOLDER, video_name))

    frame_count = 0
    interval_counts = {student: 0 for student in students}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        START_FRAME = 100
        END_FRAME = 200

        if frame_count < START_FRAME:
            continue
        if frame_count > END_FRAME:
            break

        if frame_count % FRAME_SKIP != 0:
            continue

        frame = cv2.resize(frame, (640, 480))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(rgb)

        for result in results:
            x, y, w, h = result['box']
            x, y = abs(x), abs(y)

            face = rgb[y:y+h, x:x+w]
            if face.shape[0] == 0 or face.shape[1] == 0:
                continue

            face = cv2.resize(face, (160, 160))
            embedding = embedder.embeddings([face])[0]

            similarities = cosine_similarity([embedding], known_embeddings)[0]
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]

            if best_score > THRESHOLD:
                name = known_labels[best_idx]
                interval_counts[name] += 1

                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(frame, name, (x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("Processing Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Decide presence in this video
    for student in students:
        if interval_counts[student] > 5:  # adjust threshold
            attendance[student][idx] = 1

# Final voting
final_result = {}
for student in students:
    if sum(attendance[student]) >= 2:   # your rule
        final_result[student] = "Present"
    else:
        final_result[student] = "Absent"

# Save CSV
df = pd.DataFrame(attendance).T
df.columns = [f"V{i+1}" for i in range(intervals)]
df["Final"] = [final_result[s] for s in df.index]

df.to_csv("output/attendance_multi_video.csv")

print("\nFinal Attendance:")
print(df)