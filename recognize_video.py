import cv2
import numpy as np
import pickle
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity

print("start running")
# Load embeddings
with open("embeddings/embeddings.pkl", "rb") as f:
    data = pickle.load(f)

known_embeddings = np.array(data["embeddings"])
known_labels = np.array(data["labels"])

# Initialize models
detector = MTCNN()
embedder = FaceNet()

# Load video
video_path = "test_videos/Test1.mp4"
cap = cv2.VideoCapture(video_path)

THRESHOLD = 0.6  # similarity threshold

frame_count = 0
SKIP_FRAMES = 3   # change this value

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    if frame_count % SKIP_FRAMES != 0:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    results = detector.detect_faces(rgb_frame)

    for result in results:
        x, y, w, h = result['box']
        x, y = abs(x), abs(y)

        face = rgb_frame[y:y+h, x:x+w]
        face = cv2.resize(face, (160, 160))

        # Get embedding
        embedding = embedder.embeddings([face])[0]

        # Compare with known embeddings
        similarities = cosine_similarity([embedding], known_embeddings)[0]

        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        if best_score > THRESHOLD:
            name = known_labels[best_idx]
        else:
            name = "Unknown"

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({best_score:.2f})",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2)

    cv2.imshow("Attendance Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()