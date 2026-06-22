import cv2
import numpy as np
import pickle
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity

# Load embeddings
with open("embeddings/embeddings.pkl", "rb") as f:
    data = pickle.load(f)

known_embeddings = np.array(data["embeddings"])
known_labels = np.array(data["labels"])

# Initialize models
detector = MTCNN()
embedder = FaceNet()

THRESHOLD = 0.6
FRAME_SKIP = 3

# Open webcam (Mac)
cap = cv2.VideoCapture(0)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Resize for speed (safe resize)
    h, w = frame.shape[:2]
    if w > 800:
        scale = 800 / w
        frame = cv2.resize(frame, (800, int(h * scale)))

    if frame_count % FRAME_SKIP != 0:
        continue

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
            color = (0, 255, 0)
        else:
            name = "Unknown"
            color = (0, 0, 255)

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Label
        cv2.putText(frame, f"{name} ({best_score:.2f})",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2)

    cv2.imshow("Live Face Recognition", frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()