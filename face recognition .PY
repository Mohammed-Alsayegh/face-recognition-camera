from ultralytics import YOLO
import cv2
from deepface import DeepFace
import os

# YOLO model for face detection
yolo_model = YOLO("YOLO model for face detection.pt")

# Path to face database
db_path = "database"

# Adjustable parameters
margin = 20          # pixels to expand bounding box
face_size = (224, 224)
threshold = 0.4     # recognition threshold

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces using YOLO
    yolo_results = yolo_model(frame)

    for r in yolo_results[0].boxes:
        cls = int(r.cls[0].item())
        if cls != 0:  # assuming class 0 = face
            continue

        x1, y1, x2, y2 = map(int, r.xyxy[0].tolist())
        # Expand bounding box
        y1_m = max(0, y1 - margin)
        y2_m = min(frame.shape[0], y2 + margin)
        x1_m = max(0, x1 - margin)
        x2_m = min(frame.shape[1], x2 + margin)

        # Crop and resize face for DeepFace
        face_img = cv2.cvtColor(frame[y1_m:y2_m, x1_m:x2_m], cv2.COLOR_BGR2RGB)
        face_img = cv2.resize(face_img, face_size)

        try:
            # Recognize face using DeepFace
            results = DeepFace.find(
                img_path=face_img,
                db_path=db_path,
                enforce_detection=False,
                detector_backend='opencv',
                model_name='VGG-Face',
                distance_metric='cosine'
            )

            name_to_display = "Anonymous"
            color = (0, 0, 255)  # default red for unknown

            if results and len(results[0]) > 0:
                best_match = results[0].iloc[0]
                distance = best_match["distance"]
                identity_path = best_match["identity"]
                person_name = os.path.basename(os.path.dirname(identity_path))

                if distance <= threshold:
                    name_to_display = f"{person_name} ({distance:.2f})"
                    color = (0, 255, 0)  # green for recognized
                else:
                    name_to_display = f"Anonymous ({distance:.2f})"

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, name_to_display, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        except Exception as e:
            print("Error:", e)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

