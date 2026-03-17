# import cv2
# import numpy as np
# import time
# from sklearn.metrics.pairwise import cosine_similarity
# from insightface.app import FaceAnalysis
# from config import *
# from database import init_db, mark_attendance

# # Initialize Database
# init_db()

# # Tracker for cooldown logic
# last_attendance_time = {}  # Stores { "Name": timestamp }

# print("Loading embeddings...")
# known_embeddings = np.load(EMBEDDINGS_PATH)
# known_names = np.load(NAMES_PATH)

# # Initialize InsightFace (Set to CPU)
# app = FaceAnalysis()
# app.prepare(ctx_id=-1) # -1 forces CPU usage , 0 for GPU

# # Open Webcam
# cap = cv2.VideoCapture(CAMERA_INDEX)

# print("System started. Press 'q' to quit.")

# frame_count = 0
# faces = []  # Initialize empty face list

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_count += 1

#     # OPTIMIZATION 1: Only detect faces every 3rd frame to save CPU
#     # This keeps the video smooth while still checking for faces 10 times a second
#     if frame_count % 3 == 0:
#         # OPTIMIZATION 2: Resize frame for detection (smaller = much faster)
#         small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
#         faces = app.get(small_frame)
        
#         # Scale face locations back up since we detected on a half-size image
#         for face in faces:
#             face.bbox *= 2
#             if face.landmark is not None:
#                 face.landmark *= 2

#     # Drawing and Recognition Logic

#     for face in faces:
#         embedding = face.embedding
        
#         # Compare current face with known embeddings
#         similarities = cosine_similarity([embedding], known_embeddings)[0]
#         best_index = np.argmax(similarities)
#         best_score = similarities[best_index]

#         # 1. Check if recognized
#         if best_score > SIMILARITY_THRESHOLD:
#             name = known_names[best_index]
#             current_time = time.time()

#             # 2. Check Attendance Cooldown Logic
#             if name not in last_attendance_time or (current_time - last_attendance_time[name]) > ATTENDANCE_COOLDOWN:
#                 # Mark New Attendance
#                 mark_attendance(name, float(best_score))
#                 last_attendance_time[name] = current_time
#                 print(f"Attendance marked for {name}")
                
#                 color = (0, 255, 0)  # Green for successful mark
#                 label = f"{name} {best_score:.2f}- MARKED"
#             else:
#                 # Still in Cooldown period
#                 remaining = int(ATTENDANCE_COOLDOWN - (current_time - last_attendance_time[name]))
#                 color = (255, 255, 0) # Cyan/Yellow-Blue for already logged
#                 label = f"{name} {best_score:.2f} ({remaining}s)"
#         else:
#             # 3. Unknown Face
#             label = "Unknown"
#             color = (0, 0, 255) # Red for Unknown

#         # Draw Bounding Box
#         bbox = face.bbox.astype(int)
#         cv2.rectangle(
#             frame,
#             (bbox[0], bbox[1]),
#             (bbox[2], bbox[3]),
#             color,
#             2
#         )

#         # Draw Label Text
#         cv2.putText(
#             frame,
#             label,
#             (bbox[0], bbox[1] - 10),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.7,
#             color,
#             2
#         )

#     # Show the output
#     cv2.imshow("Teacher Recognition System", frame)

#     # Quit on 'q' key
#     if cv2.waitKey(1) == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()

#.................................Updated anti-spoofing(motion detection) code............................

import cv2
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
from config import *
from database import init_db, mark_attendance

# -------------------------------
# Initialize Database
# -------------------------------
init_db()

# -------------------------------
# Duplicate Attendance Tracker
# -------------------------------
last_attendance_time = {}

# -------------------------------
# Load Embeddings
# -------------------------------
print("Loading embeddings...")
known_embeddings = np.load(EMBEDDINGS_PATH)
known_names = np.load(NAMES_PATH)

# -------------------------------
# Initialize InsightFace
# -------------------------------
app = FaceAnalysis()
app.prepare(ctx_id=-1)  # CPU mode

# -------------------------------
# Anti-Spoofing (Motion Detection)
# -------------------------------

prev_center = None
MOVEMENT_THRESHOLD = 7


def check_liveness(bbox):
    global prev_center

    center_x = (bbox[0] + bbox[2]) // 2
    center_y = (bbox[1] + bbox[3]) // 2

    current_center = np.array([center_x, center_y])

    if prev_center is None:
        prev_center = current_center
        return False

    movement = np.linalg.norm(current_center - prev_center)

    prev_center = current_center

    if movement > MOVEMENT_THRESHOLD:
        return True

    return False


# -------------------------------
# Start Webcam
# -------------------------------
cap = cv2.VideoCapture(CAMERA_INDEX)

print("System started. Press 'q' to quit.")

frame_count = 0
faces = []

# -------------------------------
# Main Loop
# -------------------------------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # --------------------------------
    # Detect Faces Every 1st Frame (anti spoofing is not working on 3rd frame)
    # --------------------------------
    if frame_count % 1 == 0:

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        faces = app.get(small_frame)

        for face in faces:
            face.bbox *= 2

            if face.landmark is not None:
                face.landmark *= 2

    # --------------------------------
    # Process Faces
    # --------------------------------
    for face in faces:

        bbox = face.bbox.astype(int)

        # -------------------------------
        # Anti Spoofing Check
        # -------------------------------
        is_real = check_liveness(bbox)

        if not is_real:

            label = "No Motion (Possible Spoof)"
            color = (0, 165, 255)

            cv2.rectangle(frame,
                          (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]),
                          color,
                          2)

            cv2.putText(frame,
                        label,
                        (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2)

            continue

        # -------------------------------
        # Face Recognition
        # -------------------------------
        embedding = face.embedding

        similarities = cosine_similarity([embedding], known_embeddings)[0]

        best_index = np.argmax(similarities)
        best_score = similarities[best_index]

        # -------------------------------
        # Recognized Person
        # -------------------------------
        if best_score > SIMILARITY_THRESHOLD:

            name = known_names[best_index]
            current_time = time.time()

            if name not in last_attendance_time or \
               (current_time - last_attendance_time[name]) > ATTENDANCE_COOLDOWN:

                mark_attendance(name, float(best_score))
                last_attendance_time[name] = current_time

                print(f"Attendance marked for {name}")

                color = (0, 255, 0)
                label = f"{name} {best_score:.2f} LIVE"

            else:

                remaining = int(
                    ATTENDANCE_COOLDOWN -
                    (current_time - last_attendance_time[name])
                )

                color = (255, 255, 0)
                label = f"{name} {best_score:.2f} ({remaining}s)"

        # -------------------------------
        # Unknown Face
        # -------------------------------
        else:

            label = "Unknown"
            color = (0, 0, 255)

        # -------------------------------
        # Draw Bounding Box
        # -------------------------------
        cv2.rectangle(frame,
                      (bbox[0], bbox[1]),
                      (bbox[2], bbox[3]),
                      color,
                      2)

        # -------------------------------
        # Draw Label
        # -------------------------------
        cv2.putText(frame,
                    label,
                    (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2)

    # --------------------------------
    # Show Frame
    # --------------------------------
    cv2.imshow("Teacher Recognition System", frame)

    if cv2.waitKey(1) == ord("q"):
        break


# -------------------------------
# Cleanup
# -------------------------------
cap.release()
cv2.destroyAllWindows()
