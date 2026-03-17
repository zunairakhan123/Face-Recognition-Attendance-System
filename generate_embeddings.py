import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from config import DATASET_PATH, EMBEDDINGS_PATH, NAMES_PATH

app = FaceAnalysis()     # This loads InsightFace ArcFace model.
app.prepare(ctx_id=0)

embeddings = []
names = []

for person_name in os.listdir(DATASET_PATH):

    person_folder = os.path.join(DATASET_PATH, person_name)

    for img_name in os.listdir(person_folder):

        img_path = os.path.join(person_folder, img_name)

        img = cv2.imread(img_path)  # Loads the teacher image.

        faces = app.get(img)

        if len(faces) == 0:
            continue

        embedding = faces[0].embedding

        embeddings.append(embedding)
        names.append(person_name)

print("Embeddings generated:", len(embeddings))

np.save(EMBEDDINGS_PATH, embeddings)
np.save(NAMES_PATH, names)

print("Embeddings saved successfully")