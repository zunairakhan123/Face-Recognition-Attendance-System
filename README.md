# 🎓 AI-Based Liveness Attendance System

An intelligent **Face Recognition Attendance System** designed specifically for teachers, integrating **real-time liveness detection (anti-spoofing)** to prevent fake attendance using photos or videos.

This system uses **InsightFace embeddings + cosine similarity + motion-based liveness detection**, making it lightweight, fast, and suitable for real-world deployment. The frontend is a fully responsive **Streamlit dashboard** with an adaptive light/dark theme that automatically follows your device's system preference.

---

## 🚀 Features

- ✅ Real-time face recognition using webcam
- 🧠 **Embedding-based (Open-Set Recognition)** — no retraining required for new teachers
- 🛡️ Motion-based **Anti-Spoofing / Liveness Detection**
- 🔁 Duplicate attendance prevention via cooldown logic
- 💾 SQLite database for persistent storage
- ⚡ Optimized performance (frame-by-frame processing + 50% resize)
- 📊 Interactive Streamlit dashboard with adaptive light/dark theme
- 📅 Date-based filtering to review historical attendance records
- 🏫 Department management — assign/update teacher departments
- 👩‍🏫 Teacher directory with avatar cards
- 📈 Analytics: shift distribution, department breakdown, daily trend charts
- 📁 CSV export support for any filtered view

---

## 🧠 System Architecture

```text
Webcam
  ↓
Motion Detection (Anti-Spoofing / Liveness Check)
  ↓
Face Detection (InsightFace)
  ↓
Face Embedding Extraction
  ↓
Cosine Similarity Matching
  ↓
Attendance Marking (SQLite)
  ↓
Streamlit Dashboard (Light / Dark Adaptive UI)
```

---

## 📂 Project Structure

```text
Face-Recognition-Attendance/
│
├── dataset/                  # Teacher face images (collected via webcam.py)
├── embeddings/               # Saved face embeddings (.npy files)
│   ├── embeddings.npy
│   └── names.npy
├── database/
│   └── attendance.db         # SQLite database
│
├── webcam.py                 # Dataset collection script (webcam capture)
├── recognize.py              # Main recognition + liveness detection script
├── generate_embeddings.py    # Generates embeddings from dataset images
├── database.py               # SQLite operations (init, mark_attendance)
├── config.py                 # Paths, thresholds, and configuration constants
├── requirements.txt          # Python dependencies
│
├── app.py                    # Streamlit dashboard (adaptive UI)
│
└── README.md
```

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/face-attendance-system.git
cd face-attendance-system
```

### 2️⃣ Create a virtual environment *(recommended)*

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS / Linux
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📸 Dataset Collection

Run the dataset collection script:

```bash
python webcam.py
```

📌 Capture **20–25 images per teacher**, covering:

- Front-facing pose
- Slight angles (left, right, up, down)
- Different lighting conditions

> Images are saved automatically to the `dataset/` folder under each teacher's name.

---

## 🧬 Generate Embeddings

```bash
python generate_embeddings.py
```

This produces two files in the `embeddings/` folder:

| File | Description |
|------|-------------|
| `embeddings.npy` | 512-d face embedding vectors |
| `names.npy` | Corresponding teacher names |

> Re-run this script whenever new teachers are added to the dataset.

---

## ▶️ Run the Recognition System

```bash
python recognize.py
```

- Press **Q** to exit the webcam window
- **Move slightly** in front of the camera to pass liveness detection
- Attendance is logged automatically once a face is confirmed as live and recognized

---

## 🌐 Run the Web Dashboard

```bash
streamlit run app.py
```

Open in your browser: **http://localhost:8501**

### Dashboard Sections

| Section | Description |
|---------|-------------|
| **Dashboard** | View attendance records with filters for date, shift, and department |
| **Analytics** | Shift distribution pie chart, department bar chart, daily trend line |
| **Directory** | Teacher cards with avatars; update department assignments |

> The dashboard theme automatically adapts to your device's **light or dark mode** setting.

---

## 🛡️ Anti-Spoofing (Liveness Detection)

This system uses **motion-based liveness detection**:

| Scenario | Result |
|----------|--------|
| ✅ Real person (moving slightly) | Attendance marked |
| ❌ Printed photo (static) | Spoof detected — rejected |
| ❌ Phone/screen display (static) | Spoof detected — rejected |

**How it works:** The system tracks the center point of the detected face bounding box across consecutive frames. If movement exceeds a set threshold (`MOVEMENT_THRESHOLD = 7` pixels), the face is considered live. Otherwise, it is flagged as a possible spoof.

---

## ⚡ Performance Optimizations

| Optimization | Details |
|---|---|
| Per-frame detection | Every frame is processed (required for reliable motion tracking) |
| Frame resizing | Detection runs on a 50% scaled frame for speed |
| CPU-only inference | InsightFace runs in CPU mode (`ctx_id=-1`) |
| Cooldown logic | Prevents duplicate logs within the `ATTENDANCE_COOLDOWN` window |
| Lightweight matching | Cosine similarity via `sklearn` — no GPU needed |

---

## 📊 Technologies Used

| Library | Purpose |
|---------|---------|
| Python 3.x | Core language |
| OpenCV | Webcam capture, frame processing, drawing |
| InsightFace | Face detection & 512-d embedding extraction |
| NumPy | Embedding storage and vector math |
| Scikit-learn | Cosine similarity matching |
| SQLite3 | Attendance and teacher database |
| Streamlit | Interactive web dashboard |
| Plotly | Analytics charts |
| Pandas | Data loading and filtering |

---

## 🔮 Future Improvements

- 🔹 Deep learning anti-spoofing (e.g. Silent Face Anti-Spoofing model)
- 🔹 Live camera feed embedded inside the Streamlit dashboard
- 🔹 Teacher registration UI (add new teachers without touching the filesystem)
- 🔹 Role-based access (admin vs. viewer)
- 🔹 Cloud deployment (Streamlit Cloud / Docker)
- 🔹 Multi-camera support
- 🔹 Email/SMS alerts for absent teachers
- 🔹 Monthly attendance reports with PDF export

---

## 🧪 Testing Scenarios

| Scenario | Expected Result |
|----------|----------------|
| Registered teacher, moving | ✅ Attendance marked |
| Registered teacher, already logged (cooldown active) | ⏳ Countdown shown, no duplicate |
| Unknown face, moving | ❌ Labeled "Unknown" |
| Any face, no motion | 🚫 Labeled "No Motion (Possible Spoof)" |

---

## 👩‍💻 Author

**Zunaira Hawwar**
📧 zunairahawar7@gmail.com

---

## 📜 License

This project is intended for **educational and research purposes only**.
