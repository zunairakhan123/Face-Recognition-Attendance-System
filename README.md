# 🎓 AI-Based Liveness Attendance System

An intelligent **Face Recognition Attendance System** designed specifically for teachers, integrating **real-time liveness detection (anti-spoofing)** to prevent fake attendance using photos or videos.

This system uses **InsightFace embeddings + cosine similarity + motion-based liveness detection**, making it lightweight, fast, and suitable for real-world deployment.

---

## 🚀 Features

- ✅ Real-time face recognition using webcam  
- 🧠 **Embedding-based (Open-Set Recognition)** – no retraining required for new teachers  
- 🛡️ Motion-based **Anti-Spoofing Detection**  
- 🔁 Duplicate attendance prevention (cooldown logic)  
- 💾 SQLite database for persistent storage  
- ⚡ Optimized performance (frame skipping + resizing)  
- 📊 Streamlit dashboard for monitoring attendance  
- 📁 CSV export support  

---

## 🧠 System Architecture

```text
Webcam
  ↓
Motion Detection (Anti-Spoofing)
  ↓
Face Detection (InsightFace)
  ↓
Face Embedding Extraction
  ↓
Cosine Similarity Matching
  ↓
Attendance Marking (SQLite)
  ↓
Streamlit Dashboard
```
---
## 📂 Project Structure
```text
Face-Recognition-Attendance/
│
├── dataset/                # Teacher images
├── embeddings/             # Saved embeddings (.npy)
├── dataset/                # SQLite database(attendance.db)
│
├── webcam.py               # Script for collecting dataset through webcam
│
├── recognize.py            # Main recognition script
├── generate_embeddings.py  # Embedding generator
├── database.py             # SQLite operations
├── config.py               # Configurations
├── requirements.txt        # Requirements to run the project
│
├── app.py                  # Streamlit frontend
│
└── README.md
```
---

## ⚙️ Setup & Installation

1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/face-attendance-system.git
cd face-attendance-system
```
2️⃣ Create virtual environment (Optional but recommended)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```
3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 📸 Dataset Collection

Run the dataset script:

```bash
python webcam.py
```

📌 Capture 20–23 images per teacher, including:

Front face

Slight angles(left,right,up,down)

Different lighting conditions

### 🧬 Generate Embeddings
```bash
python generate_embeddings.py
```

This will create:

embeddings.npy
names.npy

### ▶️ Run Face Recognition System
```bash
python recognize.py
```
Press *Q* to exit

Move slightly to pass liveness detection

### 🌐 Run Web Dashboard
```bash
streamlit run app.py
```
Open in browser:

http://localhost:8501

### 🛡️Anti-Spoofing (Liveness Detection)

This system uses motion-based detection:

✅ Real person → slight movement detected

❌ Photo / screen → no motion → rejected

---

## 🧪 Testing
Scenario	Result
Real person (moving)	✅ Attendance marked
Printed photo(Not moving)❌ Spoof detected
Phone screen(Not moving )❌ Spoof detected

---

## ⚡Performance Optimizations

Face detection every 3rd frame

Frame resizing (50%)

CPU-only inference

Lightweight embedding comparison

---

## 📊 Technologies Used

Python

OpenCV

InsightFace

NumPy

Scikit-learn

SQLite

Streamlit

---

## 🔮 Future Improvements

🔹 Deep learning anti-spoofing (Silent Face Anti-Spoofing)

🔹 Live camera feed inside Streamlit

🔹 Teacher registration UI

🔹 Cloud deployment

🔹 Multi-camera support

---

## 👩‍💻 Author

**Zunaira Hawwar**
*Email*: zunairahawar7@gmail.com

---

## 📜 License

This project is for educational and research purposes only.
---
