import sqlite3
from datetime import datetime
from config import DATABASE_PATH

def init_db():
    conn = sqlite3.connect(DATABASE_PATH)  #This creates or opens database file
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            timestamp TEXT,
            confidence REAL
        )
    """)

    conn.commit()
    conn.close()


def mark_attendance(name, confidence):

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute(
        "INSERT INTO attendance (name, timestamp, confidence) VALUES (?, ?, ?)",
        (name, time, confidence),
    )

    conn.commit()
    conn.close()

    print(f"Attendance marked for {name}")