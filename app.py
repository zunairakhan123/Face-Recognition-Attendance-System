# import streamlit as st
# import sqlite3
# import pandas as pd
# import subprocess
# import time
# from config import DATABASE_PATH

# st.set_page_config(page_title="AI Face Recognition Attendance", layout="wide")

# # ---------- DATABASE ----------
# def get_connection():
#     return sqlite3.connect(DATABASE_PATH)


# def load_attendance():
#     conn = get_connection()
#     df = pd.read_sql_query("SELECT * FROM attendance ORDER BY timestamp DESC", conn)
#     conn.close()
#     return df

# # ---------- SIDEBAR ----------
# st.sidebar.title("System Control")

# start_btn = st.sidebar.button("Start Recognition")
# stop_btn = st.sidebar.button("Stop Recognition")
# refresh_btn = st.sidebar.button("Refresh Attendance")

# # ---------- TITLE ----------
# st.title("AI Face Recognition Attendance System")
# st.markdown("Real-time teacher attendance using AI face recognition.")

# # ---------- START SYSTEM ----------
# if start_btn:
#     st.sidebar.success("Recognition Started - Wait a few seconds")
#     subprocess.Popen(["python", "recognize.py"])

# # ---------- STOP SYSTEM ----------
# if stop_btn:
#     st.sidebar.warning("Close the camera window to stop recognition.")

# # ---------- DASHBOARD METRICS ----------
# col1, col2, col3 = st.columns(3)

# try:
#     df = load_attendance()
#     total_records = len(df)
#     unique_teachers = df['name'].nunique()

#     col1.metric("Total Attendance Records", total_records)
#     col2.metric("Teachers Detected", unique_teachers)
#     col3.metric("System Status", "Running")

# except:
#     col1.metric("Total Attendance Records", 0)
#     col2.metric("Teachers Detected", 0)
#     col3.metric("System Status", "Idle")

# # ---------- ATTENDANCE TABLE ----------
# st.subheader("Attendance Log")

# if refresh_btn:
#     st.rerun()

# try:
#     df = load_attendance()

#     if not df.empty:
#         st.dataframe(df, use_container_width=True)

#         csv = df.to_csv(index=False).encode('utf-8')
#         st.download_button(
#             "Download Attendance CSV",
#             csv,
#             "attendance.csv",
#             "text/csv"
#         )
#     else:
#         st.info("No attendance records yet.")

# except:
#     st.error("Database not found or empty.")

 #........................................POLISHED UI..........................................

import streamlit as st
import sqlite3
import pandas as pd
import subprocess
import time
from config import DATABASE_PATH

# ─────────────────────────────────────────────
#  PAGE CONFIG — wide layout, custom tab title
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Face Recognition Attendance",
    layout="wide",
    page_icon="🎓",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS — adaptive light / dark theme
#  Uses CSS prefers-color-scheme so it follows
#  the user's OS setting automatically.
#  Accent colour (teal #009a7a / #00c49a) works
#  on both white and dark backgrounds.
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ─────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500&display=swap');

/* ══════════════════════════════════════════
   LIGHT MODE  (system default or explicit)
   ══════════════════════════════════════════ */
:root {
    --bg-primary:    #f4f7f9;
    --bg-card:       #ffffff;
    --bg-card2:      #eef3f7;
    --bg-sidebar:    #ffffff;
    --accent:        #009a7a;          /* darker teal — readable on white */
    --accent2:       #00c49a;
    --accent-dim:    rgba(0,154,122,0.10);
    --accent-glow:   0 2px 16px rgba(0,154,122,0.18);
    --danger:        #d93651;
    --warn:          #e07b00;
    --text-primary:  #1a2430;
    --text-muted:    #7a8ea0;
    --border:        rgba(0,154,122,0.18);
    --divider:       rgba(0,154,122,0.22);
    --shadow:        0 2px 12px rgba(0,0,0,0.07);
    --radius:        10px;
}

/* ══════════════════════════════════════════
   DARK MODE  (OS dark setting)
   ══════════════════════════════════════════ */
@media (prefers-color-scheme: dark) {
    :root {
        --bg-primary:    #0a0e14;
        --bg-card:       #111720;
        --bg-card2:      #141c27;
        --bg-sidebar:    #0c1219;
        --accent:        #00d4aa;      /* brighter teal — readable on dark */
        --accent2:       #0af0c8;
        --accent-dim:    rgba(0,212,170,0.12);
        --accent-glow:   0 0 22px rgba(0,212,170,0.30);
        --danger:        #ff4d6d;
        --warn:          #ffb74d;
        --text-primary:  #e8edf5;
        --text-muted:    #5a7080;
        --border:        rgba(0,212,170,0.14);
        --divider:       rgba(0,212,170,0.20);
        --shadow:        0 4px 24px rgba(0,0,0,0.45);
        --radius:        10px;
    }
}

/* ── Base reset ─────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ── Main container ─────────────────────── */
.main .block-container {
    padding: 2rem 2.5rem 3rem;
    max-width: 1400px;
}

/* ── Sidebar ─────────────────────────────── */
section[data-testid="stSidebar"] {
    background: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: var(--shadow) !important;
}
section[data-testid="stSidebar"] * {
    font-family: 'Rajdhani', sans-serif !important;
}

/* ── Hero header ─────────────────────────── */
.hero-header {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    margin-bottom: 0.4rem;
}
.hero-icon {
    font-size: 2.8rem;
    filter: drop-shadow(0 0 10px var(--accent));
}
.hero-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    /* gradient readable on both backgrounds */
    background: linear-gradient(90deg, var(--accent), var(--accent2), var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1;
}
.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-muted);
    letter-spacing: 1px;
    margin-top: 0.35rem;
}

/* ── Divider ─────────────────────────────── */
.styled-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--divider), transparent);
    margin: 1.2rem 0 1.8rem;
}

/* ── Metric cards ─────────────────────────── */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1.4rem 1.6rem !important;
    box-shadow: var(--accent-glow) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    position: relative;
    overflow: hidden;
}
/* Top accent stripe */
[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent), transparent);
}
[data-testid="metric-container"]:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 0 32px rgba(0,180,140,0.30) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 2.4rem !important;
    font-weight: 700 !important;
    color: var(--accent) !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.85rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}

/* ── Section headings ───────────────────── */
h3, .section-title {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: var(--text-primary) !important;
    border-left: 3px solid var(--accent);
    padding-left: 0.75rem;
    margin: 1.8rem 0 1rem !important;
}

/* ── Dataframe ──────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    box-shadow: var(--shadow) !important;
}
.stDataFrame thead tr th {
    background: var(--bg-card2) !important;
    font-family: 'Rajdhani', sans-serif !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    font-size: 0.78rem !important;
    color: var(--accent) !important;
    border-bottom: 1px solid var(--border) !important;
    padding: 0.8rem 1rem !important;
}
.stDataFrame tbody tr:nth-child(even) { background: var(--accent-dim) !important; }
.stDataFrame tbody tr:hover           { background: rgba(0,180,140,0.08) !important; transition: 0.15s; }

/* ── Buttons — shared base ──────────────── */
.stButton > button {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    border-radius: 6px !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
/* Start button — filled accent */
.stButton:nth-of-type(1) > button {
    background: linear-gradient(135deg, var(--accent2), var(--accent)) !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: var(--accent-glow) !important;
}
.stButton:nth-of-type(1) > button:hover {
    filter: brightness(1.08) !important;
    transform: translateY(-1px) !important;
}
/* Stop button — danger outline */
.stButton:nth-of-type(2) > button {
    background: transparent !important;
    color: var(--danger) !important;
    border: 1px solid var(--danger) !important;
}
.stButton:nth-of-type(2) > button:hover {
    background: rgba(210,54,81,0.08) !important;
}
/* Refresh button — neutral ghost */
.stButton:nth-of-type(3) > button {
    background: transparent !important;
    color: var(--text-muted) !important;
    border: 1px solid var(--border) !important;
}
.stButton:nth-of-type(3) > button:hover {
    color: var(--text-primary) !important;
    border-color: var(--accent) !important;
    background: var(--accent-dim) !important;
}

/* ── Download button ─────────────────────── */
.stDownloadButton > button {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    background: var(--accent-dim) !important;
    color: var(--accent) !important;
    border: 1px solid var(--accent) !important;
    border-radius: 6px !important;
    transition: all 0.2s !important;
}
.stDownloadButton > button:hover {
    background: var(--accent) !important;
    color: #ffffff !important;
    box-shadow: var(--accent-glow) !important;
}

/* ── Alert / info boxes ─────────────────── */
.stAlert {
    border-radius: var(--radius) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}

/* ── Scrollbar ──────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ── Sidebar branding ───────────────────── */
.sidebar-logo {
    text-align: center;
    padding: 1rem 0 0.5rem;
}
.sidebar-logo-icon { font-size: 2.5rem; }
.sidebar-logo-text {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent);
    margin-top: 0.3rem;
}
.sidebar-divider {
    height: 1px;
    background: var(--border);
    margin: 0.8rem 0 1.2rem;
}
.sidebar-label {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
    padding-left: 0.1rem;
}

/* ── Live status badge ──────────────────── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: var(--accent-dim);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 0.3rem 0.9rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: var(--accent);
}
.pulse-dot {
    width: 8px; height: 8px;
    background: var(--accent);
    border-radius: 50%;
    animation: pulse 1.6s ease-in-out infinite;
}
@keyframes pulse {
    0%,100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.35; transform: scale(0.65); }
}

/* ── Empty-state card ───────────────────── */
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    background: var(--bg-card);
    border: 1px dashed var(--border);
    border-radius: var(--radius);
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    box-shadow: var(--shadow);
}
.empty-state-icon { font-size: 2.8rem; margin-bottom: 0.6rem; }

/* ── Timestamp chip ─────────────────────── */
.ts-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-muted);
    background: var(--accent-dim);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
    display: inline-block;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  DATABASE HELPERS
# ─────────────────────────────────────────────

def get_connection():
    """Return a live SQLite connection to the attendance database."""
    return sqlite3.connect(DATABASE_PATH)


def load_attendance():
    """Fetch all attendance rows sorted newest-first."""
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM attendance ORDER BY timestamp DESC", conn)
    conn.close()
    return df

# ─────────────────────────────────────────────
#  SIDEBAR — system controls
# ─────────────────────────────────────────────

# Branded sidebar header
st.sidebar.markdown("""
<div class="sidebar-logo">
    <div class="sidebar-logo-icon">🎓</div>
    <div class="sidebar-logo-text">FaceAttend</div>
</div>
<div class="sidebar-divider"></div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-label">⚡ System Controls</div>', unsafe_allow_html=True)

start_btn    = st.sidebar.button("▶  Start Recognition")
stop_btn     = st.sidebar.button("⏹  Stop Recognition")
refresh_btn  = st.sidebar.button("↻  Refresh Attendance")

st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-label">ℹ️  About</div>', unsafe_allow_html=True)
st.sidebar.caption("AI-powered face recognition for real-time teacher attendance tracking.")

# ─────────────────────────────────────────────
#  HERO HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div class="hero-header">
    <span class="hero-icon">🤖</span>
    <div>
        <p class="hero-title">Face Recognition Attendance</p>
        <p class="hero-sub">// REAL-TIME AI MONITORING SYSTEM · TEACHER ATTENDANCE TRACKER</p>
    </div>
</div>
<div class="styled-divider"></div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HANDLE CONTROL BUTTON ACTIONS
# ─────────────────────────────────────────────

if start_btn:
    # Launch the recognition process in the background
    st.sidebar.success("✅ Recognition started — camera initialising…")
    subprocess.Popen(["python", "recognize.py"])

if stop_btn:
    # The user must close the OpenCV window manually
    st.sidebar.warning("⚠️ Press 'Q' from keyboard to stop recognition.")

# ─────────────────────────────────────────────
#  DASHBOARD METRICS  (3-column KPI row)
# ─────────────────────────────────────────────

col1, col2, col3 = st.columns(3)

try:
    df = load_attendance()
    total_records    = len(df)
    unique_teachers  = df['name'].nunique()

    col1.metric("Total Attendance Records", total_records)
    col2.metric("Teachers Detected", unique_teachers)
    # Animated live-status badge inside the third metric
    col3.metric("System Status", "Running")
    with col3:
        st.markdown("""
        <div style="margin-top:-0.6rem">
            <span class="status-badge">
                <span class="pulse-dot"></span>LIVE
            </span>
        </div>""", unsafe_allow_html=True)

except Exception:
    # Graceful fallback when the database hasn't been created yet
    col1.metric("Total Attendance Records", 0)
    col2.metric("Teachers Detected", 0)
    col3.metric("System Status", "Idle")

# ─────────────────────────────────────────────
#  ATTENDANCE LOG TABLE
# ─────────────────────────────────────────────

st.markdown('<h3>Attendance Log</h3>', unsafe_allow_html=True)

# Show the timestamp of the last data load
st.markdown(
    f'<span class="ts-chip">🕒 Last loaded: {time.strftime("%H:%M:%S")}</span>',
    unsafe_allow_html=True,
)

# Refresh triggers a full Streamlit rerun to reload fresh data
if refresh_btn:
    st.rerun()

try:
    df = load_attendance()

    if not df.empty:
        # Render the dataframe with full-width styling
        st.dataframe(df, use_container_width=True, height=420)

        # Spacer + download action
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇  Download Attendance CSV",
            data=csv,
            file_name="attendance.csv",
            mime="text/csv",
        )
    else:
        # Styled empty state rather than plain st.info
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">📭</div>
            No attendance records found yet.<br>
            Start the recognition system to begin logging.
        </div>""", unsafe_allow_html=True)

except Exception:
    st.error("⚠️  Database not found or could not be read. Ensure the path in config.py is correct.")
