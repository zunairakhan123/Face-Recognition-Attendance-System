import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import subprocess
import time
from datetime import datetime
from config import DATABASE_PATH
import cv2
import os
import numpy as np
from config import DATASET_PATH, EMBEDDINGS_PATH, NAMES_PATH

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Face Recognition Attendance",
    layout="wide",
    page_icon="🎓",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS — Adaptive Light / Dark Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500&display=swap');

/* ══════════════════════════════
   DARK MODE (default)
══════════════════════════════ */
:root {
    --bg:           #070b10;
    --bg-card:      #0d1520;
    --bg-card2:     #111d2c;
    --accent:       #00e5a0;
    --accent2:      #00b4ff;
    --accent3:      #7c3aed;
    --danger:       #ff4060;
    --warn:         #ffb020;
    --text:         #e2eaf4;
    --text-muted:   #4a6080;
    --border:       rgba(0,229,160,0.14);
    --glow-green:   0 0 30px rgba(0,229,160,0.25);
    --glow-blue:    0 0 30px rgba(0,180,255,0.20);
    --glow-purple:  0 0 30px rgba(124,58,237,0.20);
    --grid-line:    rgba(0,229,160,0.03);
    --sidebar-bg:   #080c12;
    --table-even:   rgba(255,255,255,0.01);
    --table-hover:  rgba(0,229,160,0.04);
    --shadow:       0 4px 24px rgba(0,0,0,0.45);
    --radius:       12px;
    --radius-lg:    18px;
}

/* ══════════════════════════════
   LIGHT MODE
══════════════════════════════ */
@media (prefers-color-scheme: light) {
    :root {
        --bg:           #f0f4f8;
        --bg-card:      #ffffff;
        --bg-card2:     #eaf2ee;
        --accent:       #00916a;
        --accent2:      #0077cc;
        --accent3:      #6d28d9;
        --danger:       #dc2626;
        --warn:         #d97706;
        --text:         #0f1f2e;
        --text-muted:   #6b8099;
        --border:       rgba(0,145,106,0.20);
        --glow-green:   0 2px 16px rgba(0,145,106,0.18);
        --glow-blue:    0 2px 16px rgba(0,119,204,0.18);
        --glow-purple:  0 2px 16px rgba(109,40,217,0.14);
        --grid-line:    rgba(0,145,106,0.05);
        --sidebar-bg:   #ffffff;
        --table-even:   rgba(0,145,106,0.03);
        --table-hover:  rgba(0,145,106,0.06);
        --shadow:       0 2px 12px rgba(0,0,0,0.09);
        --radius:       12px;
        --radius-lg:    18px;
    }
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.main .block-container {
    padding: 0 2rem 3rem !important;
    max-width: 1500px !important;
}

/* ── Background grid ── */
body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(var(--grid-line) 1px, transparent 1px),
        linear-gradient(90deg, var(--grid-line) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--sidebar-bg) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: var(--shadow) !important;
    width: 240px !important;
}
section[data-testid="stSidebar"] * {
    font-family: 'Space Mono', monospace !important;
}
section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--accent) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    border-radius: 6px !important;
    padding: 0.5rem 0.8rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
    margin-bottom: 0.4rem !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--accent) !important;
    color: #000 !important;
    border-color: var(--accent) !important;
    box-shadow: var(--glow-green) !important;
}

/* ── Top nav bar ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.4rem 0 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.8rem;
    position: relative;
}
.topbar::after {
    content: '';
    position: absolute;
    bottom: -1px; left: 0;
    width: 120px; height: 2px;
    background: linear-gradient(90deg, var(--accent), transparent);
}
.brand {
    display: flex; align-items: center; gap: 0.8rem;
}
.brand-icon {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem;
    box-shadow: var(--glow-green);
}
.brand-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.55rem;
    font-weight: 800;
    letter-spacing: 0px;
    color: var(--text);
}
.brand-name span { color: var(--accent); }
.brand-tagline {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    color: var(--text-muted);
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.live-badge {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: rgba(0,229,160,0.07);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 0.35rem 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: var(--accent);
    letter-spacing: 1px;
}
.pulse {
    width: 7px; height: 7px;
    background: var(--accent);
    border-radius: 50%;
    animation: blink 1.5s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.3;transform:scale(0.6)} }

/* ── Metric cards ── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}
.metric-card {
    background: var(--bg-card);
    border-radius: var(--radius-lg);
    padding: 1.5rem 1.8rem;
    border: 1px solid var(--border);
    position: relative;
    overflow: hidden;
    transition: transform 0.25s, box-shadow 0.25s;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--glow-green);
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--accent), transparent);
}
.metric-card.blue::before  { background: linear-gradient(90deg, var(--accent2), transparent); }
.metric-card.purple::before { background: linear-gradient(90deg, var(--accent3), transparent); }
.metric-card:hover { box-shadow: var(--glow-green); }
.metric-card.blue:hover  { box-shadow: var(--glow-blue); }
.metric-card.purple:hover { box-shadow: var(--glow-purple); }
.metric-num {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: var(--accent);
    line-height: 1;
    margin-bottom: 0.4rem;
}
.metric-card.blue .metric-num  { color: var(--accent2); }
.metric-card.purple .metric-num { color: var(--accent3); }
.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-muted);
}
.metric-icon {
    position: absolute; bottom: 1rem; right: 1.2rem;
    font-size: 2.4rem;
    opacity: 0.48;
}

/* ── Section nav pills ── */
.section-nav {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.8rem;
}
.nav-pill {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 0.45rem 1.1rem;
    border-radius: 6px;
    border: 1px solid transparent;
    cursor: pointer;
    transition: all 0.2s;
    color: var(--text-muted);
}
.nav-pill.active {
    background: var(--table-hover);
    border-color: var(--accent);
    color: var(--accent);
}
/* ── filter chip active ── */
.filter-chip.active {
    border-color: var(--accent);
    color: var(--accent);
    background: var(--table-hover);
}

/* ── Panel heading ── */
.panel-heading {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: 0.5px;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.panel-heading::before {
    content: '';
    width: 4px; height: 20px;
    background: var(--accent);
    border-radius: 2px;
    box-shadow: 0 0 10px var(--accent);
}

/* ── Interactive table ── */
.table-wrapper {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    overflow: hidden;
    box-shadow: var(--shadow);
}
.table-header {
    display: grid;
    padding: 0.75rem 1.2rem;
    background: var(--bg-card2);
    border-bottom: 1px solid var(--border);
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent);
}
.table-header.cols-6 { grid-template-columns: 2fr 1.5fr 1fr 1fr 1fr 1fr; }
.table-header.cols-4 { grid-template-columns: 2fr 1.5fr 1fr 1fr; }
.table-row {
    display: grid;
    padding: 0.85rem 1.2rem;
    border-bottom: 1px solid var(--border);
    font-size: 0.83rem;
    color: var(--text);
    transition: background 0.15s;
    align-items: center;
}
.table-row.cols-6 { grid-template-columns: 2fr 1.5fr 1fr 1fr 1fr 1fr; }
.table-row.cols-4 { grid-template-columns: 2fr 1.5fr 1fr 1fr; }
.table-row:last-child { border-bottom: none; }
.table-row:hover { background: var(--table-hover); }
.table-row:nth-child(even) { background: var(--table-even); }
.table-row:nth-child(even):hover { background: var(--table-hover); }
.badge {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 0.22rem 0.65rem;
    border-radius: 20px;
    font-weight: 700;
}
.badge-green { background: rgba(0,145,106,0.12); color: var(--accent); border: 1px solid rgba(0,145,106,0.30); }
.badge-blue  { background: rgba(0,119,204,0.12); color: var(--accent2); border: 1px solid rgba(0,119,204,0.30); }
.badge-warn  { background: rgba(217,119,6,0.12); color: var(--warn); border: 1px solid rgba(217,119,6,0.30); }
.badge-muted { background: rgba(107,128,153,0.12); color: var(--text-muted); border: 1px solid rgba(107,128,153,0.25); }
.table-footer {
    padding: 0.75rem 1.2rem;
    background: var(--bg-card2);
    border-top: 1px solid var(--border);
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--text-muted);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* ── Filter row ── */
.filter-row {
    display: flex; gap: 0.6rem; margin-bottom: 1rem; flex-wrap: wrap;
}
.filter-chip {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 0.35rem 0.85rem;
    border-radius: 20px;
    border: 1px solid var(--border);
    color: var(--text-muted);
    background: transparent;
    cursor: pointer;
    transition: all 0.2s;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 4rem 1rem;
    color: var(--text-muted);
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 1px;
}
.empty-state-icon { font-size: 3rem; margin-bottom: 0.8rem; filter: grayscale(1); opacity: 0.5; }

/* ── Chart panel ── */
.chart-panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.4rem;
    margin-bottom: 1rem;
}

/* ── Directory card ── */
.teacher-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    transition: all 0.2s;
    position: relative;
    overflow: hidden;
}
.teacher-card:hover {
    border-color: var(--accent);
    background: var(--bg-card2);
    transform: translateX(4px);
    box-shadow: var(--glow-green);
}
.teacher-avatar {
    width: 40px; height: 40px;
    border-radius: 10px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    display: flex; align-items: center; justify-content: center;
    font-family: 'Syne', sans-serif;
    font-weight: 800; font-size: 1rem;
    color: #000;
    flex-shrink: 0;
}
.teacher-name {
    font-family: 'Syne', sans-serif;
    font-size: 0.92rem;
    font-weight: 700;
    color: var(--text);
}
.teacher-dept {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 1px;
    color: var(--text-muted);
    text-transform: uppercase;
    margin-top: 0.15rem;
}
.teacher-id {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    color: var(--text-muted);
}
.teacher-status {
    margin-left: auto; flex-shrink: 0;
}

/* ── Form card ── */
.form-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.4rem 1.6rem;
}
.form-card label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}

/* ── Download btn ── */
.stDownloadButton > button {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    background: transparent !important;
    color: var(--accent) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    transition: all 0.2s !important;
}
.stDownloadButton > button:hover {
    background: var(--accent) !important;
    color: #000 !important;
    box-shadow: var(--glow-green) !important;
}

/* ── Selectbox / inputs ── */
[data-baseweb="select"], [data-baseweb="input"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
}
div[data-baseweb="select"] > div {
    background: var(--bg-card2) !important;
    border-color: var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}
div[data-baseweb="select"] > div:hover {
    border-color: var(--accent) !important;
}
/* dropdown list */
[data-baseweb="popover"] [role="option"] {
    background: var(--bg-card) !important;
    color: var(--text) !important;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [aria-selected="true"] {
    background: var(--bg-card2) !important;
    color: var(--accent) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ── Form submit button ── */
[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #000 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 8px !important;
    width: 100% !important;
    transition: all 0.2s !important;
    box-shadow: var(--glow-green) !important;
}
[data-testid="stFormSubmitButton"] > button:hover {
    filter: brightness(1.1) !important;
    transform: translateY(-1px) !important;
}

/* ── Alert ── */
.stAlert {
    background: var(--bg-card2) !important;
    border-radius: var(--radius) !important;
    border-left-color: var(--accent) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
}

/* Sidebar radio hidden — we handle nav with session state */
.stRadio { display: none; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  DATABASE HELPERS  (unchanged)
# ─────────────────────────────────────────────

def get_connection():
    return sqlite3.connect(DATABASE_PATH)

def load_attendance():
    conn = get_connection()
    query = """
        SELECT a.name, IFNULL(t.department, 'Unassigned') as department, a.date, a.time, a.shift, a.status 
        FROM attendance a
        LEFT JOIN teachers t ON a.name = t.name
        ORDER BY a.date DESC, a.time DESC
    """
    try:
        df = pd.read_sql_query(query, conn)
    except:
        df = pd.DataFrame()
    conn.close()
    return df

def load_teachers():
    conn = get_connection()
    try:
        df = pd.read_sql_query("SELECT id, name, department, status FROM teachers", conn)
    except:
        df = pd.DataFrame()
    conn.close()
    return df

def update_teacher_department(name, new_department):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE teachers SET department = ? WHERE name = ?",
        (new_department, name)
    )
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────
#  SESSION STATE — active section
# ─────────────────────────────────────────────
if "section" not in st.session_state:
    st.session_state.section = "dashboard"


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.2rem 0 0.5rem; text-align:center;">
        <div style="font-family:'Syne',sans-serif; font-size:1.25rem; font-weight:800; color:var(--accent); letter-spacing:1px;">FACE<span style="color:var(--text);">ATTEND</span></div>
        <div style="font-family:'Space Mono',monospace; font-size:0.58rem; letter-spacing:2px; color:var(--text-muted); text-transform:uppercase; margin-top:0.2rem;">AI Attendance System</div>
        <div style="height:1px; background:var(--border); margin:1rem 0;"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-family:\'Space Mono\',monospace; font-size:0.6rem; letter-spacing:2px; color:var(--text-muted); text-transform:uppercase; margin-bottom:0.5rem;">// Navigation</div>', unsafe_allow_html=True)

    if st.button("⬛  Dashboard"):
        st.session_state.section = "dashboard"
        st.rerun()
    if st.button("◈  Analytics"):
        st.session_state.section = "analytics"
        st.rerun()
    if st.button("◉  Directory"):
        st.session_state.section = "directory"
        st.rerun()

    st.markdown('<div style="height:1px; background:var(--border); margin:1rem 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:\'Space Mono\',monospace; font-size:0.6rem; letter-spacing:2px; color:var(--text-muted); text-transform:uppercase; margin-bottom:0.5rem;">// Controls</div>', unsafe_allow_html=True)

    start_btn   = st.button("▶  Start Recognition")
    stop_btn    = st.button("⏹  Stop Recognition")
    refresh_btn = st.button("↻  Refresh Data")

    st.markdown('<div style="height:1px; background:var(--border); margin:1rem 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:\'Space Mono\',monospace; font-size:0.62rem; color:var(--text-muted); line-height:1.7;">AI-powered face recognition for real-time HR analytics & teacher attendance monitoring.</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  BUTTON ACTIONS
# ─────────────────────────────────────────────
if start_btn:
    st.sidebar.success("✅ Recognition started — camera initialising…")
    subprocess.Popen(["python", "recognize.py"])

if stop_btn:
    st.sidebar.warning("⚠️ Press 'Q' from keyboard to stop recognition.")

if refresh_btn:
    st.rerun()


# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
df_attendance = load_attendance()
df_teachers   = load_teachers()

today_str = datetime.now().strftime("%Y-%m-%d")
total_records    = len(df_attendance)
unique_teachers  = df_attendance['name'].nunique() if not df_attendance.empty else 0

today_df_all = df_attendance[df_attendance['date'] == today_str] if not df_attendance.empty else pd.DataFrame()
today_count = len(today_df_all)


# ─────────────────────────────────────────────
#  TOP BAR
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="topbar">
    <div class="brand">
        <div class="brand-icon">🤖</div>
        <div>
            <div class="brand-name">Face<span>Attend</span></div>
            <div class="brand-tagline">Real-Time HR Analytics · Teacher Attendance Tracker</div>
        </div>
    </div>
    <div style="display:flex; align-items:center; gap:1rem;">
        <div style="font-family:'Space Mono',monospace; font-size:0.65rem; color:var(--text-muted); letter-spacing:1px;">{time.strftime('%a %b %d · %H:%M')}</div>
        <div class="live-badge"><span class="pulse"></span> LIVE</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  METRIC CARDS
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="metric-row">
    <div class="metric-card">
        <div class="metric-num">{total_records}</div>
        <div class="metric-label">Total Attendance Logs</div>
        <div class="metric-icon">📋</div>
    </div>
    <div class="metric-card blue">
        <div class="metric-num">{unique_teachers}</div>
        <div class="metric-label">Unique Teachers Present</div>
        <div class="metric-icon">👩‍🏫</div>
    </div>
    <div class="metric-card purple">
        <div class="metric-num">{today_count}</div>
        <div class="metric-label">Logs Today</div>
        <div class="metric-icon">📅</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HELPER: shift badge colour
# ─────────────────────────────────────────────
def shift_badge(shift):
    c = {"Morning": "badge-green", "Afternoon": "badge-blue", "Evening": "badge-warn"}.get(str(shift), "badge-muted")
    return f'<span class="badge {c}">{shift}</span>'

def status_badge(status):
    c = "badge-green" if str(status).lower() == "present" else "badge-warn"
    return f'<span class="badge {c}">{status}</span>'


# ═══════════════════════════════════════════════════════════════
#  SECTION: Updated DASHBOARD
# ═══════════════════════════════════════════════════════════════
if st.session_state.section == "dashboard":

    st.markdown('<div class="panel-heading">Attendance Records</div>', unsafe_allow_html=True)

    if not df_attendance.empty:

        # ── Filters ──
        fc1, fc2, fc3 = st.columns(3)

        with fc1:
            # Build sorted list of all available dates
            all_dates = sorted(df_attendance['date'].dropna().unique(), reverse=True)
            date_options = ["Today", "All Dates"] + [d for d in all_dates if d != today_str]
            date_filter = st.selectbox("Filter by Date", date_options)

        with fc2:
            shift_filter = st.selectbox("Filter by Shift", ["All", "Morning", "Afternoon", "Evening"])

        with fc3:
            dept_options = ["All"] + sorted(list(df_attendance['department'].dropna().unique()))
            dept_filter  = st.selectbox("Filter by Department", dept_options)

        # ── Apply filters ──
        if date_filter == "Today":
            filtered_df = df_attendance[df_attendance['date'] == today_str].copy()
        elif date_filter == "All Dates":
            filtered_df = df_attendance.copy()
        else:
            filtered_df = df_attendance[df_attendance['date'] == date_filter].copy()

        if shift_filter != "All":
            filtered_df = filtered_df[filtered_df['shift'] == shift_filter]
        if dept_filter != "All":
            filtered_df = filtered_df[filtered_df['department'] == dept_filter]

        # ── Active filter summary chip ──
        label_parts = []
        label_parts.append(date_filter)
        if shift_filter != "All":
            label_parts.append(shift_filter)
        if dept_filter != "All":
            label_parts.append(dept_filter)
        filter_summary = " · ".join(label_parts)

        st.markdown(f"""
        <div style="margin-bottom:0.8rem; display:flex; align-items:center; gap:0.6rem;">
            <span style="font-family:'Space Mono',monospace; font-size:0.65rem; color:var(--text-muted); letter-spacing:1px;">Showing:</span>
            <span style="font-family:'Space Mono',monospace; font-size:0.65rem; color:var(--accent); background:var(--table-hover); border:1px solid var(--border); border-radius:20px; padding:0.2rem 0.75rem; letter-spacing:1px;">{filter_summary}</span>
            <span style="font-family:'Space Mono',monospace; font-size:0.65rem; color:var(--text-muted); letter-spacing:1px;">{len(filtered_df)} record(s)</span>
        </div>
        """, unsafe_allow_html=True)

        if not filtered_df.empty:
            # ── Custom interactive table ──
            st.markdown("""
            <div class="table-wrapper">
                <div class="table-header cols-6">
                    <span>Name</span><span>Department</span><span>Date</span><span>Time</span><span>Shift</span><span>Status</span>
                </div>
            """, unsafe_allow_html=True)

            rows_html = ""
            for _, row in filtered_df.iterrows():
                # Highlight today's rows subtly
                is_today = row['date'] == today_str
                date_color = "color:var(--accent);" if is_today else "color:var(--text-muted);"
                rows_html += f"""
                <div class="table-row cols-6">
                    <span style="font-family:'Syne',sans-serif;font-weight:700;color:var(--text);">{row['name']}</span>
                    <span style="font-family:'Space Mono',monospace;font-size:0.72rem;color:var(--text-muted);">{row['department']}</span>
                    <span style="font-family:'Space Mono',monospace;font-size:0.72rem;{date_color}">{row['date']}</span>
                    <span style="font-family:'Space Mono',monospace;font-size:0.72rem;color:var(--accent);">{row['time']}</span>
                    {shift_badge(row['shift'])}
                    {status_badge(row['status'])}
                </div>"""

            st.markdown(rows_html, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="table-footer">
                <span>{len(filtered_df)} records shown</span>
                <span>{time.strftime('%H:%M:%S')} · auto-refresh on demand</span>
            </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Download ──
            st.markdown('<div style="height:0.8rem"></div>', unsafe_allow_html=True)
            filename = f"attendance_{date_filter.replace(' ', '_').lower()}.csv"
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇  Export Filtered Records as CSV", csv, filename, "text/csv")

        else:
            st.markdown("""
            <div class="table-wrapper">
                <div class="empty-state">
                    <div class="empty-state-icon">🔍</div>
                    No records match the selected filters.
                </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="table-wrapper">
            <div class="empty-state">
                <div class="empty-state-icon">📭</div>
                No attendance records yet. Start the recognition system.
            </div>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  SECTION: ANALYTICS
# ═══════════════════════════════════════════════════════════════
elif st.session_state.section == "analytics":

    st.markdown('<div class="panel-heading">Attendance Analytics & Trends</div>', unsafe_allow_html=True)

    if not df_attendance.empty:

        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
            shift_counts = df_attendance['shift'].value_counts().reset_index()
            shift_counts.columns = ['Shift', 'Count']
            fig_pie = px.pie(
                shift_counts, values='Count', names='Shift',
                title="Log Distribution by Shift", hole=0.5,
                color_discrete_sequence=["#00916a", "#0077cc", "#6d28d9", "#dc2626"]
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Space Mono", size=11),
                title_font=dict(family="Syne", size=14),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                margin=dict(t=50, b=20, l=10, r=10),
            )
            fig_pie.update_traces(textfont=dict(family="Space Mono", size=11))
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
            dept_counts = df_attendance.groupby('department').size().reset_index(name='Present Count')
            fig_bar = px.bar(
                dept_counts, x='department', y='Present Count',
                title="Attendance Logs by Department",
                text='Present Count',
                color='Present Count',
                color_continuous_scale=[[0, "#e0f5ee"], [0.5, "#0077cc"], [1, "#00916a"]]
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Space Mono", size=10),
                title_font=dict(family="Syne", size=14),
                xaxis=dict(gridcolor="rgba(128,128,128,0.08)", tickfont=dict(size=9)),
                yaxis=dict(gridcolor="rgba(128,128,128,0.08)"),
                coloraxis_showscale=False,
                margin=dict(t=50, b=30, l=10, r=10),
            )
            fig_bar.update_traces(textfont=dict(family="Space Mono", size=10), textposition="outside")
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Daily trend line ──
        st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        daily_trend = df_attendance.groupby('date').size().reset_index(name='Count')
        fig_line = px.line(
            daily_trend, x='date', y='Count',
            title="Daily Attendance Volume",
            markers=True,
            color_discrete_sequence=["#00916a"]
        )
        fig_line.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Space Mono", size=10),
            title_font=dict(family="Syne", size=14),
            xaxis=dict(gridcolor="rgba(128,128,128,0.08)"),
            yaxis=dict(gridcolor="rgba(128,128,128,0.08)"),
            margin=dict(t=50, b=30, l=10, r=10),
        )
        fig_line.update_traces(line=dict(width=2), marker=dict(size=6, color="#00916a", line=dict(color="rgba(0,0,0,0.15)", width=1.5)))
        st.plotly_chart(fig_line, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="empty-state" style="border:1px dashed var(--border);border-radius:12px;padding:4rem;">
            <div class="empty-state-icon">📊</div>
            Start taking attendance to generate analytics.
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  SECTION: DIRECTORY
# ═══════════════════════════════════════════════════════════════
elif st.session_state.section == "directory":

    st.markdown('<div class="panel-heading">Teacher Directory & Management</div>', unsafe_allow_html=True)

    col_form, col_cards = st.columns([1, 2])

    # ── Update form ──
    with col_form:
        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-family:\'Space Mono\',monospace;font-size:0.65rem;letter-spacing:2px;text-transform:uppercase;color:var(--accent);margin-bottom:1rem;">Update Teacher</div>', unsafe_allow_html=True)

        if not df_teachers.empty:
            with st.form("update_teacher_form"):
                teacher_list     = df_teachers['name'].tolist()
                selected_teacher = st.selectbox("Select Teacher to Update", teacher_list)
                updated_dept     = st.selectbox(
                    "Assign New Department",
                    ["Computer Science", "Mathematics", "Physics", "English", "Chemistry", "Biology"],
                    key="update_dept_select"
                )
                update_btn = st.form_submit_button("Update Department")

                if update_btn:
                    update_teacher_department(selected_teacher, updated_dept)
                    st.success(f"✓ {selected_teacher} → {updated_dept}")
                    time.sleep(1)
                    st.rerun()
        else:
            st.info("No teachers available.")

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Teacher cards ──
    with col_cards:
        if not df_teachers.empty:
            st.markdown(f'<div style="font-family:\'Space Mono\',monospace;font-size:0.65rem;letter-spacing:2px;text-transform:uppercase;color:var(--text-muted);margin-bottom:0.8rem;">{len(df_teachers)} registered teachers</div>', unsafe_allow_html=True)

            for _, row in df_teachers.iterrows():
                initials = "".join([w[0] for w in str(row['name']).split()[:2]]).upper()
                dept_badge_cls = "badge-blue"
                active = str(row.get('status', 'Active')).lower() == 'active'
                status_cls = "badge-green" if active else "badge-warn"
                status_lbl = row.get('status', 'Active')

                st.markdown(f"""
                <div class="teacher-card">
                    <div class="teacher-avatar">{initials}</div>
                    <div style="flex:1; min-width:0;">
                        <div class="teacher-name">{row['name']}</div>
                        <div class="teacher-dept">{row['department']}</div>
                    </div>
                    <div style="display:flex;flex-direction:column;align-items:flex-end;gap:0.35rem;">
                        <span class="badge {status_cls}">{status_lbl}</span>
                        <span class="teacher-id"># {row['id']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-state" style="border:1px dashed var(--border);border-radius:12px;">
                <div class="empty-state-icon">👤</div>
                Directory is empty.
            </div>""", unsafe_allow_html=True)
            