import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2
import os
from collections import Counter

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Autoshot",
    page_icon="⊙",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@200;300;400;500;600&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body { font-family: 'Inter', sans-serif; background: #f8f7f5; color: #0a0a0a; }

[data-testid="stAppViewContainer"] { background: #f8f7f5 !important; }
[data-testid="stHeader"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
#MainMenu, footer { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
.stTabs [data-baseweb="tab-list"] { display: none !important; }

/* ── Header ── */
.as-header {
    background: #0a0a0a;
    padding: 0 48px;
    height: 64px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
}
.as-logo { display: flex; align-items: center; gap: 16px; text-decoration: none; }
.as-wordmark { display: flex; flex-direction: column; gap: 1px; }
.as-name {
    font-size: 13px; font-weight: 400; letter-spacing: 6px;
    color: #fff; text-transform: uppercase; line-height: 1;
}
.as-tagline { font-size: 8px; font-weight: 300; letter-spacing: 2px; color: #3a3a3a; text-transform: uppercase; }
.as-header-right { display: flex; align-items: center; gap: 8px; }
.as-country-label {
    font-size: 8px; letter-spacing: 2px; color: #3a3a3a;
    text-transform: uppercase; margin-right: 4px;
}

/* ── Nav ── */
.as-nav {
    background: #fff;
    border-bottom: 1px solid #ebebeb;
    padding: 0 48px;
    display: flex;
    align-items: center;
    gap: 0;
}
.as-nav-item {
    font-size: 9px; font-weight: 400; letter-spacing: 2px;
    color: #c0c0c0; text-transform: uppercase;
    padding: 16px 24px 14px;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: all 0.15s ease;
    user-select: none;
}
.as-nav-item:hover { color: #666; }
.as-nav-item.active { color: #0a0a0a; border-bottom-color: #cc0000; }
.as-nav-divider { width: 1px; height: 14px; background: #ebebeb; margin: 0 4px; }

/* ── Page wrapper ── */
.as-page { padding: 48px; min-height: calc(100vh - 120px); }
.as-page-narrow { max-width: 960px; }

/* ── Section headers ── */
.as-eyebrow { font-size: 9px; font-weight: 400; letter-spacing: 3px; color: #c0c0c0; text-transform: uppercase; margin-bottom: 8px; }
.as-h1 { font-size: 28px; font-weight: 200; color: #0a0a0a; letter-spacing: -0.5px; line-height: 1.2; margin-bottom: 8px; }
.as-h2 { font-size: 18px; font-weight: 300; color: #0a0a0a; letter-spacing: -0.3px; margin-bottom: 16px; }
.as-body { font-size: 12px; font-weight: 300; color: #888; line-height: 1.7; max-width: 480px; }
.as-rule { height: 1px; background: #ebebeb; margin: 32px 0; }

/* ── Upload zone ── */
.as-upload-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 1px;
    background: #ebebeb;
    border: 1px solid #ebebeb;
    margin: 24px 0;
}
.as-upload-cell {
    background: #fff;
    padding: 20px 16px;
    text-align: center;
    cursor: pointer;
    transition: background 0.15s;
    min-height: 100px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 8px;
}
.as-upload-cell:hover { background: #fafafa; }
.as-upload-cell.required { }
.as-upload-cell.optional { background: #fefefe; }
.as-upload-cell.done { background: #f8f7f5; }
.as-angle-icon { opacity: 0.3; }
.as-upload-cell.done .as-angle-icon { opacity: 1; }
.as-upload-name { font-size: 9px; font-weight: 500; letter-spacing: 2px; color: #0a0a0a; text-transform: uppercase; }
.as-upload-hint { font-size: 9px; font-weight: 300; color: #c0c0c0; }
.as-upload-status {
    font-size: 8px; letter-spacing: 1px; padding: 2px 8px; text-transform: uppercase;
    margin-top: 4px;
}
.as-status-req { background: #f5f5f3; color: #c0c0c0; border: 1px solid #e8e8e8; }
.as-status-opt { background: #fff; color: #ddd; border: 1px solid #ebebeb; }
.as-status-done { background: #0a0a0a; color: #fff; }

/* ── Progress ── */
.as-progress-wrap { margin: 0 0 24px; }
.as-progress-bar { height: 1px; background: #ebebeb; }
.as-progress-fill { height: 1px; background: #cc0000; transition: width 0.4s ease; }
.as-progress-label { font-size: 9px; font-weight: 300; color: #c0c0c0; letter-spacing: 1px; margin-top: 8px; }

/* ── CTA Button ── */
.as-btn-primary {
    display: inline-block;
    background: #0a0a0a; color: #fff;
    font-family: 'Inter', sans-serif;
    font-size: 9px; font-weight: 400; letter-spacing: 3px;
    text-transform: uppercase;
    padding: 14px 32px;
    border: none; cursor: pointer;
    transition: background 0.15s;
    text-align: center;
}
.as-btn-primary:hover { background: #1a1a1a; }
.as-btn-secondary {
    display: inline-block;
    background: #f5f5f3; color: #0a0a0a;
    font-family: 'Inter', sans-serif;
    font-size: 9px; font-weight: 400; letter-spacing: 3px;
    text-transform: uppercase;
    padding: 14px 32px;
    border: none; cursor: pointer;
}

/* ── Damage report ── */
.as-damage-list { margin: 24px 0; }
.as-damage-item {
    display: flex; align-items: center; gap: 16px;
    padding: 16px 0;
    border-bottom: 1px solid #f5f5f5;
}
.as-damage-item:last-child { border-bottom: none; }
.as-damage-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
.as-damage-info { flex: 1; }
.as-damage-name { font-size: 13px; font-weight: 400; color: #0a0a0a; text-transform: capitalize; }
.as-damage-meta { font-size: 10px; font-weight: 300; color: #c0c0c0; margin-top: 2px; }
.as-damage-right { text-align: right; min-width: 80px; }
.as-damage-sev { font-size: 8px; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 6px; }
.as-conf-bar { height: 1px; background: #ebebeb; width: 64px; margin-left: auto; }
.as-conf-fill { height: 1px; background: #cc0000; }
.as-conf-pct { font-size: 10px; font-weight: 300; color: #c0c0c0; margin-top: 4px; }

/* ── Metrics strip ── */
.as-metrics {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: #ebebeb;
    border: 1px solid #ebebeb;
    margin: 24px 0;
}
.as-metric {
    background: #fff;
    padding: 20px;
    text-align: center;
}
.as-metric-val { font-size: 32px; font-weight: 200; color: #0a0a0a; line-height: 1; }
.as-metric-val-red { font-size: 32px; font-weight: 200; color: #cc0000; line-height: 1; }
.as-metric-label { font-size: 8px; font-weight: 400; letter-spacing: 2px; color: #c0c0c0; text-transform: uppercase; margin-top: 6px; }

/* ── Cost table ── */
.as-cost-table { margin: 24px 0; }
.as-cost-row {
    display: flex; justify-content: space-between; align-items: baseline;
    padding: 12px 0; border-bottom: 1px solid #f5f5f5;
}
.as-cost-row:last-child { border-bottom: none; }
.as-cost-label { font-size: 10px; font-weight: 300; letter-spacing: 1px; color: #888; text-transform: uppercase; }
.as-cost-val { font-size: 14px; font-weight: 300; color: #0a0a0a; }
.as-cost-val-accent { font-size: 14px; font-weight: 300; color: #cc0000; }
.as-cost-total {
    display: flex; justify-content: space-between; align-items: baseline;
    padding: 20px 0 0;
    border-top: 1px solid #0a0a0a;
    margin-top: 8px;
}
.as-cost-total-label { font-size: 10px; font-weight: 500; letter-spacing: 3px; color: #0a0a0a; text-transform: uppercase; }
.as-cost-total-val { font-size: 28px; font-weight: 200; color: #cc0000; }
.as-country-note { font-size: 10px; font-weight: 300; color: #c0c0c0; margin-top: 12px; letter-spacing: 0.5px; }

/* ── Offer screen ── */
.as-offer-hero { margin-bottom: 40px; }
.as-offer-label { font-size: 12px; font-weight: 300; color: #888; letter-spacing: 1px; margin-bottom: 4px; }
.as-offer-amount { font-size: 64px; font-weight: 200; color: #0a0a0a; letter-spacing: -2px; line-height: 1; }
.as-offer-amount span { color: #cc0000; }
.as-offer-sub { font-size: 11px; font-weight: 300; color: #c0c0c0; letter-spacing: 1px; margin-top: 12px; }

.as-calculator { margin: 32px 0; }
.as-calc-label { font-size: 9px; font-weight: 400; letter-spacing: 2px; color: #c0c0c0; text-transform: uppercase; margin-bottom: 8px; display: block; }

.as-breakdown {
    background: #0a0a0a;
    padding: 28px 32px;
    margin: 24px 0;
}
.as-breakdown-row {
    display: flex; justify-content: space-between; align-items: baseline;
    padding: 8px 0;
}
.as-breakdown-label { font-size: 9px; letter-spacing: 2px; color: #3a3a3a; text-transform: uppercase; }
.as-breakdown-val { font-size: 14px; font-weight: 300; color: #fff; }
.as-breakdown-val-red { font-size: 14px; font-weight: 300; color: #cc0000; }
.as-breakdown-divider { height: 1px; background: #1a1a1a; margin: 12px 0; }
.as-breakdown-total-label { font-size: 9px; letter-spacing: 2px; color: #555; text-transform: uppercase; }
.as-breakdown-total-val { font-size: 36px; font-weight: 200; color: #fff; letter-spacing: -1px; }

/* ── Empty state ── */
.as-empty {
    background: #fff; border: 1px solid #ebebeb;
    padding: 64px 32px; text-align: center;
}
.as-empty-icon { font-size: 32px; margin-bottom: 16px; opacity: 0.2; }
.as-empty-text { font-size: 12px; font-weight: 300; color: #c0c0c0; letter-spacing: 1px; }

/* ── Status badges ── */
.as-status-clean { color: #2d7d3a; font-size: 12px; font-weight: 400; letter-spacing: 1px; }
.as-status-damaged { color: #cc0000; font-size: 12px; font-weight: 400; letter-spacing: 1px; }

/* ── White label strip ── */
.as-wl-bar {
    background: #f8f7f5; border-top: 1px solid #ebebeb;
    padding: 12px 48px;
    display: flex; align-items: center; justify-content: space-between;
}
.as-wl-dealer { font-size: 12px; font-weight: 300; letter-spacing: 4px; color: #0a0a0a; text-transform: uppercase; }
.as-wl-powered { font-size: 8px; letter-spacing: 2px; color: #c0c0c0; text-transform: uppercase; }

/* ── Footer ── */
.as-footer {
    border-top: 1px solid #ebebeb;
    padding: 20px 48px;
    display: flex; align-items: center; justify-content: space-between;
    background: #fff;
}
.as-footer-left { font-size: 8px; letter-spacing: 2px; color: #c0c0c0; text-transform: uppercase; }
.as-footer-right { font-size: 8px; letter-spacing: 2px; color: #c0c0c0; text-transform: uppercase; }

/* ── Streamlit overrides ── */
.stFileUploader { margin: 0 !important; }
.stFileUploader > div { 
    background: #fff !important;
    border: 1px dashed #ebebeb !important;
    border-radius: 0 !important;
    padding: 20px !important;
}
.stFileUploader label { display: none !important; }
.stSlider { padding: 0 !important; }
.stSlider label { 
    font-size: 9px !important; font-weight: 400 !important;
    letter-spacing: 2px !important; color: #c0c0c0 !important;
    text-transform: uppercase !important;
    font-family: 'Inter', sans-serif !important;
}
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: #0a0a0a !important;
    border-color: #0a0a0a !important;
}
.stSelectbox label {
    font-size: 9px !important; font-weight: 400 !important;
    letter-spacing: 2px !important; color: #c0c0c0 !important;
    text-transform: uppercase !important;
}
div[data-baseweb="select"] > div {
    border: none !important;
    border-bottom: 1px solid #ebebeb !important;
    border-radius: 0 !important;
    background: transparent !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 300 !important;
}
.stNumberInput label {
    font-size: 9px !important; font-weight: 400 !important;
    letter-spacing: 2px !important; color: #c0c0c0 !important;
    text-transform: uppercase !important;
}
.stNumberInput input {
    border: none !important;
    border-bottom: 1px solid #ebebeb !important;
    border-radius: 0 !important;
    background: transparent !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 18px !important;
    font-weight: 200 !important;
    padding: 8px 0 !important;
}
.stSpinner > div { border-top-color: #cc0000 !important; }
[data-testid="stImage"] { margin: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get(
    "AUTOSHOT_MODEL",
    "runs/detect/runs/detect/autoshot_combined_v12/weights/best.pt"
)
FALLBACK_MODEL = "runs/detect/runs/detect/autoshot_v4_final/weights/best.pt"

CLASS_NAMES = ['dent', 'scratch', 'crack', 'glass shatter', 'lamp broken', 'tire flat', 'paint damage']
CONF_DEFAULT = 0.40

SEVERITY = {
    'dent':         ('Medium', '#fb923c'),
    'scratch':      ('Low',    '#d4a843'),
    'crack':        ('High',   '#f87171'),
    'glass shatter':('Critical','#cc0000'),
    'lamp broken':  ('High',   '#f87171'),
    'tire flat':    ('Critical','#cc0000'),
    'paint damage': ('Low',    '#d4a843'),
}

REPAIR_COSTS = {
    'es': {'dent':(150,400),'scratch':(80,220),'crack':(200,500),'glass shatter':(300,900),'lamp broken':(150,600),'tire flat':(80,200),'paint damage':(200,500)},
    'pt': {'dent':(130,370),'scratch':(70,200),'crack':(180,460),'glass shatter':(270,830),'lamp broken':(140,550),'tire flat':(75,185),'paint damage':(185,460)},
    'br': {'dent':(800,2200),'scratch':(400,1200),'crack':(1100,2700),'glass shatter':(1600,4900),'lamp broken':(800,3200),'tire flat':(400,1100),'paint damage':(1100,2700)},
    'us': {'dent':(160,430),'scratch':(85,240),'crack':(215,540),'glass shatter':(325,970),'lamp broken':(160,650),'tire flat':(85,215),'paint damage':(215,540)},
    'mx': {'dent':(2900,7800),'scratch':(1600,4300),'crack':(3900,9800),'glass shatter':(5800,17500),'lamp broken':(2900,11700),'tire flat':(1600,3900),'paint damage':(3900,9800)},
}

CURRENCY    = {'es':'€','pt':'€','br':'R$','us':'$','mx':'MX$'}
COUNTRY_NAMES = {
    'es':'🇪🇸  Spain','pt':'🇵🇹  Portugal',
    'br':'🇧🇷  Brazil','us':'🇺🇸  USA','mx':'🇲🇽  México'
}

ANGLES = [
    ('front',  'Front',       'Bumper & grille visible',  True),
    ('rear',   'Rear',        'Boot & rear bumper',       True),
    ('left',   'Left side',   'Full profile, doors visible', True),
    ('right',  'Right side',  'Full profile, doors visible', True),
    ('overhead','Overhead',   'Roof — improves accuracy +18%', False),
]

# ── Session state ──────────────────────────────────────────────────────────────
if 'screen' not in st.session_state:
    st.session_state.screen = 'inspect'
if 'wl_name' not in st.session_state:
    st.session_state.wl_name = ''
if 'results_cache' not in st.session_state:
    st.session_state.results_cache = None

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    for path in [MODEL_PATH, FALLBACK_MODEL]:
        if os.path.exists(path):
            return YOLO(path), path
    return None, None

model, model_path = load_model()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="as-header">
  <div class="as-logo">
    <svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="16" cy="16" r="15" stroke="white" stroke-width="0.75" opacity="0.25"/>
      <circle cx="16" cy="16" r="8"  stroke="white" stroke-width="0.5"  opacity="0.2"/>
      <circle cx="16" cy="16" r="3"  fill="#cc0000"/>
      <line x1="16" y1="1"  x2="16" y2="8"  stroke="#cc0000" stroke-width="1.5"/>
      <line x1="24" y1="16" x2="31" y2="16" stroke="#cc0000" stroke-width="1.5"/>
      <line x1="16" y1="24" x2="16" y2="31" stroke="white" stroke-width="0.75" opacity="0.25"/>
      <line x1="1"  y1="16" x2="8"  y2="16" stroke="white" stroke-width="0.75" opacity="0.25"/>
    </svg>
    <div class="as-wordmark">
      <div class="as-name">Autoshot</div>
      <div class="as-tagline">Vehicle damage intelligence</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Nav ────────────────────────────────────────────────────────────────────────
screens = ['inspect', 'report', 'offer']
labels  = ['01 — Inspect', '02 — Report', '03 — Offer']
nav_html = '<div class="as-nav">'
for s, l in zip(screens, labels):
    active = 'active' if st.session_state.screen == s else ''
    nav_html += f'<div class="as-nav-item {active}" onclick="void(0)">{l}</div>'
nav_html += '</div>'
st.markdown(nav_html, unsafe_allow_html=True)

# Navigation via Streamlit buttons (invisible, synced with nav)
nav_cols = st.columns(3)
with nav_cols[0]:
    if st.button("01 — Inspect", use_container_width=True, key="nav_inspect"):
        st.session_state.screen = 'inspect'
        st.rerun()
with nav_cols[1]:
    if st.button("02 — Report", use_container_width=True, key="nav_report"):
        st.session_state.screen = 'report'
        st.rerun()
with nav_cols[2]:
    if st.button("03 — Offer", use_container_width=True, key="nav_offer"):
        st.session_state.screen = 'offer'
        st.rerun()

st.markdown("""
<style>
[data-testid="stHorizontalBlock"] button {
    background: transparent !important;
    color: transparent !important;
    border: none !important;
    padding: 0 !important;
    height: 0 !important;
    min-height: 0 !important;
    overflow: hidden !important;
    position: absolute !important;
    pointer-events: none !important;
}
</style>
""", unsafe_allow_html=True)

# ── SCREEN: INSPECT ────────────────────────────────────────────────────────────
if st.session_state.screen == 'inspect':
    st.markdown('<div class="as-page">', unsafe_allow_html=True)
    st.markdown("""
    <div class="as-eyebrow">Step 01</div>
    <div class="as-h1">Vehicle inspection</div>
    <div class="as-body">Upload photos from each angle below for the most accurate damage assessment. Front, rear, left and right are required.</div>
    <div class="as-rule"></div>
    """, unsafe_allow_html=True)

    col_settings, col_spacer = st.columns([1, 2])
    with col_settings:
        country = st.selectbox(
            "Market",
            list(COUNTRY_NAMES.keys()),
            format_func=lambda x: COUNTRY_NAMES[x],
            key="country"
        )

    # File uploader
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    uploads = st.file_uploader(
        "Upload vehicle photos",
        type=["jpg","jpeg","png","webp"],
        accept_multiple_files=True,
        key="uploads",
        help="Upload up to 5 photos — front, rear, left side, right side, overhead"
    )

    conf = st.slider(
        "Detection confidence",
        min_value=0.10, max_value=0.90,
        value=CONF_DEFAULT, step=0.05,
        key="conf",
        help="Lower = more detections. Higher = fewer but more certain."
    )

    # Show uploaded images
    if uploads:
        n = len(uploads)
        progress = min(n / 4, 1.0)
        st.markdown(f"""
        <div class="as-progress-wrap">
            <div class="as-progress-bar">
                <div class="as-progress-fill" style="width:{progress*100:.0f}%"></div>
            </div>
            <div class="as-progress-label">{n} photo{'s' if n>1 else ''} uploaded · {'Ready to analyse' if n>=4 else f'{4-n} more required angle{"s" if 4-n>1 else ""}'}</div>
        </div>
        """, unsafe_allow_html=True)

        cols = st.columns(min(n, 5))
        for i, f in enumerate(uploads):
            with cols[i]:
                st.image(Image.open(f), caption=f.name[:20], use_container_width=True)

        st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

        col_btn, col_spacer = st.columns([1, 3])
        with col_btn:
            if st.button("Analyse damage →", key="analyse_btn", use_container_width=True):
                st.session_state.screen = 'report'
                st.session_state.results_cache = None
                st.rerun()
    else:
        st.markdown("""
        <div class="as-empty">
            <div class="as-empty-icon">⊙</div>
            <div class="as-empty-text">Upload vehicle photos to begin inspection</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── SCREEN: REPORT ─────────────────────────────────────────────────────────────
elif st.session_state.screen == 'report':
    st.markdown('<div class="as-page">', unsafe_allow_html=True)
    st.markdown("""
    <div class="as-eyebrow">Step 02</div>
    <div class="as-h1">Damage assessment</div>
    """, unsafe_allow_html=True)

    uploads = st.session_state.get("uploads", None)
    country = st.session_state.get("country", "es")
    conf    = st.session_state.get("conf", CONF_DEFAULT)
    sym     = CURRENCY[country]

    if not uploads:
        st.markdown("""
        <div class="as-empty">
            <div class="as-empty-icon">⊙</div>
            <div class="as-empty-text">No photos uploaded — go back to Inspect</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("← Back to Inspect", key="back_inspect"):
            st.session_state.screen = 'inspect'
            st.rerun()
    elif model is None:
        st.error(f"⚠ Model not found. Expected: {MODEL_PATH}")
    else:
        # Run inference (cached)
        if st.session_state.results_cache is None:
            all_classes, all_confs, annotated_imgs = [], [], []
            with st.spinner("Analysing damage across all photos..."):
                for f in uploads:
                    img = Image.open(f).convert("RGB")
                    arr = np.array(img)
                    res = model.predict(arr, conf=conf, verbose=False)[0]
                    if res.boxes and len(res.boxes) > 0:
                        all_classes += [
                            CLASS_NAMES[int(c)] if int(c) < len(CLASS_NAMES) else f"class_{int(c)}"
                            for c in res.boxes.cls.cpu().numpy()
                        ]
                        all_confs += list(res.boxes.conf.cpu().numpy())
                    ann = res.plot(line_width=2, pil=False, img=arr.copy())
                    annotated_imgs.append((f.name, cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)))
            st.session_state.results_cache = (all_classes, all_confs, annotated_imgs)
        else:
            all_classes, all_confs, annotated_imgs = st.session_state.results_cache

        # Annotated images
        if annotated_imgs:
            img_cols = st.columns(min(len(annotated_imgs), 5))
            for i, (name, img) in enumerate(annotated_imgs):
                with img_cols[i]:
                    st.image(img, caption=name[:20], use_container_width=True)

        st.markdown('<div class="as-rule"></div>', unsafe_allow_html=True)

        if not all_classes:
            st.markdown('<p class="as-status-clean">✓ No damage detected across all photos</p>', unsafe_allow_html=True)
        else:
            counts   = Counter(all_classes)
            avg_conf = float(np.mean(all_confs))
            n_photos = len(uploads)

            st.markdown(f'<p class="as-status-damaged">⚠ {len(all_classes)} damage instance{"s" if len(all_classes)>1 else ""} detected across {n_photos} photo{"s" if n_photos>1 else ""}</p>', unsafe_allow_html=True)

            # Metrics
            st.markdown(f"""
            <div class="as-metrics">
                <div class="as-metric">
                    <div class="as-metric-val-red">{len(all_classes)}</div>
                    <div class="as-metric-label">Detections</div>
                </div>
                <div class="as-metric">
                    <div class="as-metric-val">{len(counts)}</div>
                    <div class="as-metric-label">Damage types</div>
                </div>
                <div class="as-metric">
                    <div class="as-metric-val">{avg_conf:.0%}</div>
                    <div class="as-metric-label">Avg confidence</div>
                </div>
                <div class="as-metric">
                    <div class="as-metric-val">{n_photos}</div>
                    <div class="as-metric-label">Photos analysed</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Damage list
            st.markdown('<div class="as-h2">Damage breakdown</div>', unsafe_allow_html=True)
            damage_html = '<div class="as-damage-list">'
            costs_data  = REPAIR_COSTS[country]
            total_low, total_high = 0, 0

            # Group confs by class
            class_confs = {}
            for cls, c in zip(all_classes, all_confs):
                class_confs.setdefault(cls, []).append(c)

            for cls, count in sorted(counts.items(), key=lambda x: -max(class_confs[x[0]])):
                sev, color = SEVERITY.get(cls, ('Medium', '#fb923c'))
                max_conf   = max(class_confs[cls])
                bar_w      = int(max_conf * 64)
                low, high  = costs_data.get(cls, (100, 300))
                total_low  += low  * count
                total_high += high * count
                damage_html += f"""
                <div class="as-damage-item">
                    <div class="as-damage-dot" style="background:{color}"></div>
                    <div class="as-damage-info">
                        <div class="as-damage-name">{cls} <span style="color:#c0c0c0;font-size:11px;font-weight:300">×{count}</span></div>
                        <div class="as-damage-meta">{sev} severity · {sym} {low*count:,}–{high*count:,} est.</div>
                    </div>
                    <div class="as-damage-right">
                        <div class="as-damage-sev" style="color:{color}">{sev}</div>
                        <div class="as-conf-bar"><div class="as-conf-fill" style="width:{bar_w}px"></div></div>
                        <div class="as-conf-pct">{max_conf:.0%}</div>
                    </div>
                </div>"""
            damage_html += '</div>'
            st.markdown(damage_html, unsafe_allow_html=True)

            # Cost summary
            st.markdown('<div class="as-rule"></div>', unsafe_allow_html=True)
            st.markdown('<div class="as-h2">Repair estimate</div>', unsafe_allow_html=True)

            cost_rows = ""
            for cls, count in sorted(counts.items()):
                low, high = costs_data.get(cls, (100, 300))
                cost_rows += f"""
                <div class="as-cost-row">
                    <span class="as-cost-label">{cls} ×{count}</span>
                    <span class="as-cost-val">{sym} {low*count:,} – {high*count:,}</span>
                </div>"""

            st.markdown(f"""
            <div class="as-cost-table">
                {cost_rows}
                <div class="as-cost-total">
                    <span class="as-cost-total-label">Total estimate</span>
                    <span class="as-cost-total-val">{sym} {total_low:,}–{total_high:,}</span>
                </div>
            </div>
            <div class="as-country-note">Prices indicative for {COUNTRY_NAMES[country].strip()}. Labour rates vary by region.</div>
            """, unsafe_allow_html=True)

            # Store repair for offer screen
            st.session_state['repair_low']  = total_low
            st.session_state['repair_high'] = total_high

        st.markdown('<div style="height:32px"></div>', unsafe_allow_html=True)
        col_back, col_next, col_spacer = st.columns([1, 1, 2])
        with col_back:
            if st.button("← Re-inspect", key="back_btn", use_container_width=True):
                st.session_state.screen = 'inspect'
                st.rerun()
        with col_next:
            if st.button("Calculate offer →", key="offer_btn", use_container_width=True):
                st.session_state.screen = 'offer'
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ── SCREEN: OFFER ──────────────────────────────────────────────────────────────
elif st.session_state.screen == 'offer':
    st.markdown('<div class="as-page">', unsafe_allow_html=True)
    st.markdown("""
    <div class="as-eyebrow">Step 03</div>
    <div class="as-h1">Dealer offer calculator</div>
    """, unsafe_allow_html=True)

    country = st.session_state.get("country", "es")
    sym     = CURRENCY[country]
    repair_low  = st.session_state.get('repair_low', 0)
    repair_high = st.session_state.get('repair_high', 0)
    repair_mid  = (repair_low + repair_high) // 2

    st.markdown('<div class="as-rule"></div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1], gap="large")
    with col_l:
        market_val = st.number_input(
            f"Market value ({sym})",
            min_value=0, value=12000, step=500, key="market_val"
        )
        margin = st.slider(
            "Dealer margin",
            min_value=5, max_value=30, value=15, step=1,
            key="margin", format="%d%%"
        )

    with col_r:
        repair_input = st.number_input(
            f"Repair estimate ({sym})",
            min_value=0, value=repair_mid, step=50, key="repair_input",
            help="Pre-filled from damage report. Adjust if needed."
        )
        st.markdown(f'<div style="font-size:10px;color:#c0c0c0;margin-top:-8px">Range from report: {sym} {repair_low:,} – {repair_high:,}</div>', unsafe_allow_html=True)

    # Calculate
    margin_amt = (market_val - repair_input) * margin / 100
    max_offer  = max(0, market_val - repair_input - margin_amt)

    # Hero number
    st.markdown(f"""
    <div class="as-offer-hero" style="margin-top:32px">
        <div class="as-offer-label">Maximum purchase offer</div>
        <div class="as-offer-amount"><span>{sym} {int(max_offer):,}</span></div>
        <div class="as-offer-sub">Based on market value · repair costs · {margin}% dealer margin</div>
    </div>
    """, unsafe_allow_html=True)

    # Breakdown
    st.markdown(f"""
    <div class="as-breakdown">
        <div class="as-breakdown-row">
            <span class="as-breakdown-label">Market value</span>
            <span class="as-breakdown-val">{sym} {market_val:,}</span>
        </div>
        <div class="as-breakdown-row">
            <span class="as-breakdown-label">Repair estimate</span>
            <span class="as-breakdown-val-red">— {sym} {int(repair_input):,}</span>
        </div>
        <div class="as-breakdown-row">
            <span class="as-breakdown-label">Dealer margin {margin}%</span>
            <span class="as-breakdown-val-red">— {sym} {int(margin_amt):,}</span>
        </div>
        <div class="as-breakdown-divider"></div>
        <div class="as-breakdown-row">
            <span class="as-breakdown-total-label">Max offer</span>
            <span class="as-breakdown-total-val">{sym} {int(max_offer):,}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_back, col_export, col_spacer = st.columns([1, 1, 2])
    with col_back:
        if st.button("← Back to report", key="back_report", use_container_width=True):
            st.session_state.screen = 'report'
            st.rerun()
    with col_export:
        st.button("Export PDF", key="export_btn", use_container_width=True, disabled=True)

    # White label
    st.markdown('<div class="as-rule"></div>', unsafe_allow_html=True)
    wl_col, wl_spacer = st.columns([2, 2])
    with wl_col:
        wl_name = st.text_input(
            "White label — dealer name",
            value=st.session_state.wl_name,
            placeholder="e.g. Motorvip Barcelona",
            key="wl_input"
        )
        st.session_state.wl_name = wl_name

    if wl_name:
        st.markdown(f"""
        <div class="as-wl-bar">
            <div class="as-wl-dealer">{wl_name}</div>
            <div class="as-wl-powered">Powered by Autoshot</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
model_name = os.path.basename(os.path.dirname(os.path.dirname(model_path))) if model_path else "No model"
st.markdown(f"""
<div class="as-footer">
    <div class="as-footer-left">Autoshot · Vehicle damage intelligence</div>
    <div class="as-footer-right">Model: {model_name} · Early access</div>
</div>
""", unsafe_allow_html=True)
