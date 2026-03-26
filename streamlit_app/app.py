import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2
import os
from collections import Counter

st.set_page_config(
    page_title="Autoshot — Vehicle Damage Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@200;300;400;500&display=swap');

* { box-sizing: border-box; }
html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #ffffff !important;
    color: #0a0a0a !important;
}
[data-testid="stAppViewContainer"] { background: #ffffff; }
[data-testid="stHeader"] { display: none; }
[data-testid="stToolbar"] { display: none; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* Header */
.as-header {
    background: #0a0a0a;
    padding: 20px 40px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0;
}
.as-logo { display: flex; align-items: center; gap: 14px; }
.as-name { font-size: 16px; font-weight: 300; letter-spacing: 5px; color: #fff; text-transform: uppercase; }
.as-sub { font-size: 8px; font-weight: 300; letter-spacing: 2px; color: #444; text-transform: uppercase; margin-top: 2px; }

/* Nav tabs */
.as-nav {
    display: flex;
    border-bottom: 1px solid #e8e8e8;
    background: #fff;
    padding: 0 40px;
}
.as-nav-tab {
    padding: 14px 0;
    margin-right: 32px;
    font-size: 9px;
    letter-spacing: 2px;
    color: #bbb;
    text-transform: uppercase;
    cursor: pointer;
    border-bottom: 2px solid transparent;
}
.as-nav-tab.active {
    color: #0a0a0a;
    border-bottom: 2px solid #cc0000;
}

/* Content */
.as-content { padding: 40px; }
.as-label { font-size: 9px; letter-spacing: 3px; color: #bbb; text-transform: uppercase; margin-bottom: 6px; }
.as-title { font-size: 24px; font-weight: 200; color: #0a0a0a; letter-spacing: -0.5px; margin-bottom: 6px; }
.as-body { font-size: 12px; font-weight: 300; color: #999; line-height: 1.6; margin-bottom: 32px; }

/* Angle grid */
.angle-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: #e8e8e8; margin-bottom: 24px; }
.angle-cell {
    background: #fff;
    padding: 20px;
    position: relative;
    cursor: pointer;
}
.angle-name { font-size: 10px; font-weight: 500; letter-spacing: 2px; color: #0a0a0a; text-transform: uppercase; margin-bottom: 4px; }
.angle-hint { font-size: 10px; font-weight: 300; color: #bbb; }
.angle-badge {
    position: absolute; top: 12px; right: 12px;
    font-size: 8px; letter-spacing: 1px; padding: 3px 8px; text-transform: uppercase;
}
.badge-req { background: #f5f5f3; color: #bbb; border: 1px solid #e8e8e8; }
.badge-done { background: #0a0a0a; color: #fff; }

/* Damage items */
.damage-item {
    display: flex; align-items: center; gap: 16px;
    padding: 14px 0; border-bottom: 1px solid #f5f5f5;
}
.damage-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.damage-name { font-size: 12px; font-weight: 400; color: #0a0a0a; text-transform: capitalize; }
.damage-loc { font-size: 10px; font-weight: 300; color: #bbb; margin-top: 2px; }
.conf-track { height: 1px; background: #f0f0f0; margin-top: 6px; width: 60px; }
.conf-fill { height: 1px; background: #cc0000; }

/* Cost section */
.cost-section { background: #f5f5f3; padding: 24px; margin: 24px 0; }
.cost-row { display: flex; justify-content: space-between; align-items: baseline; padding: 8px 0; }
.cost-label { font-size: 10px; font-weight: 300; letter-spacing: 1px; color: #888; text-transform: uppercase; }
.cost-val { font-size: 14px; font-weight: 300; color: #0a0a0a; }
.cost-val-red { font-size: 14px; font-weight: 300; color: #cc0000; }
.cost-divider { height: 1px; background: #e8e8e8; margin: 8px 0; }
.cost-total-label { font-size: 10px; font-weight: 500; letter-spacing: 2px; color: #0a0a0a; text-transform: uppercase; }
.cost-total-val { font-size: 24px; font-weight: 200; color: #cc0000; }

/* Breakdown dark box */
.breakdown {
    background: #0a0a0a; padding: 24px; margin: 16px 0;
}
.breakdown-row { display: flex; justify-content: space-between; align-items: baseline; padding: 6px 0; }
.breakdown-label { font-size: 9px; letter-spacing: 2px; color: #444; text-transform: uppercase; }
.breakdown-val { font-size: 13px; font-weight: 300; color: #fff; }
.breakdown-val-red { font-size: 13px; font-weight: 300; color: #cc0000; }
.breakdown-divider { height: 1px; background: #1e1e1e; margin: 10px 0; }
.breakdown-total-label { font-size: 9px; letter-spacing: 2px; color: #666; text-transform: uppercase; }
.breakdown-total-val { font-size: 28px; font-weight: 200; color: #fff; }

/* Metrics */
.metric-row { display: flex; gap: 1px; background: #e8e8e8; margin: 16px 0; }
.metric-box { flex: 1; background: #fff; padding: 20px; text-align: center; }
.metric-val { font-size: 28px; font-weight: 200; color: #0a0a0a; }
.metric-lbl { font-size: 8px; letter-spacing: 2px; color: #bbb; text-transform: uppercase; margin-top: 4px; }

/* Status */
.status-clean { color: #3a7d44; font-size: 13px; font-weight: 400; letter-spacing: 1px; }
.status-damaged { color: #cc0000; font-size: 13px; font-weight: 400; letter-spacing: 1px; }

/* Empty state */
.empty-state {
    background: #f5f5f3; padding: 48px; text-align: center;
    font-size: 12px; font-weight: 300; color: #bbb; letter-spacing: 1px;
}

/* Action buttons */
.action-row { display: flex; gap: 1px; margin-top: 16px; }
.action-primary {
    flex: 1; padding: 14px; background: #0a0a0a; color: #fff;
    font-family: 'Inter', sans-serif; font-size: 9px; letter-spacing: 3px;
    text-transform: uppercase; border: none; cursor: pointer;
}
.action-secondary {
    flex: 1; padding: 14px; background: #f5f5f3; color: #0a0a0a;
    font-family: 'Inter', sans-serif; font-size: 9px; letter-spacing: 3px;
    text-transform: uppercase; border: none; cursor: pointer;
}

/* Powered by */
.powered {
    text-align: center; padding: 20px;
    font-size: 8px; letter-spacing: 2px; color: #ddd; text-transform: uppercase;
    border-top: 1px solid #f5f5f3; margin-top: 40px;
}

/* Hide streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none; }
.stSelectbox label, .stSlider label, .stFileUploader label { 
    font-size: 9px !important; letter-spacing: 2px !important; 
    color: #bbb !important; text-transform: uppercase !important;
    font-family: 'Inter', sans-serif !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #0a0a0a !important;
}
div[data-baseweb="select"] {
    border: none !important;
    border-bottom: 1px solid #e8e8e8 !important;
    border-radius: 0 !important;
}
.stNumberInput input {
    border: none !important;
    border-bottom: 1px solid #e8e8e8 !important;
    border-radius: 0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 18px !important;
    font-weight: 200 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("AUTOSHOT_MODEL", "runs/detect/runs/detect/autoshot_combined_v12/weights/best.pt")
CLASS_NAMES = ['dent', 'scratch', 'crack', 'glass shatter', 'lamp broken', 'tire flat', 'paint damage']
CONF_THRESHOLD = 0.40

SEVERITY = {
    'dent': ('Medium', '#fb923c'),
    'scratch': ('Low', '#facc15'),
    'crack': ('High', '#f87171'),
    'glass shatter': ('Critical', '#cc0000'),
    'lamp broken': ('High', '#f87171'),
    'tire flat': ('Critical', '#cc0000'),
    'paint damage': ('Low', '#facc15'),
}

REPAIR_COSTS = {
    'es': {'dent':(150,400),'scratch':(80,220),'crack':(200,500),'glass shatter':(300,900),'lamp broken':(150,600),'tire flat':(80,200),'paint damage':(200,500)},
    'pt': {'dent':(130,370),'scratch':(70,200),'crack':(180,460),'glass shatter':(270,830),'lamp broken':(140,550),'tire flat':(75,185),'paint damage':(185,460)},
    'br': {'dent':(800,2200),'scratch':(400,1200),'crack':(1100,2700),'glass shatter':(1600,4900),'lamp broken':(800,3200),'tire flat':(400,1100),'paint damage':(1100,2700)},
    'us': {'dent':(160,430),'scratch':(85,240),'crack':(215,540),'glass shatter':(325,970),'lamp broken':(160,650),'tire flat':(85,215),'paint damage':(215,540)},
    'mx': {'dent':(2900,7800),'scratch':(1600,4300),'crack':(3900,9800),'glass shatter':(5800,17500),'lamp broken':(2900,11700),'tire flat':(1600,3900),'paint damage':(3900,9800)},
}
CURRENCY = {'es':'€','pt':'€','br':'R$','us':'$','mx':'MX$'}
COUNTRY_NAMES = {'es':'🇪🇸 Spain','pt':'🇵🇹 Portugal','br':'🇧🇷 Brazil','us':'🇺🇸 USA','mx':'🇲🇽 México'}

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return YOLO(MODEL_PATH)
    return None

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="as-header">
  <div class="as-logo">
    <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
      <circle cx="14" cy="14" r="13" stroke="#fff" stroke-width="0.75" opacity="0.4"/>
      <circle cx="14" cy="14" r="7" stroke="#fff" stroke-width="0.5" opacity="0.3"/>
      <circle cx="14" cy="14" r="2.5" fill="#cc0000"/>
      <line x1="14" y1="1" x2="14" y2="7" stroke="#cc0000" stroke-width="1.25"/>
      <line x1="21" y1="14" x2="27" y2="14" stroke="#cc0000" stroke-width="1.25"/>
      <line x1="14" y1="21" x2="14" y2="27" stroke="#fff" stroke-width="0.75" opacity="0.3"/>
      <line x1="1" y1="14" x2="7" y2="14" stroke="#fff" stroke-width="0.75" opacity="0.3"/>
    </svg>
    <div>
      <div class="as-name">Autoshot</div>
      <div class="as-sub">Vehicle damage intelligence</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["01 — Inspect", "02 — Report", "03 — Offer"])

model = load_model()

# ── TAB 1: INSPECT ────────────────────────────────────────────────────────────
with tab1:
    st.markdown("""
    <div class="as-content">
      <div class="as-label">Vehicle inspection</div>
      <div class="as-title">Photograph all angles</div>
      <div class="as-body">Upload photos from each angle for the most accurate assessment. Front and rear are required.</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="as-label" style="padding:0 0 8px 40px">Country</div>', unsafe_allow_html=True)
        country = st.selectbox("Country", list(COUNTRY_NAMES.keys()),
                               format_func=lambda x: COUNTRY_NAMES[x],
                               label_visibility="collapsed",
                               key="country")

    st.markdown('<div style="padding: 0 40px">', unsafe_allow_html=True)
    uploads = st.file_uploader(
        "Upload vehicle photos",
        type=["jpg","jpeg","png","webp"],
        accept_multiple_files=True,
        key="uploads"
    )
    conf = st.slider("Confidence threshold", 0.10, 0.90, CONF_THRESHOLD, 0.05, key="conf")
    st.markdown('</div>', unsafe_allow_html=True)

    if uploads:
        cols = st.columns(min(len(uploads), 3))
        for i, f in enumerate(uploads):
            with cols[i % 3]:
                st.image(Image.open(f), caption=f.name, use_container_width=True)

# ── TAB 2: REPORT ─────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="as-content">', unsafe_allow_html=True)
    st.markdown('<div class="as-label">Damage report</div>', unsafe_allow_html=True)

    if not uploads:
        st.markdown('<div class="empty-state">Upload photos in the Inspect tab to begin analysis</div>', unsafe_allow_html=True)
    elif model is None:
        st.error(f"Model not found at: {MODEL_PATH}")
    else:
        country = st.session_state.get("country", "es")
        conf = st.session_state.get("conf", CONF_THRESHOLD)
        sym = CURRENCY[country]

        all_classes, all_confs, annotated_imgs = [], [], []

        with st.spinner("Analysing damage..."):
            for f in uploads:
                img = Image.open(f).convert("RGB")
                img_array = np.array(img)
                results = model.predict(img_array, conf=conf, verbose=False)[0]

                if results.boxes and len(results.boxes) > 0:
                    all_classes += [CLASS_NAMES[int(c)] if int(c) < len(CLASS_NAMES) else f"class_{int(c)}"
                                    for c in results.boxes.cls.cpu().numpy()]
                    all_confs += list(results.boxes.conf.cpu().numpy())

                annotated = results.plot(line_width=2, pil=False, img=img_array.copy())
                annotated_imgs.append((f.name, cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)))

        # Show annotated images
        if annotated_imgs:
            img_cols = st.columns(min(len(annotated_imgs), 3))
            for i, (name, img) in enumerate(annotated_imgs):
                with img_cols[i % 3]:
                    st.image(img, caption=name, use_container_width=True)

        if not all_classes:
            st.markdown('<p class="status-clean">✓ No damage detected across all photos</p>', unsafe_allow_html=True)
        else:
            counts = Counter(all_classes)
            avg_conf = float(np.mean(all_confs))

            st.markdown(f'<p class="status-damaged">⚠ {len(all_classes)} damage instance(s) detected across {len(uploads)} photo(s)</p>', unsafe_allow_html=True)

            # Metrics
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-box"><div class="metric-val">{len(all_classes)}</div><div class="metric-lbl">Detections</div></div>
                <div class="metric-box"><div class="metric-val">{len(counts)}</div><div class="metric-lbl">Types</div></div>
                <div class="metric-box"><div class="metric-val">{avg_conf:.0%}</div><div class="metric-lbl">Avg confidence</div></div>
                <div class="metric-box"><div class="metric-val">{len(uploads)}</div><div class="metric-lbl">Photos</div></div>
            </div>
            """, unsafe_allow_html=True)

            # Damage breakdown + costs
            st.markdown('<div class="as-label" style="margin-top:24px">Repair estimate</div>', unsafe_allow_html=True)
            costs = REPAIR_COSTS[country]
            total_low, total_high = 0, 0

            cost_rows = ""
            for cls, count in sorted(counts.items()):
                sev, color = SEVERITY.get(cls, ('Medium','#fb923c'))
                low, high = costs.get(cls, (100,300))
                low_total, high_total = low*count, high*count
                total_low += low_total
                total_high += high_total
                cost_rows += f"""
                <div class="cost-row">
                    <span class="cost-label">{cls} ×{count}</span>
                    <span class="cost-val">{sym} {low_total:,} – {high_total:,}</span>
                </div>"""

            st.markdown(f"""
            <div class="cost-section">
                {cost_rows}
                <div class="cost-divider"></div>
                <div class="cost-row">
                    <span class="cost-total-label">Total estimate</span>
                    <span class="cost-total-val">{sym} {total_low:,} – {total_high:,}</span>
                </div>
            </div>
            <p style="font-size:10px;color:#bbb;font-weight:300">Prices indicative for {COUNTRY_NAMES[country]}. Labour rates vary by region.</p>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 3: OFFER ─────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="as-content">', unsafe_allow_html=True)
    st.markdown('<div class="as-label">Dealer offer calculator</div>', unsafe_allow_html=True)

    country = st.session_state.get("country", "es")
    sym = CURRENCY[country]

    col_l, col_r = st.columns(2)
    with col_l:
        market_val = st.number_input(f"Market value ({sym})", min_value=0, value=12000, step=500, key="market")
        margin = st.slider("Dealer margin (%)", 5, 30, 15, key="margin")

    # Get repair estimate from session if available
    repair_mid = 0
    if uploads and model is not None:
        country_costs = REPAIR_COSTS[country]
        # Use cached counts if available
        try:
            repair_mid = (total_low + total_high) // 2
        except:
            repair_mid = 0

    with col_r:
        repair_input = st.number_input(f"Repair estimate ({sym})", min_value=0, value=repair_mid, step=50, key="repair")

    margin_amt = (market_val - repair_input) * margin / 100
    max_offer = max(0, market_val - repair_input - margin_amt)

    st.markdown(f"""
    <div style="margin-top:8px">
        <div style="font-size:36px;font-weight:200;color:#0a0a0a;letter-spacing:-1px;margin-bottom:4px">
            Max offer
        </div>
        <div style="font-size:48px;font-weight:200;color:#cc0000;letter-spacing:-2px;line-height:1">
            {sym} {int(max_offer):,}
        </div>
        <div style="font-size:11px;font-weight:300;color:#bbb;letter-spacing:1px;margin-top:8px;margin-bottom:24px">
            Based on market value, repairs, and margin
        </div>
    </div>
    <div class="breakdown">
        <div class="breakdown-row">
            <span class="breakdown-label">Market value</span>
            <span class="breakdown-val">{sym} {market_val:,}</span>
        </div>
        <div class="breakdown-row">
            <span class="breakdown-label">Repair estimate</span>
            <span class="breakdown-val-red">— {sym} {int(repair_input):,}</span>
        </div>
        <div class="breakdown-row">
            <span class="breakdown-label">Dealer margin {margin}%</span>
            <span class="breakdown-val-red">— {sym} {int(margin_amt):,}</span>
        </div>
        <div class="breakdown-divider"></div>
        <div class="breakdown-row">
            <span class="breakdown-total-label">Max offer</span>
            <span class="breakdown-total-val">{sym} {int(max_offer):,}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<div class="powered">Autoshot · Vehicle damage intelligence</div>', unsafe_allow_html=True)
