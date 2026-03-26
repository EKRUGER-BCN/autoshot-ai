import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2
import os
from collections import Counter

st.set_page_config(
    page_title="Autoshot",
    page_icon="⊙",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@200;300;400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif !important;
    background: #f8f7f5 !important;
    color: #0a0a0a !important;
}
[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], #MainMenu, footer { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
[data-testid="stSidebar"] { display: none !important; }

/* Header */
.as-header {
    background: #0a0a0a; padding: 0 48px; height: 64px;
    display: flex; align-items: center; justify-content: space-between;
}
.as-name { font-size: 13px; font-weight: 400; letter-spacing: 6px; color: #fff; text-transform: uppercase; }
.as-tagline { font-size: 8px; font-weight: 300; letter-spacing: 2px; color: #3a3a3a; text-transform: uppercase; margin-top: 2px; }

/* Layout */
.as-main { display: grid; grid-template-columns: 380px 1fr; min-height: calc(100vh - 64px); }
.as-sidebar { background: #fff; border-right: 1px solid #ebebeb; padding: 40px 32px; }
.as-content { background: #f8f7f5; padding: 40px; }

/* Typography */
.as-section { font-size: 9px; font-weight: 400; letter-spacing: 3px; color: #c0c0c0; text-transform: uppercase; margin-bottom: 20px; padding-bottom: 12px; border-bottom: 1px solid #f0f0f0; }
.as-h2 { font-size: 20px; font-weight: 200; color: #0a0a0a; letter-spacing: -0.3px; margin-bottom: 24px; }

/* Status */
.as-clean { background: #f0faf2; border: 1px solid #c3e6cb; padding: 16px 20px; color: #2d7d3a; font-size: 12px; font-weight: 400; letter-spacing: 0.5px; margin-bottom: 24px; }
.as-damaged { background: #fff5f5; border: 1px solid #fcc; padding: 16px 20px; color: #cc0000; font-size: 12px; font-weight: 400; letter-spacing: 0.5px; margin-bottom: 24px; }

/* Metrics */
.as-metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px; background: #ebebeb; margin-bottom: 28px; }
.as-metric { background: #fff; padding: 16px; text-align: center; }
.as-metric-val { font-size: 28px; font-weight: 200; color: #0a0a0a; line-height: 1; }
.as-metric-val-red { font-size: 28px; font-weight: 200; color: #cc0000; line-height: 1; }
.as-metric-lbl { font-size: 8px; font-weight: 400; letter-spacing: 2px; color: #c0c0c0; text-transform: uppercase; margin-top: 4px; }

/* Damage list */
.as-damage-item { display: flex; align-items: center; gap: 12px; padding: 12px 0; border-bottom: 1px solid #f5f5f5; }
.as-damage-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
.as-damage-name { font-size: 13px; font-weight: 400; color: #0a0a0a; text-transform: capitalize; flex: 1; }
.as-damage-count { font-size: 11px; font-weight: 300; color: #c0c0c0; }
.as-damage-sev { font-size: 8px; letter-spacing: 1px; text-transform: uppercase; padding: 2px 8px; }
.as-damage-cost { font-size: 12px; font-weight: 300; color: #0a0a0a; min-width: 100px; text-align: right; }

/* Cost total */
.as-total { display: flex; justify-content: space-between; align-items: baseline; padding: 16px 0 0; margin-top: 8px; border-top: 1px solid #0a0a0a; }
.as-total-label { font-size: 9px; font-weight: 500; letter-spacing: 3px; color: #0a0a0a; text-transform: uppercase; }
.as-total-val { font-size: 28px; font-weight: 200; color: #cc0000; }

/* Offer box */
.as-offer { background: #0a0a0a; padding: 28px 32px; margin-top: 28px; }
.as-offer-row { display: flex; justify-content: space-between; align-items: baseline; padding: 6px 0; }
.as-offer-lbl { font-size: 9px; letter-spacing: 2px; color: #3a3a3a; text-transform: uppercase; }
.as-offer-val { font-size: 13px; font-weight: 300; color: #fff; }
.as-offer-val-red { font-size: 13px; font-weight: 300; color: #cc0000; }
.as-offer-divider { height: 1px; background: #1a1a1a; margin: 10px 0; }
.as-offer-total-lbl { font-size: 9px; letter-spacing: 2px; color: #555; text-transform: uppercase; }
.as-offer-total-val { font-size: 36px; font-weight: 200; color: #fff; letter-spacing: -1px; }

/* Country note */
.as-note { font-size: 10px; font-weight: 300; color: #c0c0c0; margin-top: 8px; }

/* Empty */
.as-empty { background: #fff; border: 1px solid #ebebeb; padding: 64px 32px; text-align: center; }
.as-empty-text { font-size: 12px; font-weight: 300; color: #c0c0c0; letter-spacing: 1px; }

/* Footer */
.as-footer { border-top: 1px solid #ebebeb; padding: 16px 48px; background: #fff; font-size: 8px; letter-spacing: 2px; color: #c0c0c0; text-transform: uppercase; }

/* Streamlit overrides */
.stFileUploader > div { border: 1px dashed #ebebeb !important; border-radius: 0 !important; background: #fff !important; }
.stFileUploader label { font-size: 9px !important; font-weight: 400 !important; letter-spacing: 2px !important; color: #c0c0c0 !important; text-transform: uppercase !important; }
.stSlider label { font-size: 9px !important; font-weight: 400 !important; letter-spacing: 2px !important; color: #c0c0c0 !important; text-transform: uppercase !important; }
.stSlider [role="slider"] { background: #0a0a0a !important; border-color: #0a0a0a !important; }
.stSelectbox label { font-size: 9px !important; font-weight: 400 !important; letter-spacing: 2px !important; color: #c0c0c0 !important; text-transform: uppercase !important; }
div[data-baseweb="select"] > div { border: none !important; border-bottom: 1px solid #ebebeb !important; border-radius: 0 !important; background: transparent !important; font-family: 'Inter', sans-serif !important; font-size: 13px !important; font-weight: 300 !important; }
.stNumberInput label { font-size: 9px !important; font-weight: 400 !important; letter-spacing: 2px !important; color: #c0c0c0 !important; text-transform: uppercase !important; }
.stNumberInput input { border: none !important; border-bottom: 1px solid #ebebeb !important; border-radius: 0 !important; background: transparent !important; font-family: 'Inter', sans-serif !important; font-size: 18px !important; font-weight: 200 !important; padding: 8px 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("AUTOSHOT_MODEL", "runs/detect/runs/detect/autoshot_combined_v12/weights/best.pt")
FALLBACK   = "runs/detect/runs/detect/autoshot_v4_final/weights/best.pt"
CLASS_NAMES = ['dent','scratch','crack','glass shatter','lamp broken','tire flat','paint damage']
CONF_DEFAULT = 0.40

SEVERITY = {
    'dent':('Medium','#fb923c'), 'scratch':('Low','#d4a843'),
    'crack':('High','#f87171'), 'glass shatter':('Critical','#cc0000'),
    'lamp broken':('High','#f87171'), 'tire flat':('Critical','#cc0000'),
    'paint damage':('Low','#d4a843'),
}
REPAIR = {
    'es':{'dent':(150,400),'scratch':(80,220),'crack':(200,500),'glass shatter':(300,900),'lamp broken':(150,600),'tire flat':(80,200),'paint damage':(200,500)},
    'pt':{'dent':(130,370),'scratch':(70,200),'crack':(180,460),'glass shatter':(270,830),'lamp broken':(140,550),'tire flat':(75,185),'paint damage':(185,460)},
    'br':{'dent':(800,2200),'scratch':(400,1200),'crack':(1100,2700),'glass shatter':(1600,4900),'lamp broken':(800,3200),'tire flat':(400,1100),'paint damage':(1100,2700)},
    'us':{'dent':(160,430),'scratch':(85,240),'crack':(215,540),'glass shatter':(325,970),'lamp broken':(160,650),'tire flat':(85,215),'paint damage':(215,540)},
    'mx':{'dent':(2900,7800),'scratch':(1600,4300),'crack':(3900,9800),'glass shatter':(5800,17500),'lamp broken':(2900,11700),'tire flat':(1600,3900),'paint damage':(3900,9800)},
}
SYM   = {'es':'€','pt':'€','br':'R$','us':'$','mx':'MX$'}
CNAME = {'es':'🇪🇸 Spain','pt':'🇵🇹 Portugal','br':'🇧🇷 Brazil','us':'🇺🇸 USA','mx':'🇲🇽 México'}

@st.cache_resource
def load_model():
    for p in [MODEL_PATH, FALLBACK]:
        if os.path.exists(p):
            return YOLO(p), p
    return None, None

model, model_path = load_model()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="as-header">
  <div style="display:flex;align-items:center;gap:16px">
    <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
      <circle cx="16" cy="16" r="15" stroke="white" stroke-width="0.75" opacity="0.25"/>
      <circle cx="16" cy="16" r="8" stroke="white" stroke-width="0.5" opacity="0.2"/>
      <circle cx="16" cy="16" r="3" fill="#cc0000"/>
      <line x1="16" y1="1" x2="16" y2="8" stroke="#cc0000" stroke-width="1.5"/>
      <line x1="24" y1="16" x2="31" y2="16" stroke="#cc0000" stroke-width="1.5"/>
      <line x1="16" y1="24" x2="16" y2="31" stroke="white" stroke-width="0.75" opacity="0.25"/>
      <line x1="1" y1="16" x2="8" y2="16" stroke="white" stroke-width="0.75" opacity="0.25"/>
    </svg>
    <div>
      <div class="as-name">Autoshot</div>
      <div class="as-tagline">Vehicle damage intelligence</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Two column layout ──────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.6], gap="large")

# ── LEFT: Controls ─────────────────────────────────────────────────────────────
with col_left:
    st.markdown('<div style="padding:40px 32px;background:#fff;min-height:calc(100vh - 64px);border-right:1px solid #ebebeb">', unsafe_allow_html=True)

    st.markdown('<div class="as-section">Market</div>', unsafe_allow_html=True)
    country = st.selectbox("Country", list(CNAME.keys()), format_func=lambda x: CNAME[x], label_visibility="collapsed")
    sym = SYM[country]

    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="as-section">Upload vehicle photos</div>', unsafe_allow_html=True)
    uploads = st.file_uploader(
        "Photos", type=["jpg","jpeg","png","webp"],
        accept_multiple_files=True, label_visibility="collapsed"
    )

    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    conf = st.slider("Detection confidence", 0.10, 0.90, CONF_DEFAULT, 0.05)

    if uploads:
        st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
        for f in uploads:
            st.image(Image.open(f), caption=f.name[:24], use_container_width=True)

    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="as-section">Dealer offer calculator</div>', unsafe_allow_html=True)
    market_val = st.number_input(f"Market value ({sym})", min_value=0, value=12000, step=500)
    margin = st.slider("Dealer margin", 5, 30, 15, format="%d%%")

    st.markdown('</div>', unsafe_allow_html=True)

# ── RIGHT: Results ──────────────────────────────────────────────────────────────
with col_right:
    st.markdown('<div style="padding:40px">', unsafe_allow_html=True)

    if not uploads:
        st.markdown("""
        <div class="as-empty">
            <div style="font-size:40px;opacity:0.1;margin-bottom:16px">⊙</div>
            <div class="as-empty-text">Upload vehicle photos to begin analysis</div>
        </div>
        """, unsafe_allow_html=True)

    elif model is None:
        st.error(f"Model not found: {MODEL_PATH}")

    else:
        # Run inference
        all_classes, all_confs, annotated_imgs = [], [], []
        with st.spinner("Analysing..."):
            for f in uploads:
                img = Image.open(f).convert("RGB")
                arr = np.array(img)
                res = model.predict(arr, conf=conf, verbose=False)[0]
                if res.boxes and len(res.boxes) > 0:
                    all_classes += [CLASS_NAMES[int(c)] if int(c) < len(CLASS_NAMES) else f"class_{int(c)}" for c in res.boxes.cls.cpu().numpy()]
                    all_confs   += list(res.boxes.conf.cpu().numpy())
                ann = res.plot(line_width=2, pil=False, img=arr.copy())
                annotated_imgs.append((f.name, cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)))

        # Show annotated images
        img_cols = st.columns(min(len(annotated_imgs), 3))
        for i, (name, img) in enumerate(annotated_imgs):
            with img_cols[i % 3]:
                st.image(img, use_container_width=True)

        st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

        # ── No damage ──
        if not all_classes:
            st.markdown('<div class="as-clean">✓ No damage detected</div>', unsafe_allow_html=True)
            total_low, total_high = 0, 0

        # ── Damage found ──
        else:
            counts   = Counter(all_classes)
            avg_conf = float(np.mean(all_confs))
            n        = len(uploads)

            st.markdown(f'<div class="as-damaged">⚠ {len(all_classes)} damage instance{"s" if len(all_classes)>1 else ""} detected across {n} photo{"s" if n>1 else ""}</div>', unsafe_allow_html=True)

            # Metrics
            st.markdown(f"""
            <div class="as-metrics">
                <div class="as-metric"><div class="as-metric-val-red">{len(all_classes)}</div><div class="as-metric-lbl">Detections</div></div>
                <div class="as-metric"><div class="as-metric-val">{len(counts)}</div><div class="as-metric-lbl">Types</div></div>
                <div class="as-metric"><div class="as-metric-val">{avg_conf:.0%}</div><div class="as-metric-lbl">Confidence</div></div>
            </div>
            """, unsafe_allow_html=True)

            # Damage + costs
            st.markdown('<div class="as-section">Damage breakdown & repair estimate</div>', unsafe_allow_html=True)
            costs_table = REPAIR[country]
            total_low, total_high = 0, 0
            items_html = ""

            class_confs = {}
            for cls, c in zip(all_classes, all_confs):
                class_confs.setdefault(cls, []).append(c)

            for cls, count in sorted(counts.items(), key=lambda x: -max(class_confs[x[0]])):
                sev, color = SEVERITY.get(cls, ('Medium','#fb923c'))
                low, high  = costs_table.get(cls, (100,300))
                total_low  += low  * count
                total_high += high * count
                items_html += f"""
                <div class="as-damage-item">
                    <div class="as-damage-dot" style="background:{color}"></div>
                    <div class="as-damage-name">{cls}</div>
                    <div class="as-damage-count">×{count}</div>
                    <div class="as-damage-sev" style="color:{color}">{sev}</div>
                    <div class="as-damage-cost">{sym} {low*count:,}–{high*count:,}</div>
                </div>"""

            st.markdown(f"""
            {items_html}
            <div class="as-total">
                <span class="as-total-label">Total repair estimate</span>
                <span class="as-total-val">{sym} {total_low:,}–{total_high:,}</span>
            </div>
            <div class="as-note">Indicative prices for {CNAME[country]}. Labour rates vary by region.</div>
            """, unsafe_allow_html=True)

        # ── Offer calculator ──
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        repair_mid = (total_low + total_high) // 2
        margin_amt = (market_val - repair_mid) * margin / 100
        max_offer  = max(0, market_val - repair_mid - margin_amt)

        st.markdown(f"""
        <div class="as-offer">
            <div class="as-offer-row">
                <span class="as-offer-lbl">Market value</span>
                <span class="as-offer-val">{sym} {market_val:,}</span>
            </div>
            <div class="as-offer-row">
                <span class="as-offer-lbl">Repair estimate</span>
                <span class="as-offer-val-red">— {sym} {repair_mid:,}</span>
            </div>
            <div class="as-offer-row">
                <span class="as-offer-lbl">Dealer margin {margin}%</span>
                <span class="as-offer-val-red">— {sym} {int(margin_amt):,}</span>
            </div>
            <div class="as-offer-divider"></div>
            <div class="as-offer-row">
                <span class="as-offer-total-lbl">Max offer</span>
                <span class="as-offer-total-val">{sym} {int(max_offer):,}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
model_name = os.path.basename(os.path.dirname(os.path.dirname(model_path))) if model_path else "No model"
st.markdown(f'<div class="as-footer">Autoshot · Vehicle damage intelligence &nbsp;·&nbsp; {model_name}</div>', unsafe_allow_html=True)
