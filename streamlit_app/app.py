import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2
import os
from collections import Counter

st.set_page_config(page_title="Autoshot", page_icon="🔍", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@200;300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: #fff; color: #0a0a0a; }
.header { background: #0a0a0a; padding: 20px 32px; margin: -1rem -1rem 2rem; }
.header h1 { font-size: 20px; font-weight: 300; letter-spacing: 5px; color: #fff; text-transform: uppercase; margin: 0; }
.header p { font-size: 9px; letter-spacing: 2px; color: #444; margin: 4px 0 0; text-transform: uppercase; }
.damage-card { border-bottom: 1px solid #f5f5f5; padding: 14px 0; display: flex; align-items: center; justify-content: space-between; }
.metric { text-align: center; padding: 20px; border: 1px solid #f0f0f0; }
.metric-val { font-size: 28px; font-weight: 200; color: #0a0a0a; }
.metric-lbl { font-size: 9px; letter-spacing: 2px; color: #bbb; text-transform: uppercase; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = os.environ.get("AUTOSHOT_MODEL", "model/weights/best.pt")
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
    'es': {'dent': (150,400), 'scratch': (80,220), 'crack': (200,500), 'glass shatter': (300,900), 'lamp broken': (150,600), 'tire flat': (80,200), 'paint damage': (200,500)},
    'pt': {'dent': (130,370), 'scratch': (70,200), 'crack': (180,460), 'glass shatter': (270,830), 'lamp broken': (140,550), 'tire flat': (75,185), 'paint damage': (185,460)},
    'br': {'dent': (800,2200), 'scratch': (400,1200), 'crack': (1100,2700), 'glass shatter': (1600,4900), 'lamp broken': (800,3200), 'tire flat': (400,1100), 'paint damage': (1100,2700)},
    'us': {'dent': (160,430), 'scratch': (85,240), 'crack': (215,540), 'glass shatter': (325,970), 'lamp broken': (160,650), 'tire flat': (85,215), 'paint damage': (215,540)},
}
CURRENCY = {'es': '€', 'pt': '€', 'br': 'R$', 'us': '$'}

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return YOLO(MODEL_PATH)
    return None

st.markdown('<div class="header"><h1>Autoshot</h1><p>Vehicle damage intelligence</p></div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("##### Country")
    country = st.selectbox("Country", ['es', 'pt', 'br', 'us'],
                           format_func=lambda x: {'es':'🇪🇸 Spain','pt':'🇵🇹 Portugal','br':'🇧🇷 Brazil','us':'🇺🇸 USA'}[x],
                           label_visibility="collapsed")
    sym = CURRENCY[country]

    st.markdown("##### Upload vehicle photos")
    uploads = st.file_uploader("Upload", type=["jpg","jpeg","png","webp"],
                                accept_multiple_files=True,
                                label_visibility="collapsed")
    conf = st.slider("Confidence threshold", 0.10, 0.90, CONF_THRESHOLD, 0.05)

    if uploads:
        for f in uploads:
            st.image(Image.open(f), caption=f.name, use_container_width=True)

with col_right:
    st.markdown("##### Damage analysis")

    model = load_model()
    if model is None:
        st.info("Model not loaded. Set AUTOSHOT_MODEL env var or place weights at model/weights/best.pt")
    elif not uploads:
        st.markdown('<div style="padding:3rem;text-align:center;color:#ddd;border:1px solid #f5f5f5;border-radius:4px;">Upload photos to begin</div>', unsafe_allow_html=True)
    else:
        all_classes = []
        all_confs = []

        for f in uploads:
            img = Image.open(f).convert("RGB")
            img_array = np.array(img)
            results = model.predict(img_array, conf=conf, verbose=False)[0]
            if results.boxes:
                all_classes += [CLASS_NAMES[int(c)] if int(c) < len(CLASS_NAMES) else f"class_{int(c)}"
                                 for c in results.boxes.cls.cpu().numpy()]
                all_confs += list(results.boxes.conf.cpu().numpy())

            annotated = results.plot(line_width=2, pil=False, img=img_array.copy())
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        if not all_classes:
            st.success("✓ No damage detected across all photos")
        else:
            counts = Counter(all_classes)
            avg_conf = float(np.mean(all_confs))

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'<div class="metric"><div class="metric-val">{len(all_classes)}</div><div class="metric-lbl">Detections</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric"><div class="metric-val">{len(counts)}</div><div class="metric-lbl">Types</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric"><div class="metric-val">{avg_conf:.0%}</div><div class="metric-lbl">Avg conf</div></div>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("##### Damage breakdown")

            total_low, total_high = 0, 0
            for cls, count in sorted(counts.items()):
                sev, color = SEVERITY.get(cls, ('Medium', '#fb923c'))
                costs = REPAIR_COSTS[country].get(cls, (100, 300))
                low, high = costs[0] * count, costs[1] * count
                total_low += low
                total_high += high
                st.markdown(f"""
                <div class="damage-card">
                    <div>
                        <div style="font-size:12px;font-weight:400;text-transform:capitalize">{cls} ×{count}</div>
                        <div style="font-size:10px;color:#bbb;margin-top:2px">{sev} severity</div>
                    </div>
                    <div style="text-align:right">
                        <div style="font-size:13px;font-weight:300;color:#0a0a0a">{sym} {low:,}–{high:,}</div>
                        <div style="font-size:9px;color:{color};letter-spacing:1px;text-transform:uppercase">{sev}</div>
                    </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("##### Repair estimate")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Low estimate", f"{sym} {total_low:,}")
            with col_b:
                st.metric("High estimate", f"{sym} {total_high:,}")

            st.markdown("---")
            st.markdown("##### Dealer offer calculator")
            market_val = st.number_input(f"Market value ({sym})", min_value=0, value=12000, step=500)
            margin = st.slider("Dealer margin", 5, 30, 15)
            repair_mid = (total_low + total_high) // 2
            margin_amt = (market_val - repair_mid) * margin / 100
            max_offer = max(0, market_val - repair_mid - margin_amt)

            st.markdown(f"""
            <div style="background:#0a0a0a;padding:24px;margin-top:8px">
                <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1e1e1e">
                    <span style="font-size:9px;letter-spacing:2px;color:#444;text-transform:uppercase">Market value</span>
                    <span style="font-size:13px;font-weight:300;color:#fff">{sym} {market_val:,}</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1e1e1e">
                    <span style="font-size:9px;letter-spacing:2px;color:#444;text-transform:uppercase">Repair (mid)</span>
                    <span style="font-size:13px;font-weight:300;color:#cc0000">— {sym} {repair_mid:,}</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1e1e1e">
                    <span style="font-size:9px;letter-spacing:2px;color:#444;text-transform:uppercase">Margin {margin}%</span>
                    <span style="font-size:13px;font-weight:300;color:#cc0000">— {sym} {int(margin_amt):,}</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:12px 0 0">
                    <span style="font-size:9px;letter-spacing:2px;color:#666;text-transform:uppercase">Max offer</span>
                    <span style="font-size:28px;font-weight:200;color:#fff">{sym} {int(max_offer):,}</span>
                </div>
            </div>""", unsafe_allow_html=True)
