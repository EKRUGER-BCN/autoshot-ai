import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2
import os
import base64
import json
import io
import requests
from collections import Counter

st.set_page_config(
    page_title="Autoshot",
    page_icon="⊙",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'DM Sans', sans-serif !important;
    background: #ffffff !important;
    color: #0a0a0a !important;
}
[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], #MainMenu, footer { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
[data-testid="stSidebar"] { display: none !important; }

.as-nav {
    background: #ffffff; border-bottom: 1px solid #f0f0f0;
    padding: 0 48px; height: 60px;
    display: flex; align-items: center; justify-content: space-between;
}
.as-logo-name { font-size: 12px; font-weight: 500; letter-spacing: 0.16em; color: #0a0a0a; text-transform: uppercase; }
.as-logo-tag { font-size: 9px; letter-spacing: 0.22em; color: #ccc; text-transform: uppercase; margin-top: 2px; font-weight: 300; }
.as-page { max-width: 1140px; margin: 0 auto; padding: 0 48px 80px; }

.as-step { display: flex; align-items: center; gap: 12px; margin: 48px 0 24px; }
.as-step-num {
    width: 26px; height: 26px; border-radius: 50%;
    border: 1px solid #0a0a0a;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 500; flex-shrink: 0;
}
.as-step-title { font-size: 13px; font-weight: 500; letter-spacing: 0.04em; }
.as-field-label { font-size: 9px; font-weight: 500; letter-spacing: 0.18em; text-transform: uppercase; color: #bbb; display: block; margin-bottom: 8px; }

.as-photo-slot {
    border: 1.5px dashed #e0e0e0; border-radius: 14px; aspect-ratio: 4/3;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    cursor: pointer; background: #fff; padding: 16px 12px; text-align: center;
}
.as-slot-active { border-color: #0a0a0a !important; background: #fafafa !important; }
.as-slot-label { font-size: 10px; font-weight: 500; letter-spacing: 0.16em; text-transform: uppercase; color: #0a0a0a; margin-top: 12px; }
.as-slot-label-muted { font-size: 10px; font-weight: 500; letter-spacing: 0.16em; text-transform: uppercase; color: #ccc; margin-top: 12px; }
.as-slot-hint { font-size: 9px; color: #ccc; margin-top: 4px; line-height: 1.5; }

.as-divider { height: 1px; background: #f0f0f0; margin: 48px 0 0; }

.as-vehicle-card { background: #fff; border: 1px solid #f0f0f0; border-radius: 14px; padding: 26px 28px; }
.as-vc-make { font-size: 22px; font-weight: 300; letter-spacing: -0.5px; }
.as-vc-detail { font-size: 12px; color: #aaa; margin-top: 4px; font-weight: 300; }
.as-vc-div { height: 1px; background: #f5f5f5; margin: 18px 0; }
.as-vc-row { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 10px; }
.as-vc-lbl { font-size: 9px; letter-spacing: 0.15em; text-transform: uppercase; color: #ccc; }
.as-vc-val { font-size: 13px; font-weight: 400; font-family: 'DM Mono', monospace; color: #0a0a0a; }
.as-vc-val-red { font-size: 13px; font-weight: 400; font-family: 'DM Mono', monospace; color: #cc0000; }

.as-dmg-row { display: flex; align-items: center; gap: 10px; padding: 10px 0; border-bottom: 1px solid #f5f5f5; }
.as-dmg-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
.as-dmg-name { font-size: 12px; color: #0a0a0a; flex: 1; text-transform: capitalize; }
.as-dmg-cost { font-size: 11px; color: #999; font-family: 'DM Mono', monospace; }

.as-offer-card {
    background: #0a0a0a; border-radius: 14px; padding: 30px 32px;
    display: flex; flex-direction: column; justify-content: space-between; min-height: 340px;
}
.as-oc-label { font-size: 9px; letter-spacing: 0.2em; text-transform: uppercase; color: #444; }
.as-oc-offer { font-size: 52px; font-weight: 300; color: #fff; letter-spacing: -2px; line-height: 1; margin: 14px 0 6px; }
.as-oc-sub { font-size: 11px; color: #444; letter-spacing: 0.04em; }
.as-oc-div { height: 1px; background: #1a1a1a; margin: 20px 0; }
.as-oc-row { display: flex; justify-content: space-between; margin-bottom: 9px; }
.as-oc-rlbl { font-size: 10px; color: #555; letter-spacing: 0.04em; }
.as-oc-rval { font-size: 10px; color: #888; font-family: 'DM Mono', monospace; }
.as-oc-rval-red { font-size: 10px; color: #cc4444; font-family: 'DM Mono', monospace; }

.as-clean { background: #f0faf2; border: 1px solid #c3e6cb; border-radius: 8px; padding: 12px 18px; color: #2d7d3a; font-size: 12px; margin-bottom: 20px; }
.as-damaged { background: #fff8f0; border: 1px solid #fdd9b5; border-radius: 8px; padding: 12px 18px; color: #b45309; font-size: 12px; margin-bottom: 20px; }
.as-note { font-size: 10px; color: #ccc; margin-top: 8px; line-height: 1.5; font-weight: 300; }

.as-footer {
    border-top: 1px solid #f0f0f0; padding: 14px 48px; background: #fff;
    font-size: 9px; letter-spacing: 0.16em; color: #ccc; text-transform: uppercase;
    display: flex; justify-content: space-between;
}

div[data-testid="stImage"] img { border-radius: 10px !important; border: 1px solid #f0f0f0 !important; }
.stSelectbox label { font-size: 9px !important; font-weight: 500 !important; letter-spacing: 0.18em !important; color: #bbb !important; text-transform: uppercase !important; }
div[data-baseweb="select"] > div { border: 1px solid #e8e8e8 !important; border-radius: 8px !important; background: #fff !important; font-family: 'DM Sans', sans-serif !important; font-size: 13px !important; }
div[data-baseweb="select"] > div:focus-within { border-color: #0a0a0a !important; box-shadow: none !important; }
.stTextInput input { border: 1px solid #e8e8e8 !important; border-radius: 8px !important; font-family: 'DM Mono', monospace !important; font-size: 14px !important; letter-spacing: 0.12em !important; text-transform: uppercase !important; }
.stTextInput label { font-size: 9px !important; font-weight: 500 !important; letter-spacing: 0.18em !important; color: #bbb !important; text-transform: uppercase !important; }
.stFileUploader > div { border: 1.5px dashed #e0e0e0 !important; border-radius: 14px !important; background: #fff !important; }
.stFileUploader label { font-size: 9px !important; font-weight: 500 !important; letter-spacing: 0.18em !important; color: #bbb !important; text-transform: uppercase !important; }
div[data-testid="stButton"] button {
    background: #0a0a0a !important; color: #fff !important; border: none !important;
    border-radius: 10px !important; padding: 14px 44px !important;
    font-size: 13px !important; font-weight: 500 !important;
    font-family: 'DM Sans', sans-serif !important; letter-spacing: 0.06em !important;
}
div[data-testid="stButton"] button:hover { background: #222 !important; }
div[data-testid="stButton"] button:disabled { background: #e0e0e0 !important; color: #bbb !important; cursor: not-allowed !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH    = os.environ.get("AUTOSHOT_MODEL", "runs/detect/runs/detect/autoshot_combined_v12/weights/best.pt")
FALLBACK      = "runs/detect/runs/detect/autoshot_v4_final/weights/best.pt"
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLASS_NAMES   = ['dent','scratch','crack','glass shatter','lamp broken','tire flat','paint damage']

SEVERITY = {
    'dent':          ('Medium',  '#fb923c'),
    'scratch':       ('Low',     '#d4a843'),
    'crack':         ('High',    '#f87171'),
    'glass shatter': ('Critical','#cc0000'),
    'lamp broken':   ('High',    '#f87171'),
    'tire flat':     ('Critical','#cc0000'),
    'paint damage':  ('Low',     '#d4a843'),
}
REPAIR = {
    'es': {'dent':(150,400),'scratch':(80,220),'crack':(200,500),'glass shatter':(300,900),'lamp broken':(150,600),'tire flat':(80,200),'paint damage':(200,500)},
    'pt': {'dent':(130,370),'scratch':(70,200),'crack':(180,460),'glass shatter':(270,830),'lamp broken':(140,550),'tire flat':(75,185),'paint damage':(185,460)},
    'br': {'dent':(800,2200),'scratch':(400,1200),'crack':(1100,2700),'glass shatter':(1600,4900),'lamp broken':(800,3200),'tire flat':(400,1100),'paint damage':(1100,2700)},
    'us': {'dent':(160,430),'scratch':(85,240),'crack':(215,540),'glass shatter':(325,970),'lamp broken':(160,650),'tire flat':(85,215),'paint damage':(215,540)},
    'mx': {'dent':(2900,7800),'scratch':(1600,4300),'crack':(3900,9800),'glass shatter':(5800,17500),'lamp broken':(2900,11700),'tire flat':(1600,3900),'paint damage':(3900,9800)},
}
SYM   = {'es':'€','pt':'€','br':'R$','us':'$','mx':'MX$'}
CNAME = {'es':'Spain','pt':'Portugal','br':'Brazil','us':'United States','mx':'México'}
CFLAG = {
    'es': 'https://cdn.jsdelivr.net/gh/lipis/flag-icons@7.2.3/flags/4x3/es.svg',
    'pt': 'https://cdn.jsdelivr.net/gh/lipis/flag-icons@7.2.3/flags/4x3/pt.svg',
    'br': 'https://cdn.jsdelivr.net/gh/lipis/flag-icons@7.2.3/flags/4x3/br.svg',
    'us': 'https://cdn.jsdelivr.net/gh/lipis/flag-icons@7.2.3/flags/4x3/us.svg',
    'mx': 'https://cdn.jsdelivr.net/gh/lipis/flag-icons@7.2.3/flags/4x3/mx.svg',
}

SVG_FRONT = """<svg width="90" height="52" viewBox="0 0 90 52" fill="none">
  <rect x="8" y="20" width="74" height="24" rx="4" stroke="{c}" stroke-width="1.3"/>
  <path d="M20 20 L30 8 L60 8 L70 20" stroke="{c}" stroke-width="1.3" fill="none"/>
  <circle cx="22" cy="44" r="6" stroke="{c}" stroke-width="1.3"/>
  <circle cx="68" cy="44" r="6" stroke="{c}" stroke-width="1.3"/>
  <rect x="10" y="24" width="18" height="9" rx="2" fill="{f}"/>
  <rect x="62" y="24" width="18" height="9" rx="2" fill="{f}"/>
  <rect x="32" y="22" width="26" height="7" rx="1.5" fill="{f}"/>
  <circle cx="45" cy="25.5" r="2.5" fill="{a}"/>
</svg>"""

SVG_REAR = """<svg width="90" height="52" viewBox="0 0 90 52" fill="none">
  <rect x="8" y="20" width="74" height="24" rx="4" stroke="{c}" stroke-width="1.3"/>
  <path d="M20 20 L30 8 L60 8 L70 20" stroke="{c}" stroke-width="1.3" fill="none"/>
  <circle cx="22" cy="44" r="6" stroke="{c}" stroke-width="1.3"/>
  <circle cx="68" cy="44" r="6" stroke="{c}" stroke-width="1.3"/>
  <rect x="10" y="24" width="18" height="9" rx="2" fill="{f}"/>
  <rect x="62" y="24" width="18" height="9" rx="2" fill="{f}"/>
  <rect x="32" y="29" width="26" height="5" rx="1" fill="{f}"/>
  <line x1="34" y1="31" x2="56" y2="31" stroke="{c}" stroke-width="0.8" opacity="0.4"/>
</svg>"""

SVG_SIDE = """<svg width="100" height="52" viewBox="0 0 100 52" fill="none">
  <rect x="4" y="22" width="92" height="20" rx="3" stroke="{c}" stroke-width="1.3"/>
  <path d="M14 22 L24 10 L66 10 L78 22" stroke="{c}" stroke-width="1.3" fill="none"/>
  <circle cx="22" cy="42" r="6" stroke="{c}" stroke-width="1.3"/>
  <circle cx="78" cy="42" r="6" stroke="{c}" stroke-width="1.3"/>
  <rect x="26" y="11" width="18" height="11" rx="2" fill="{f}"/>
  <rect x="48" y="11" width="16" height="11" rx="2" fill="{f}"/>
  <line x1="4" y1="32" x2="96" y2="32" stroke="{c}" stroke-width="0.7" opacity="0.2"/>
  <rect x="80" y="26" width="12" height="6" rx="1" fill="{f}"/>
</svg>"""

def car_svg(template, active=False):
    if active:
        return template.format(c="#0a0a0a", f="#e8e8e8", a="#cc0000")
    return template.format(c="#d0d0d0", f="#f5f5f5", a="#e0e0e0")

@st.cache_resource
def load_model():
    for p in [MODEL_PATH, FALLBACK]:
        if os.path.exists(p):
            return YOLO(p), p
    return None, None

model, model_path = load_model()

def identify_vehicle(image_pil, country_code, sym):
    if not ANTHROPIC_KEY:
        return None
    buf = io.BytesIO()
    image_pil.save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    country_name = CNAME.get(country_code, "Spain")
    prompt = f"""You are an automotive expert. Analyse this vehicle photo and respond ONLY with raw JSON — no markdown, no preamble.

Return this exact structure:
{{
  "make": "e.g. Volkswagen",
  "model": "e.g. Golf",
  "year_range": "e.g. 2018-2020",
  "trim": "e.g. 1.6 TDI Comfortline",
  "fuel": "e.g. Diesel",
  "body": "e.g. Hatchback",
  "market_value_low": 11000,
  "market_value_high": 14500,
  "market_value_mid": 12500,
  "reasoning": "One sentence on valuation basis."
}}

Base market values on current {country_name} used car prices ({sym}). Return ONLY the JSON."""
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 512,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                        {"type": "text", "text": prompt}
                    ]
                }]
            },
            timeout=30
        )
        raw = resp.json()["content"][0]["text"].strip()
        raw = raw.replace("```json","").replace("```","").strip()
        return json.loads(raw)
    except Exception:
        return None

# ── Nav ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="as-nav">
  <div style="display:flex;align-items:center;gap:14px">
    <svg width="26" height="26" viewBox="0 0 26 26" fill="none">
      <circle cx="13" cy="13" r="12" stroke="#0a0a0a" stroke-width="1.2"/>
      <circle cx="13" cy="13" r="6.5" stroke="#0a0a0a" stroke-width="0.8"/>
      <circle cx="13" cy="13" r="2.2" fill="#cc0000"/>
      <line x1="13" y1="1" x2="13" y2="6.5" stroke="#0a0a0a" stroke-width="1.2"/>
      <line x1="13" y1="19.5" x2="13" y2="25" stroke="#0a0a0a" stroke-width="1.2"/>
      <line x1="1" y1="13" x2="6.5" y2="13" stroke="#0a0a0a" stroke-width="1.2"/>
      <line x1="19.5" y1="13" x2="25" y2="13" stroke="#0a0a0a" stroke-width="1.2"/>
    </svg>
    <div>
      <div class="as-logo-name">Autoshot</div>
      <div class="as-logo-tag">Vehicle Damage Intelligence</div>
    </div>
  </div>
  <div style="font-size:10px;color:#ccc;letter-spacing:0.08em;font-family:'DM Sans',sans-serif">AI-Powered Assessment</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="as-page">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Configure
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="as-step">
  <div class="as-step-num">1</div>
  <div class="as-step-title">Configure assessment</div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown('<span class="as-field-label">Market</span>', unsafe_allow_html=True)
    country = st.selectbox(
        "Market", list(CNAME.keys()),
        format_func=lambda x: CNAME[x],
        label_visibility="collapsed", key="country"
    )
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;margin-top:-6px">
      <img src="{CFLAG[country]}" style="width:22px;height:15px;border-radius:2px;border:1px solid #f0f0f0;object-fit:cover"/>
      <span style="font-size:10px;color:#bbb;font-family:'DM Sans',sans-serif;letter-spacing:0.06em">{country.upper()} · {SYM[country]}</span>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown('<span class="as-field-label">License Plate</span>', unsafe_allow_html=True)
    plate = st.text_input("Plate", placeholder="1234 ABC", label_visibility="collapsed", key="plate")

with col3:
    st.markdown('<span class="as-field-label">Dealer Margin</span>', unsafe_allow_html=True)
    margin = st.selectbox(
        "Margin", [10, 12, 15, 18, 20, 25], index=2,
        format_func=lambda x: f"{x}%",
        label_visibility="collapsed", key="margin"
    )

sym = SYM[country]

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Upload photos
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="as-step">
  <div class="as-step-num">2</div>
  <div class="as-step-title">Upload vehicle photos</div>
</div>
""", unsafe_allow_html=True)

SLOTS = [
    ("Front",      SVG_FRONT, "Full front view · straight on"),
    ("Rear",       SVG_REAR,  "Full rear view · straight on"),
    ("Side Left",  SVG_SIDE,  "Driver side · full length"),
    ("Side Right", SVG_SIDE,  "Passenger side · full length"),
]

photo_cols = st.columns(4, gap="small")
uploads = {}

for i, (label, svg_tpl, hint) in enumerate(SLOTS):
    with photo_cols[i]:
        file = st.file_uploader(
            label, type=["jpg","jpeg","png","webp"],
            key=f"slot_{i}", label_visibility="collapsed"
        )
        uploads[label] = file
        if file:
            st.image(Image.open(file), use_container_width=True)
            st.markdown(f'<div style="text-align:center;margin-top:2px"><span style="font-size:9px;font-weight:500;letter-spacing:0.12em;text-transform:uppercase;color:#2d7d3a">✓ {label}</span></div>', unsafe_allow_html=True)
        else:
            active = (i == 0)
            st.markdown(f"""
            <div class="as-photo-slot {'as-slot-active' if active else ''}">
              {car_svg(svg_tpl, active=active)}
              <div class="{'as-slot-label' if active else 'as-slot-label-muted'}">{label}</div>
              <div class="as-slot-hint">{hint}</div>
            </div>""", unsafe_allow_html=True)

all_uploads = [f for f in uploads.values() if f is not None]

st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
_, btn_col, _ = st.columns([2, 1, 2])
with btn_col:
    analyse = st.button("⊙  Analyse Vehicle", key="analyse", disabled=(len(all_uploads) == 0))

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Results
# ══════════════════════════════════════════════════════════════════════════════
if analyse or st.session_state.get("show_results"):

    if analyse:
        st.session_state["show_results"] = True
        st.session_state["vehicle_data"] = None

    st.markdown('<div class="as-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="as-step">
      <div class="as-step-num">3</div>
      <div class="as-step-title">Assessment results</div>
    </div>""", unsafe_allow_html=True)

    if not all_uploads:
        st.warning("Please upload at least one vehicle photo.")
        st.session_state["show_results"] = False

    elif model is None:
        st.error("Model not found. Check AUTOSHOT_MODEL environment variable.")

    else:
        # Claude Vision
        if ANTHROPIC_KEY and st.session_state.get("vehicle_data") is None:
            first_img = Image.open(all_uploads[0]).convert("RGB")
            with st.spinner("Identifying vehicle..."):
                vd = identify_vehicle(first_img, country, sym)
                st.session_state["vehicle_data"] = vd
        vd = st.session_state.get("vehicle_data")

        default_market = vd["market_value_mid"] if vd and vd.get("market_value_mid") else 12000

        # YOLO inference
        all_classes, all_confs, annotated_imgs = [], [], []
        with st.spinner("Detecting damage..."):
            for f in all_uploads:
                f.seek(0)
                img = Image.open(f).convert("RGB")
                arr = np.array(img)
                res = model.predict(arr, conf=0.40, verbose=False)[0]
                if res.boxes and len(res.boxes) > 0:
                    all_classes += [
                        CLASS_NAMES[int(c)] if int(c) < len(CLASS_NAMES) else f"class_{int(c)}"
                        for c in res.boxes.cls.cpu().numpy()
                    ]
                    all_confs += list(res.boxes.conf.cpu().numpy())
                ann = res.plot(line_width=2, pil=False, img=arr.copy())
                annotated_imgs.append((f.name, cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)))

        # Annotated images
        if annotated_imgs:
            img_cols = st.columns(min(len(annotated_imgs), 4), gap="small")
            for i, (name, img) in enumerate(annotated_imgs):
                with img_cols[i % 4]:
                    st.image(img, use_container_width=True)
            st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

        # Costs
        costs_table = REPAIR[country]
        total_low, total_high = 0, 0
        counts = Counter(all_classes)
        if all_classes:
            for cls, count in counts.items():
                low, high   = costs_table.get(cls, (100, 300))
                total_low  += low * count
                total_high += high * count

        repair_mid = (total_low + total_high) // 2
        market_val = default_market
        margin_amt = int((market_val - repair_mid) * margin / 100)
        max_offer  = max(0, market_val - repair_mid - margin_amt)

        # Results grid
        left_col, right_col = st.columns(2, gap="medium")

        with left_col:
            if vd:
                make_model = f"{vd.get('make','')} {vd.get('model','')}"
                detail     = f"{vd.get('year_range','')} · {vd.get('trim','')} · {vd.get('fuel','')} · {vd.get('body','')}"
                mv_str     = f"{sym} {vd.get('market_value_low',0):,} – {sym} {vd.get('market_value_high',0):,}"
            else:
                make_model = "Vehicle"
                detail     = "Set ANTHROPIC_API_KEY for AI identification"
                mv_str     = f"{sym} {market_val:,}"

            if not all_classes:
                status_html = '<div class="as-clean">✓ No damage detected across all photos</div>'
                repair_str  = f"{sym} 0"
            else:
                n = len(all_uploads)
                status_html = f'<div class="as-damaged">⚠ {len(all_classes)} damage instance{"s" if len(all_classes)>1 else ""} detected across {n} photo{"s" if n>1 else ""}</div>'
                repair_str  = f"{sym} {total_low:,} – {total_high:,}"

            dmg_html = ""
            if all_classes:
                for cls, count in sorted(counts.items(), key=lambda x: -x[1]):
                    _, color = SEVERITY.get(cls, ('Medium','#fb923c'))
                    low, high = costs_table.get(cls, (100,300))
                    dmg_html += f"""
                    <div class="as-dmg-row">
                      <div class="as-dmg-dot" style="background:{color}"></div>
                      <div class="as-dmg-name">{cls} &times;{count}</div>
                      <div class="as-dmg-cost">{sym} {low*count:,}–{high*count:,}</div>
                    </div>"""
            else:
                dmg_html = '<div style="font-size:12px;color:#ccc;padding:10px 0">No defects found</div>'

            st.markdown(f"""
            {status_html}
            <div class="as-vehicle-card">
              <div class="as-vc-make">{make_model}</div>
              <div class="as-vc-detail">{detail}</div>
              <div class="as-vc-div"></div>
              <div class="as-vc-row">
                <span class="as-vc-lbl">Market Value</span>
                <span class="as-vc-val">{mv_str}</span>
              </div>
              <div class="as-vc-row">
                <span class="as-vc-lbl">Repair Estimate</span>
                <span class="as-vc-val-red">{repair_str}</span>
              </div>
              <div class="as-vc-div"></div>
              {dmg_html}
              <div class="as-note">{CNAME[country]} labour rates · Indicative estimates</div>
            </div>""", unsafe_allow_html=True)

        with right_col:
            st.markdown(f"""
            <div class="as-offer-card">
              <div>
                <div class="as-oc-label">Maximum Purchase Offer</div>
                <div class="as-oc-offer">{sym} {int(max_offer):,}</div>
                <div class="as-oc-sub">Recommended dealer bid · {CNAME[country]}</div>
              </div>
              <div>
                <div class="as-oc-div"></div>
                <div class="as-oc-row">
                  <span class="as-oc-rlbl">Market value</span>
                  <span class="as-oc-rval">{sym} {market_val:,}</span>
                </div>
                <div class="as-oc-row">
                  <span class="as-oc-rlbl">Repair estimate</span>
                  <span class="as-oc-rval-red">− {sym} {repair_mid:,}</span>
                </div>
                <div class="as-oc-row">
                  <span class="as-oc-rlbl">Dealer margin {margin}%</span>
                  <span class="as-oc-rval-red">− {sym} {margin_amt:,}</span>
                </div>
                <div style="height:1px;background:#1a1a1a;margin:14px 0"></div>
                <div style="font-size:9px;color:#333;letter-spacing:0.1em;line-height:1.7;font-family:'DM Sans',sans-serif">
                  Autoshot · Vehicle Damage Intelligence<br>
                  Model v12 · mAP 0.601 · 7 damage classes
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
model_name = os.path.basename(os.path.dirname(os.path.dirname(model_path))) if model_path else "No model"
st.markdown(f"""
<div class="as-footer">
  <span>Autoshot · Vehicle Damage Intelligence</span>
  <span>{model_name} · 7 damage classes</span>
</div>""", unsafe_allow_html=True)
