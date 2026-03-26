import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2
import os
import base64
import json
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
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&family=DM+Mono:wght@400&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'DM Sans', sans-serif !important;
    background: #ffffff !important;
    color: #0a0a0a !important;
}
[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], #MainMenu, footer { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
[data-testid="stSidebar"] { display: none !important; }

/* ── Nav ── */
.as-nav {
    background: #ffffff;
    border-bottom: 1px solid #f0f0f0;
    padding: 0 40px;
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
}
.as-logo-name {
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.14em;
    color: #0a0a0a;
    text-transform: uppercase;
}
.as-logo-sub {
    font-size: 9px;
    letter-spacing: 0.2em;
    color: #c0c0c0;
    font-weight: 300;
    margin-top: 1px;
    text-transform: uppercase;
}
.as-nav-badge {
    font-size: 10px;
    background: #f5f5f5;
    color: #888;
    padding: 4px 12px;
    border-radius: 20px;
    margin-left: 8px;
    font-weight: 400;
}
.as-nav-badge-red {
    font-size: 10px;
    background: #fff0f0;
    color: #cc0000;
    padding: 4px 12px;
    border-radius: 20px;
    margin-left: 8px;
    font-weight: 400;
}

/* ── Body grid ── */
.as-body { display: grid; grid-template-columns: 320px 1fr; min-height: calc(100vh - 60px); }
.as-left { background: #ffffff; border-right: 1px solid #f0f0f0; padding: 36px 28px; }
.as-right { background: #fafafa; padding: 36px 40px; }

/* ── Section labels ── */
.as-lbl {
    font-size: 9px;
    font-weight: 500;
    letter-spacing: 0.18em;
    color: #c0c0c0;
    text-transform: uppercase;
    margin-bottom: 12px;
}

/* ── Vehicle ID card ── */
.as-vehicle-card {
    background: #ffffff;
    border: 1px solid #f0f0f0;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 20px;
}
.as-vehicle-make {
    font-size: 18px;
    font-weight: 400;
    color: #0a0a0a;
    letter-spacing: -0.3px;
}
.as-vehicle-detail {
    font-size: 12px;
    color: #999;
    font-weight: 300;
    margin-top: 3px;
}
.as-vehicle-price {
    font-size: 22px;
    font-weight: 300;
    color: #0a0a0a;
    margin-top: 12px;
    letter-spacing: -0.5px;
}
.as-vehicle-price-lbl {
    font-size: 9px;
    letter-spacing: 0.15em;
    color: #c0c0c0;
    text-transform: uppercase;
    margin-top: 2px;
}
.as-vehicle-confidence {
    display: inline-block;
    font-size: 9px;
    background: #f0faf2;
    color: #2d7d3a;
    padding: 3px 10px;
    border-radius: 20px;
    margin-top: 10px;
    letter-spacing: 0.05em;
}

/* ── Metrics ── */
.as-metrics {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-bottom: 24px;
}
.as-metric {
    background: #ffffff;
    border: 1px solid #f0f0f0;
    border-radius: 10px;
    padding: 14px 16px;
}
.as-metric-val {
    font-size: 24px;
    font-weight: 300;
    color: #0a0a0a;
    line-height: 1;
    letter-spacing: -0.5px;
}
.as-metric-val-red {
    font-size: 24px;
    font-weight: 300;
    color: #cc0000;
    line-height: 1;
    letter-spacing: -0.5px;
}
.as-metric-lbl {
    font-size: 9px;
    font-weight: 400;
    letter-spacing: 0.12em;
    color: #c0c0c0;
    text-transform: uppercase;
    margin-top: 5px;
}

/* ── Status banner ── */
.as-clean {
    background: #f0faf2;
    border: 1px solid #c3e6cb;
    border-radius: 8px;
    padding: 12px 18px;
    color: #2d7d3a;
    font-size: 12px;
    margin-bottom: 20px;
    letter-spacing: 0.02em;
}
.as-damaged {
    background: #fff8f0;
    border: 1px solid #fdd9b5;
    border-radius: 8px;
    padding: 12px 18px;
    color: #b45309;
    font-size: 12px;
    margin-bottom: 20px;
    letter-spacing: 0.02em;
}

/* ── Damage list ── */
.as-damage-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 11px 0;
    border-bottom: 1px solid #f5f5f5;
}
.as-damage-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.as-damage-name { font-size: 13px; font-weight: 400; color: #0a0a0a; text-transform: capitalize; flex: 1; }
.as-damage-count { font-size: 11px; color: #c0c0c0; font-family: 'DM Mono', monospace; }
.as-damage-sev { font-size: 8px; letter-spacing: 0.08em; text-transform: uppercase; padding: 2px 8px; border-radius: 20px; }
.as-damage-cost { font-size: 12px; font-weight: 400; color: #0a0a0a; min-width: 110px; text-align: right; font-family: 'DM Mono', monospace; }

.as-total {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 14px 0 0;
    margin-top: 6px;
    border-top: 1px solid #0a0a0a;
}
.as-total-label { font-size: 9px; font-weight: 500; letter-spacing: 0.16em; color: #0a0a0a; text-transform: uppercase; }
.as-total-val { font-size: 24px; font-weight: 300; color: #cc0000; letter-spacing: -0.5px; font-family: 'DM Mono', monospace; }

/* ── Offer box ── */
.as-offer {
    background: #0a0a0a;
    border-radius: 14px;
    padding: 26px 30px;
    margin-top: 28px;
}
.as-offer-row { display: flex; justify-content: space-between; align-items: baseline; padding: 5px 0; }
.as-offer-lbl { font-size: 9px; letter-spacing: 0.15em; color: #444; text-transform: uppercase; }
.as-offer-val { font-size: 13px; font-weight: 300; color: #fff; font-family: 'DM Mono', monospace; }
.as-offer-val-red { font-size: 13px; font-weight: 300; color: #cc0000; font-family: 'DM Mono', monospace; }
.as-offer-divider { height: 1px; background: #1a1a1a; margin: 12px 0; }
.as-offer-total-lbl { font-size: 9px; letter-spacing: 0.18em; color: #555; text-transform: uppercase; }
.as-offer-total-val { font-size: 38px; font-weight: 300; color: #fff; letter-spacing: -1.5px; }

/* ── Note ── */
.as-note { font-size: 10px; font-weight: 300; color: #c0c0c0; margin-top: 8px; line-height: 1.5; }

/* ── Empty state ── */
.as-empty {
    background: #ffffff;
    border: 1px solid #f0f0f0;
    border-radius: 14px;
    padding: 80px 40px;
    text-align: center;
}
.as-empty-icon { font-size: 32px; opacity: 0.08; margin-bottom: 14px; }
.as-empty-text { font-size: 13px; font-weight: 300; color: #c0c0c0; letter-spacing: 0.04em; }

/* ── AI thinking ── */
.as-ai-thinking {
    background: #f8f8ff;
    border: 1px solid #e8e8ff;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 12px;
    color: #5555aa;
    margin-bottom: 16px;
    font-style: italic;
}

/* ── Footer ── */
.as-footer {
    border-top: 1px solid #f0f0f0;
    padding: 14px 40px;
    background: #fff;
    font-size: 9px;
    letter-spacing: 0.18em;
    color: #c0c0c0;
    text-transform: uppercase;
    display: flex;
    justify-content: space-between;
}

/* ── Streamlit widget overrides ── */
.stFileUploader > div {
    border: 1px dashed #e8e8e8 !important;
    border-radius: 10px !important;
    background: #fafafa !important;
    padding: 20px !important;
}
.stFileUploader label {
    font-size: 9px !important; font-weight: 500 !important;
    letter-spacing: 0.18em !important; color: #c0c0c0 !important;
    text-transform: uppercase !important;
}
.stSlider label {
    font-size: 9px !important; font-weight: 500 !important;
    letter-spacing: 0.18em !important; color: #c0c0c0 !important;
    text-transform: uppercase !important;
}
.stSlider [role="slider"] { background: #0a0a0a !important; border-color: #0a0a0a !important; }
.stSelectbox label {
    font-size: 9px !important; font-weight: 500 !important;
    letter-spacing: 0.18em !important; color: #c0c0c0 !important;
    text-transform: uppercase !important;
}
div[data-baseweb="select"] > div {
    border: 1px solid #e8e8e8 !important;
    border-radius: 8px !important;
    background: #ffffff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 400 !important;
}
.stNumberInput label {
    font-size: 9px !important; font-weight: 500 !important;
    letter-spacing: 0.18em !important; color: #c0c0c0 !important;
    text-transform: uppercase !important;
}
.stNumberInput input {
    border: 1px solid #e8e8e8 !important;
    border-radius: 8px !important;
    background: #ffffff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 16px !important;
    font-weight: 300 !important;
    padding: 8px 12px !important;
}
.stTextInput input {
    border: 1px solid #e8e8e8 !important;
    border-radius: 8px !important;
    background: #ffffff !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 14px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 10px 14px !important;
}
.stTextInput label {
    font-size: 9px !important; font-weight: 500 !important;
    letter-spacing: 0.18em !important; color: #c0c0c0 !important;
    text-transform: uppercase !important;
}
div[data-testid="stImage"] img {
    border-radius: 10px !important;
    border: 1px solid #f0f0f0 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH   = os.environ.get("AUTOSHOT_MODEL", "runs/detect/runs/detect/autoshot_combined_v12/weights/best.pt")
FALLBACK     = "runs/detect/runs/detect/autoshot_v4_final/weights/best.pt"
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLASS_NAMES  = ['dent','scratch','crack','glass shatter','lamp broken','tire flat','paint damage']
CONF_DEFAULT = 0.40

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
CNAME = {'es':'Spain','pt':'Portugal','br':'Brazil','us':'USA','mx':'México'}
CFLAG = {'es':'🇪🇸','pt':'🇵🇹','br':'🇧🇷','us':'🇺🇸','mx':'🇲🇽'}

@st.cache_resource
def load_model():
    for p in [MODEL_PATH, FALLBACK]:
        if os.path.exists(p):
            return YOLO(p), p
    return None, None

model, model_path = load_model()

# ── Claude Vision: identify vehicle + market value ─────────────────────────────
def identify_vehicle_with_claude(image_pil, country_code, sym):
    """Send image to Claude claude-sonnet-4-20250514 for vehicle ID + market value estimate."""
    if not ANTHROPIC_KEY:
        return None

    # Encode image to base64
    import io
    buf = io.BytesIO()
    image_pil.save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    country_name = CNAME.get(country_code, "Spain")
    currency = sym

    prompt = f"""You are an automotive expert. Look at this vehicle photo and respond ONLY with a JSON object — no preamble, no markdown, just raw JSON.

Identify the vehicle and estimate its current market value in {country_name} ({currency}).

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
  "confidence": "High",
  "reasoning": "One sentence explaining the valuation."
}}

If you cannot identify the vehicle clearly, set confidence to "Low" and use conservative estimates.
Base market values on current {country_name} used car prices ({currency}).
Return ONLY the JSON object."""

    try:
        response = requests.post(
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
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_b64,
                            }
                        },
                        {"type": "text", "text": prompt}
                    ]
                }]
            },
            timeout=30
        )
        data = response.json()
        raw = data["content"][0]["text"].strip()
        # Strip any accidental markdown fences
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception:
        return None

# ── Nav ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="as-nav">
  <div style="display:flex;align-items:center;gap:14px">
    <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
      <circle cx="14" cy="14" r="13" stroke="#0a0a0a" stroke-width="1.2"/>
      <circle cx="14" cy="14" r="7" stroke="#0a0a0a" stroke-width="0.8"/>
      <circle cx="14" cy="14" r="2.5" fill="#cc0000"/>
      <line x1="14" y1="1" x2="14" y2="7" stroke="#0a0a0a" stroke-width="1.2"/>
      <line x1="14" y1="21" x2="14" y2="27" stroke="#0a0a0a" stroke-width="1.2"/>
      <line x1="1" y1="14" x2="7" y2="14" stroke="#0a0a0a" stroke-width="1.2"/>
      <line x1="21" y1="14" x2="27" y2="14" stroke="#0a0a0a" stroke-width="1.2"/>
    </svg>
    <div>
      <div class="as-logo-name">Autoshot</div>
      <div class="as-logo-sub">Vehicle Damage Intelligence</div>
    </div>
  </div>
  <div style="display:flex;align-items:center">
    <span class="as-nav-badge">AI-Powered</span>
    <span class="as-nav-badge-red">v12 · mAP 0.601</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Two-column layout ──────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.8], gap="large")

# ── LEFT: Controls ─────────────────────────────────────────────────────────────
with col_left:
    st.markdown('<div style="padding:4px 0">', unsafe_allow_html=True)

    # Market
    st.markdown('<div class="as-lbl">Market</div>', unsafe_allow_html=True)
    country = st.selectbox(
        "Country", list(CNAME.keys()),
        format_func=lambda x: f"{CFLAG[x]}  {CNAME[x]}",
        label_visibility="collapsed"
    )
    sym = SYM[country]

    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

    # License plate (optional)
    st.markdown('<div class="as-lbl">License Plate (optional)</div>', unsafe_allow_html=True)
    plate_input = st.text_input(
        "Plate", placeholder="1234 ABC",
        label_visibility="collapsed",
        help="Enter plate for future API lookup. Currently uses AI vision identification."
    )
    if plate_input:
        st.markdown(f'<div class="as-note">Plate noted: <strong>{plate_input.upper()}</strong> · AI vision will identify the vehicle from your photos.</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

    # Upload
    st.markdown('<div class="as-lbl">Vehicle Photos</div>', unsafe_allow_html=True)
    uploads = st.file_uploader(
        "Photos", type=["jpg","jpeg","png","webp"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

    # Confidence
    st.markdown('<div class="as-lbl">Detection Threshold</div>', unsafe_allow_html=True)
    conf = st.slider("Confidence", 0.10, 0.90, CONF_DEFAULT, 0.05, label_visibility="collapsed")

    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

    # Dealer calculator
    st.markdown('<div class="as-lbl">Dealer Offer Calculator</div>', unsafe_allow_html=True)

    # Show AI-suggested value if available
    if "vehicle_data" in st.session_state and st.session_state.vehicle_data:
        vd = st.session_state.vehicle_data
        suggested = vd.get("market_value_mid", 12000)
        st.markdown(f'<div class="as-note" style="margin-bottom:8px">AI estimate: {sym} {vd.get("market_value_low",0):,} – {sym} {vd.get("market_value_high",0):,}</div>', unsafe_allow_html=True)
        market_val = st.number_input(f"Market value ({sym})", min_value=0, value=suggested, step=500)
    else:
        market_val = st.number_input(f"Market value ({sym})", min_value=0, value=12000, step=500)

    margin = st.slider("Dealer margin", 5, 30, 15, format="%d%%")

    st.markdown('</div>', unsafe_allow_html=True)

# ── RIGHT: Results ─────────────────────────────────────────────────────────────
with col_right:
    st.markdown('<div style="padding:4px 0">', unsafe_allow_html=True)

    if not uploads:
        st.markdown("""
        <div class="as-empty">
          <div class="as-empty-icon">⊙</div>
          <div class="as-empty-text">Upload vehicle photos to begin analysis</div>
        </div>
        """, unsafe_allow_html=True)

    elif model is None:
        st.error(f"Model not found. Check AUTOSHOT_MODEL path: {MODEL_PATH}")

    else:
        # ── Step 1: Claude Vision — vehicle identification ──────────────────────
        first_image = Image.open(uploads[0]).convert("RGB")

        # Only re-run if uploads changed
        upload_key = "_".join([f.name for f in uploads])
        if st.session_state.get("last_upload_key") != upload_key:
            st.session_state["last_upload_key"] = upload_key
            st.session_state["vehicle_data"] = None

        if ANTHROPIC_KEY and st.session_state.get("vehicle_data") is None:
            with st.spinner("Identifying vehicle with AI..."):
                vd = identify_vehicle_with_claude(first_image, country, sym)
                st.session_state["vehicle_data"] = vd
        elif not ANTHROPIC_KEY:
            st.session_state["vehicle_data"] = None

        # Show vehicle ID card
        vd = st.session_state.get("vehicle_data")
        if vd:
            conf_color = {"High": "#f0faf2", "Medium": "#fffbeb", "Low": "#fff5f5"}
            conf_text  = {"High": "#2d7d3a",  "Medium": "#92400e", "Low": "#cc0000"}
            c = vd.get("confidence", "Medium")
            st.markdown(f"""
            <div class="as-vehicle-card">
              <div class="as-lbl">Vehicle Identified</div>
              <div class="as-vehicle-make">{vd.get('make','')} {vd.get('model','')}</div>
              <div class="as-vehicle-detail">{vd.get('year_range','')} · {vd.get('trim','')} · {vd.get('fuel','')} · {vd.get('body','')}</div>
              <div class="as-vehicle-price">{sym} {vd.get('market_value_low',0):,} – {vd.get('market_value_high',0):,}</div>
              <div class="as-vehicle-price-lbl">Estimated market value · {CNAME[country]}</div>
              <div style="margin-top:8px">
                <span style="font-size:9px;background:{conf_color.get(c,'#f5f5f5')};color:{conf_text.get(c,'#888')};padding:3px 10px;border-radius:20px;letter-spacing:0.05em">
                  {c} confidence · {vd.get('reasoning','')}
                </span>
              </div>
            </div>
            """, unsafe_allow_html=True)
        elif not ANTHROPIC_KEY:
            st.markdown('<div class="as-note" style="margin-bottom:16px">Set ANTHROPIC_API_KEY environment variable to enable AI vehicle identification.</div>', unsafe_allow_html=True)

        # ── Step 2: YOLO damage detection ──────────────────────────────────────
        all_classes, all_confs, annotated_imgs = [], [], []
        with st.spinner("Analysing damage..."):
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

        # Annotated images grid
        img_cols = st.columns(min(len(annotated_imgs), 3))
        for i, (name, img) in enumerate(annotated_imgs):
            with img_cols[i % 3]:
                st.image(img, use_container_width=True)

        st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

        # ── Step 3: Results ─────────────────────────────────────────────────────
        total_low, total_high = 0, 0

        if not all_classes:
            st.markdown('<div class="as-clean">✓ No damage detected across all photos</div>', unsafe_allow_html=True)

        else:
            counts   = Counter(all_classes)
            avg_conf = float(np.mean(all_confs))
            n        = len(uploads)

            st.markdown(f'<div class="as-damaged">⚠ {len(all_classes)} damage instance{"s" if len(all_classes)>1 else ""} detected across {n} photo{"s" if n>1 else ""}</div>', unsafe_allow_html=True)

            # Metrics
            st.markdown(f"""
            <div class="as-metrics">
              <div class="as-metric">
                <div class="as-metric-val-red">{len(all_classes)}</div>
                <div class="as-metric-lbl">Defects</div>
              </div>
              <div class="as-metric">
                <div class="as-metric-val">{len(counts)}</div>
                <div class="as-metric-lbl">Types</div>
              </div>
              <div class="as-metric">
                <div class="as-metric-val">{avg_conf:.0%}</div>
                <div class="as-metric-lbl">Confidence</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Damage breakdown
            st.markdown('<div class="as-lbl">Damage Breakdown & Repair Estimate</div>', unsafe_allow_html=True)
            costs_table = REPAIR[country]

            class_confs = {}
            for cls, c in zip(all_classes, all_confs):
                class_confs.setdefault(cls, []).append(c)

            items_html = ""
            for cls, count in sorted(counts.items(), key=lambda x: -max(class_confs[x[0]])):
                sev, color = SEVERITY.get(cls, ('Medium','#fb923c'))
                low, high  = costs_table.get(cls, (100,300))
                total_low  += low  * count
                total_high += high * count
                sev_bg = {"Critical":"#fff0f0","High":"#fff5f0","Medium":"#fffbeb","Low":"#fafff0"}.get(sev,"#f5f5f5")
                items_html += f"""
                <div class="as-damage-item">
                  <div class="as-damage-dot" style="background:{color}"></div>
                  <div class="as-damage-name">{cls}</div>
                  <div class="as-damage-count">×{count}</div>
                  <div class="as-damage-sev" style="color:{color};background:{sev_bg}">{sev}</div>
                  <div class="as-damage-cost">{sym} {low*count:,}–{high*count:,}</div>
                </div>"""

            st.markdown(f"""
            {items_html}
            <div class="as-total">
              <span class="as-total-label">Total Repair Estimate</span>
              <span class="as-total-val">{sym} {total_low:,}–{total_high:,}</span>
            </div>
            <div class="as-note">Indicative prices for {CFLAG[country]} {CNAME[country]}. Labour rates vary by region.</div>
            """, unsafe_allow_html=True)

        # ── Step 4: Offer calculator ────────────────────────────────────────────
        st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
        repair_mid = (total_low + total_high) // 2
        margin_amt = (market_val - repair_mid) * margin / 100
        max_offer  = max(0, market_val - repair_mid - int(margin_amt))

        # Show AI-sourced value note
        ai_note = ""
        if vd and vd.get("market_value_mid"):
            ai_note = f'<div style="font-size:9px;color:#888;margin-bottom:6px;letter-spacing:0.05em">Market value pre-filled from AI vehicle identification</div>'

        st.markdown(f"""
        {ai_note}
        <div class="as-offer">
          <div class="as-offer-row">
            <span class="as-offer-lbl">Market Value</span>
            <span class="as-offer-val">{sym} {market_val:,}</span>
          </div>
          <div class="as-offer-row">
            <span class="as-offer-lbl">Repair Estimate (mid)</span>
            <span class="as-offer-val-red">− {sym} {repair_mid:,}</span>
          </div>
          <div class="as-offer-row">
            <span class="as-offer-lbl">Dealer Margin {margin}%</span>
            <span class="as-offer-val-red">− {sym} {int(margin_amt):,}</span>
          </div>
          <div class="as-offer-divider"></div>
          <div class="as-offer-row" style="margin-top:4px">
            <span class="as-offer-total-lbl">Max Purchase Offer</span>
            <span class="as-offer-total-val">{sym} {int(max_offer):,}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
model_name = os.path.basename(os.path.dirname(os.path.dirname(model_path))) if model_path else "No model loaded"
st.markdown(f"""
<div class="as-footer">
  <span>Autoshot · Vehicle Damage Intelligence</span>
  <span>{model_name} · 7 damage classes</span>
</div>
""", unsafe_allow_html=True)
