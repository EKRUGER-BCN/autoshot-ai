from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2
import base64
import os
import io
import json
import requests
from collections import Counter
from typing import List

app = FastAPI(title="Autoshot API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH    = os.environ.get("AUTOSHOT_MODEL", "runs/detect/runs/detect/autoshot_combined_v12/weights/best.pt")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLASS_NAMES   = ['dent', 'scratch', 'crack', 'glass shatter', 'lamp broken', 'tire flat', 'paint damage']
CONF_DEFAULT  = 0.40

REPAIR_COSTS = {
    'es': {'dent':(150,400),'scratch':(80,220),'crack':(200,500),'glass shatter':(300,900),'lamp broken':(150,600),'tire flat':(80,200),'paint damage':(200,500)},
    'pt': {'dent':(130,370),'scratch':(70,200),'crack':(180,460),'glass shatter':(270,830),'lamp broken':(140,550),'tire flat':(75,185),'paint damage':(185,460)},
    'br': {'dent':(800,2200),'scratch':(400,1200),'crack':(1100,2700),'glass shatter':(1600,4900),'lamp broken':(800,3200),'tire flat':(400,1100),'paint damage':(1100,2700)},
    'us': {'dent':(160,430),'scratch':(85,240),'crack':(215,540),'glass shatter':(325,970),'lamp broken':(160,650),'tire flat':(85,215),'paint damage':(215,540)},
    'mx': {'dent':(2900,7800),'scratch':(1600,4300),'crack':(3900,9800),'glass shatter':(5800,17500),'lamp broken':(2900,11700),'tire flat':(1600,3900),'paint damage':(3900,9800)},
}
CURRENCY     = {'es':'€','pt':'€','br':'R$','us':'$','mx':'MX$'}
COUNTRY_NAME = {'es':'Spain','pt':'Portugal','br':'Brazil','us':'United States','mx':'México'}

model = None

def get_model():
    global model
    if model is None:
        for p in [MODEL_PATH, "runs/detect/runs/detect/autoshot_v4_final/weights/best.pt"]:
            if os.path.exists(p):
                model = YOLO(p)
                break
    return model

def img_to_base64(img_array: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', img_array)
    return base64.b64encode(buffer).decode('utf-8')

# ── Serve frontend ─────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)

# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    m = get_model()
    return {"status": "ok", "model_loaded": m is not None, "model_path": MODEL_PATH}

# ── Claude Vision: identify vehicle ───────────────────────────────────────────
@app.post("/identify")
async def identify(
    file: UploadFile = File(...),
    country: str = Query(default="es"),
):
    if not ANTHROPIC_KEY:
        return JSONResponse(status_code=200, content={"error": "No ANTHROPIC_API_KEY set", "vehicle": None})

    contents = await file.read()
    img_b64  = base64.b64encode(contents).decode("utf-8")
    sym      = CURRENCY.get(country, "€")
    cname    = COUNTRY_NAME.get(country, "Spain")

    prompt = f"""You are an automotive expert. Analyse this vehicle photo and respond ONLY with raw JSON — no markdown, no preamble.

Return this exact structure:
{{
  "make": "e.g. Volkswagen",
  "model": "e.g. Golf",
  "year_range": "e.g. 2018-2020",
  "trim": "e.g. 1.6 TDI",
  "fuel": "e.g. Diesel",
  "body": "e.g. Hatchback",
  "market_value_low": 11000,
  "market_value_high": 14500,
  "market_value_mid": 12500,
  "reasoning": "One sentence on valuation."
}}

Base market values on current {cname} used car prices ({sym}). Return ONLY the JSON."""

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": "claude-sonnet-4-20250514", "max_tokens": 512, "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                {"type": "text", "text": prompt}
            ]}]},
            timeout=30
        )
        raw = resp.json()["content"][0]["text"].strip().replace("```json","").replace("```","").strip()
        vehicle = json.loads(raw)
        return {"vehicle": vehicle}
    except Exception as e:
        return JSONResponse(status_code=200, content={"error": str(e), "vehicle": None})

# ── Detect damage ──────────────────────────────────────────────────────────────
@app.post("/detect")
async def detect(
    files: List[UploadFile] = File(...),
    country: str = Query(default="es"),
    conf: float = Query(default=CONF_DEFAULT),
    margin: float = Query(default=0.15),
    market_value: float = Query(default=0),
):
    m = get_model()
    if m is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded"})

    all_detections = []
    annotated_images = []

    for file in files:
        contents = await file.read()
        img      = Image.open(io.BytesIO(contents)).convert("RGB")
        img_arr  = np.array(img)
        results  = m.predict(img_arr, conf=conf, verbose=False)[0]

        if results.boxes and len(results.boxes) > 0:
            for box, cls, confidence in zip(
                results.boxes.xyxy.cpu().numpy(),
                results.boxes.cls.cpu().numpy(),
                results.boxes.conf.cpu().numpy()
            ):
                cls_id   = int(cls)
                cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
                all_detections.append({
                    "file": file.filename,
                    "class_name": cls_name,
                    "confidence": round(float(confidence), 3),
                    "bbox": [round(float(x), 1) for x in box],
                })

        annotated_bgr = results.plot(line_width=2, pil=False, img=img_arr.copy())
        annotated_images.append({
            "filename": file.filename,
            "image_b64": img_to_base64(annotated_bgr),
        })

    counts   = Counter(d["class_name"] for d in all_detections)
    costs    = REPAIR_COSTS.get(country, REPAIR_COSTS["es"])
    sym      = CURRENCY.get(country, "€")

    breakdown = []
    total_low, total_high = 0, 0
    for cls_name, count in counts.items():
        low, high = costs.get(cls_name, (100, 300))
        breakdown.append({"class": cls_name, "count": count, "cost_low": low*count, "cost_high": high*count})
        total_low  += low  * count
        total_high += high * count

    repair_mid = (total_low + total_high) / 2
    margin_amt = (market_value - repair_mid) * margin if market_value > 0 else 0
    max_offer  = max(0, market_value - repair_mid - margin_amt) if market_value > 0 else None

    return {
        "total_detections": len(all_detections),
        "detections": all_detections,
        "repair": {
            "breakdown": breakdown,
            "total_low": total_low,
            "total_high": total_high,
            "currency": sym,
        },
        "offer": {
            "market_value": market_value,
            "repair_mid": round(repair_mid),
            "margin_pct": margin * 100,
            "margin_amount": round(margin_amt),
            "max_offer": round(max_offer) if max_offer is not None else None,
            "currency": sym,
        } if market_value > 0 else None,
        "annotated_images": annotated_images,
    }
