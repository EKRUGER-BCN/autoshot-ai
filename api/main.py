from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from ultralytics import YOLO
import numpy as np
import cv2
import base64
import os
import io
import json
import requests
import datetime
import shutil
from collections import Counter
from typing import List

COLLECT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "collected")

def save_collected(file_bytes: bytes, filename: str):
    try:
        day = datetime.date.today().isoformat()
        dest = os.path.join(COLLECT_DIR, day)
        os.makedirs(dest, exist_ok=True)
        ts = datetime.datetime.now().strftime("%H%M%S")
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)
        with open(os.path.join(dest, f"{ts}_{safe}"), "wb") as f:
            f.write(file_bytes)
    except Exception:
        pass  # never block the request over storage

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
CONF_DEFAULT  = 0.25

REPAIR_COSTS = {
    'es': {'dent':(150,400),'scratch':(80,220),'crack':(200,500),'glass shatter':(300,900),'lamp broken':(150,600),'tire flat':(80,200),'paint damage':(200,500)},
    'pt': {'dent':(130,370),'scratch':(70,200),'crack':(180,460),'glass shatter':(270,830),'lamp broken':(140,550),'tire flat':(75,185),'paint damage':(185,460)},
    'fr': {'dent':(180,450),'scratch':(90,250),'crack':(230,580),'glass shatter':(350,1000),'lamp broken':(180,700),'tire flat':(90,230),'paint damage':(230,580)},
    'de': {'dent':(200,500),'scratch':(100,280),'crack':(250,620),'glass shatter':(380,1100),'lamp broken':(200,750),'tire flat':(100,250),'paint damage':(250,620)},
    'it': {'dent':(165,430),'scratch':(85,230),'crack':(220,550),'glass shatter':(330,960),'lamp broken':(165,660),'tire flat':(85,215),'paint damage':(220,550)},
    'nl': {'dent':(170,430),'scratch':(85,235),'crack':(215,540),'glass shatter':(320,950),'lamp broken':(170,650),'tire flat':(85,215),'paint damage':(215,540)},
    'br': {'dent':(800,2200),'scratch':(400,1200),'crack':(1100,2700),'glass shatter':(1600,4900),'lamp broken':(800,3200),'tire flat':(400,1100),'paint damage':(1100,2700)},
    'us': {'dent':(160,430),'scratch':(85,240),'crack':(215,540),'glass shatter':(325,970),'lamp broken':(160,650),'tire flat':(85,215),'paint damage':(215,540)},
    'mx': {'dent':(2900,7800),'scratch':(1600,4300),'crack':(3900,9800),'glass shatter':(5800,17500),'lamp broken':(2900,11700),'tire flat':(1600,3900),'paint damage':(3900,9800)},
}
CURRENCY     = {'es':'€','pt':'€','fr':'€','de':'€','it':'€','nl':'€','br':'R$','us':'$','mx':'MX$'}
COUNTRY_NAME = {'es':'Spain','pt':'Portugal','fr':'France','de':'Germany','it':'Italy','nl':'Netherlands','br':'Brazil','us':'United States','mx':'México'}

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
def _read_html(name: str) -> HTMLResponse:
    path = os.path.join(os.path.dirname(__file__), name)
    if os.path.exists(path):
        with open(path, "r") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content=f"<h1>{name} not found</h1>", status_code=404)

def _static(name: str, media_type: str):
    path = os.path.join(os.path.dirname(__file__), name)
    if os.path.exists(path):
        return FileResponse(path, media_type=media_type)
    return JSONResponse(status_code=404, content={"error": f"{name} not found"})

@app.get("/", response_class=HTMLResponse)
async def serve_landing():
    return _read_html("landing.html")

@app.get("/app", response_class=HTMLResponse)
async def serve_app():
    return _read_html("index.html")

@app.get("/manifest.json")
async def serve_manifest():
    return _static("manifest.json", "application/manifest+json")

@app.get("/sw.js")
async def serve_sw():
    return _static("sw.js", "application/javascript")

@app.get("/icon.svg")
async def serve_icon():
    return _static("icon.svg", "image/svg+xml")

@app.get("/icon-maskable.svg")
async def serve_icon_maskable():
    return _static("icon-maskable.svg", "image/svg+xml")

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
    lang: str = Query(default="en"),
):
    if not ANTHROPIC_KEY:
        return JSONResponse(status_code=200, content={"error": "No ANTHROPIC_API_KEY set", "vehicle": None})

    contents = await file.read()
    img_b64  = base64.b64encode(contents).decode("utf-8")
    sym      = CURRENCY.get(country, "€")
    cname    = COUNTRY_NAME.get(country, "Spain")
    lang_name = LANG_NAME.get(lang, "English")

    prompt = f"""You are an automotive expert. Analyse this vehicle photo and respond ONLY with raw JSON — no markdown, no preamble.

Return this exact structure:
{{
  "make": "e.g. Volkswagen",
  "model": "e.g. Golf",
  "year_range": "e.g. 2019-2020",
  "trim": "e.g. 1.6 TDI",
  "fuel": "e.g. Diesel",
  "body": "e.g. Hatchback",
  "market_value_low": 11000,
  "market_value_high": 14500,
  "market_value_mid": 12500,
  "reasoning": "One sentence on valuation written in {lang_name}."
}}

Be precise on year_range — maximum 2 years span based on visible facelift/generation details.
Base market values on current {cname} used car prices ({sym}). Return ONLY the JSON."""

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": "claude-sonnet-4-20250514", "max_tokens": 512, "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                {"type": "text", "text": prompt}
            ]}]},
            timeout=55
        )
        raw = resp.json()["content"][0]["text"].strip().replace("```json","").replace("```","").strip()
        vehicle = json.loads(raw)
        return {"vehicle": vehicle}
    except Exception as e:
        return JSONResponse(status_code=200, content={"error": str(e), "vehicle": None})

# ── Claude Vision: analyse damage ─────────────────────────────────────────────
LANG_NAME = {'en':'English','es':'Spanish','fr':'French','de':'German','it':'Italian','nl':'Dutch','pt':'Portuguese'}

@app.post("/analyze")
async def analyze(
    files: List[UploadFile] = File(...),
    country: str = Query(default="es"),
    lang: str = Query(default="en"),
    slots: str = Query(default=""),  # comma-separated slot names e.g. "front,rear,left,right"
):
    if not ANTHROPIC_KEY:
        return JSONResponse(status_code=200, content={"error": "No ANTHROPIC_API_KEY set", "analysis": None})

    cname    = COUNTRY_NAME.get(country, "Spain")
    sym      = CURRENCY.get(country, "€")
    costs    = REPAIR_COSTS.get(country, REPAIR_COSTS["es"])
    lang_name = LANG_NAME.get(lang, "English")

    slot_list   = [s.strip() for s in slots.split(",")] if slots else []
    ALL_SLOTS   = ["front", "rear", "left", "right"]
    SLOT_LABELS = {"front": "front", "rear": "rear", "left": "left side", "right": "right side"}

    all_items = []
    all_imgs  = []  # store processed PIL images for annotation later
    for idx, file in enumerate(files):
        slot_name  = SLOT_LABELS.get(slot_list[idx], "vehicle") if idx < len(slot_list) else "vehicle"
        contents  = await file.read()
        img_fixed = ImageOps.exif_transpose(Image.open(io.BytesIO(contents))).convert("RGB")
        all_imgs.append(img_fixed)
        buf = io.BytesIO()
        img_fixed.save(buf, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        prompt = f"""You are a professional automotive damage assessor with 20 years of experience evaluating used cars for dealerships in {cname}. Respond in {lang_name}.

This is the {slot_name} photo of the vehicle.

Carefully examine this vehicle photo and identify ALL visible damage — including subtle scratches, paint chips, scuffs, and surface damage that automated systems might miss.

Also assess photo coverage: how much of the vehicle's {slot_name} surface is clearly visible and in frame (0–100).

Respond ONLY with raw JSON — no markdown, no preamble:
{{
  "damage_found": true,
  "overall_condition": "good",
  "damage_items": [
    {{
      "type": "scratch|dent|crack|paint_damage|rust|glass_damage|lamp_damage|tire_damage|other",
      "location": "e.g. rear left door",
      "severity": "minor|moderate|severe",
      "description": "e.g. 15cm surface scratch, paint intact"
    }}
  ],
  "repair_urgency": "none|cosmetic|soon|urgent",
  "notes": "One sentence overall assessment.",
  "coverage_pct": 85,
  "coverage_note": "Brief note if coverage is poor, else empty string."
}}

If no damage is visible respond with damage_found: false and empty damage_items array.
Write all location, description, notes, and coverage_note text in {lang_name}. Return ONLY the JSON."""

        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": "claude-sonnet-4-20250514", "max_tokens": 1024, "messages": [{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                    {"type": "text", "text": prompt}
                ]}]},
                timeout=55
            )
            raw = resp.json()["content"][0]["text"].strip().replace("```json","").replace("```","").strip()
            result = json.loads(raw)
            for item in result.get("damage_items", []):
                item["photo"] = file.filename
            all_items.append(result)
        except Exception as e:
            all_items.append({"error": str(e), "damage_found": False, "damage_items": []})

    # Merge results across all photos
    merged_items = []
    for r in all_items:
        merged_items.extend(r.get("damage_items", []))

    # Map Claude damage types to repair cost estimates
    TYPE_MAP = {
        "scratch": "scratch", "dent": "dent", "crack": "crack",
        "paint_damage": "paint damage", "glass_damage": "glass shatter",
        "lamp_damage": "lamp broken", "tire_damage": "tire flat",
        "rust": "paint damage", "other": "dent",
    }
    SEVERITY_MULT = {"minor": 0.5, "moderate": 1.0, "severe": 1.5}

    total_low, total_high = 0, 0
    for item in merged_items:
        mapped = TYPE_MAP.get(item.get("type","other"), "dent")
        low, high = costs.get(mapped, (100, 300))
        mult = SEVERITY_MULT.get(item.get("severity","moderate"), 1.0)
        item["cost_low"]  = round(low  * mult)
        item["cost_high"] = round(high * mult)
        total_low  += item["cost_low"]
        total_high += item["cost_high"]

    overall_conditions = [r.get("overall_condition","fair") for r in all_items if r.get("damage_found")]
    notes = [r.get("notes","") for r in all_items if r.get("notes")]

    # Build coverage report
    coverage_panels = []
    for idx, r in enumerate(all_items):
        slot = slot_list[idx] if idx < len(slot_list) else f"photo_{idx+1}"
        coverage_panels.append({
            "slot": slot,
            "pct": int(r.get("coverage_pct", 0)),
            "note": r.get("coverage_note", ""),
        })
    # Add 0% for slots that weren't photographed
    photographed_slots = set(slot_list[:len(files)])
    for s in ALL_SLOTS:
        if s not in photographed_slots:
            coverage_panels.append({"slot": s, "pct": 0, "note": "Not photographed"})
    # Sort in canonical order
    order = {s: i for i, s in enumerate(ALL_SLOTS)}
    coverage_panels.sort(key=lambda p: order.get(p["slot"], 99))
    overall_coverage = round(sum(p["pct"] for p in coverage_panels) / max(len(ALL_SLOTS), 1))

    # Build annotated images with Claude findings overlaid
    annotated_images = []
    for file, img, result in zip(files, all_imgs, all_items):
        arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        findings = result.get("damage_items", [])
        if findings:
            h, w = arr.shape[:2]
            row_h = 28
            overlay_h = 10 + len(findings) * row_h + 10
            panel = arr.copy()
            cv2.rectangle(panel, (0, h - overlay_h), (w, h), (15, 15, 15), -1)
            cv2.addWeighted(panel, 0.82, arr, 0.18, 0, arr)
            SEV_COLORS_BGR = {"severe": (0, 0, 200), "moderate": (0, 130, 255), "minor": (0, 200, 255)}
            y = h - overlay_h + row_h - 4
            for item in findings:
                color = SEV_COLORS_BGR.get(item.get("severity", "minor"), (0, 200, 255))
                label = f"{item.get('type','').replace('_',' ').title()}  {item.get('location','')}"
                cv2.putText(arr, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
                y += row_h
        _, buf = cv2.imencode('.jpg', arr)
        annotated_images.append({
            "filename": file.filename,
            "image_b64": base64.b64encode(buf).decode("utf-8"),
        })

    return {
        "damage_found": len(merged_items) > 0,
        "damage_items": merged_items,
        "total_items": len(merged_items),
        "repair_low": total_low,
        "repair_high": total_high,
        "currency": sym,
        "overall_condition": overall_conditions[0] if overall_conditions else "good",
        "notes": " | ".join(notes) if notes else None,
        "annotated_images": annotated_images,
        "coverage": {"overall_pct": overall_coverage, "panels": coverage_panels},
    }

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
        save_collected(contents, file.filename)
        img      = ImageOps.exif_transpose(Image.open(io.BytesIO(contents))).convert("RGB")
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

    # Deduplicate across photos: the same damage appears in multiple angles.
    # Take the max count per class from any single photo, not the sum across all photos.
    per_photo: dict = {}
    for d in all_detections:
        per_photo.setdefault(d["file"], Counter())[d["class_name"]] += 1
    counts: Counter = Counter()
    for photo_counts in per_photo.values():
        for cls, cnt in photo_counts.items():
            counts[cls] = max(counts[cls], cnt)

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
