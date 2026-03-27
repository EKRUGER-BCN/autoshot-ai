# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Autoshot is an AI-powered vehicle damage assessment system for used car dealers. It detects damage via YOLO object detection and identifies vehicles using Claude Vision, then calculates repair costs and maximum purchase offers across multiple markets (ES, PT, BR, US, MX).

## Architecture

Two independent components with separate dependencies:

**`api/`** — FastAPI backend
- `main.py`: REST endpoints (`/identify` for Claude Vision vehicle ID, `/detect` for YOLO damage detection), serves `index.html` as static frontend
- Loads YOLO model at startup; exposes annotated images as base64 in JSON responses

**`streamlit_app/`** — Streamlit frontend (standalone, can run without the API)
- `app.py`: 3-step wizard — configure market/plate/margin → upload 4 photos → view assessment results
- Runs YOLO locally (same model) and calls Claude API directly (not via `api/`)

Both components share the same YOLO model weights and call the Claude API independently using the same prompt structure.

## Running

**API backend:**
```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Streamlit app:**
```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

## Required Environment Variables

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Required for vehicle identification (Claude Vision) |
| `AUTOSHOT_MODEL` | Path to YOLO `.pt` weights (optional — falls back to `runs/detect/runs/detect/autoshot_combined_v12/weights/best.pt`, then `autoshot_v4_final`) |

## Key Technical Details

- **Claude model used:** `claude-sonnet-4-20250514` (called via raw HTTP, not SDK)
- **YOLO classes (7):** dent, scratch, crack, glass shatter, lamp broken, tire flat, paint damage
- **Detection confidence threshold:** 0.40
- **Model weights** (`.pt` files) are gitignored — must be provided separately
- Repair costs are hardcoded per country/damage class in both `api/main.py` and `streamlit_app/app.py` — update both if changing pricing
