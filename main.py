import os, joblib, nltk
import re, random, httpx

from fastapi import FastAPI, Query
from pydantic import BaseModel
from nltk.sentiment import SentimentIntensityAnalyzer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from urllib.parse import unquote

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

MODEL_PATH = os.getenv("MODEL_PATH", "pipeline_v1.joblib")
pipe = joblib.load(MODEL_PATH)
sia = SentimentIntensityAnalyzer()

app = FastAPI(title="Buzz Predictor API")

# === Hugging Face config ===
HF_TOKEN = os.getenv("HF_TOKEN")                           # מגיע מה-Environment Group
HF_MODEL = os.getenv("HF_MODEL", "google/flan-t5-base")   # אפשר לשנות לדגם אחר
HF_TIMEOUT = int(os.getenv("HF_TIMEOUT", "25"))           # שניות


# CORS — לאפשר את האתר שלך (אפשר גם ["*"] אבל נקשיח לדומיין של GitHub Pages)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://maxost874.github.io"],  # ← עדכן לכתובת האתר שלך
    allow_origin_regex=r"https://.*\.github\.io",   # גיבוי לכל אתרי github.io
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=False,
)

# גיבוי: OPTIONS לכל נתיב (אם פרוקסי עוצר לפני ה-middleware)
@app.options("/{full_path:path}")
def options_catchall(full_path: str):
    return JSONResponse({"ok": True})

class PostData(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "API is running. Go to /docs, /health, /predict (POST), or /predict_qs (GET)."}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: PostData):
    text = data.text
    pred  = int(pipe.predict([text])[0])
    proba = float(pipe.predict_proba([text])[0, 1])
    sentiment = sia.polarity_scores(text)
    return {"prediction": pred, "probability": round(proba, 4), "sentiment": sentiment}

@app.get("/predict_qs")
def predict_qs(text: str = Query(..., min_length=1)):
    raw_text = unquote(text)
    pred  = int(pipe.predict([raw_text])[0])
    proba = float(pipe.predict_proba([raw_text])[0, 1])
    sentiment = sia.polarity_scores(raw_text)
    return {"prediction": pred, "probability": round(proba, 4), "sentiment": sentiment}


# ===== AI Improver (Hugging Face) =====
class ImproveReq(BaseModel):
    text: str

def _prompt(text: str) -> str:
    return f"""You are a sharp social media copywriter.
Rewrite the post in English for high engagement.
Output EXACTLY in this format (no extra text):
title: <3-6 word catchy title>
variant1: <first improved one-liner>
variant2: <second improved one-liner>
variant3: <third improved one-liner>
tags: #tag1 #tag2 #tag3 #tag4 #tag5

Original: {text}"""

async def _hf_complete(prompt: str) -> str:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN missing")
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 220, "temperature": 0.8, "top_p": 0.95},
        "options": {"wait_for_model": True}
    }
    async with httpx.AsyncClient(timeout=HF_TIMEOUT) as client:
        r = await client.post(url, headers=headers, json=payload)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(data["error"])
    return data[0].get("generated_text", "") if isinstance(data, list) else ""

def _parse(text: str):
    def grab(key):
        m = re.search(rf"^{key}\s*:\s*(.*)$", text, re.IGNORECASE | re.MULTILINE)
        return m.group(1).strip() if m else ""
    title = grab("title")
    variants = [grab("variant1"), grab("variant2"), grab("variant3")]
    tags_line = grab("tags")
    tags = re.findall(r"#\w+", tags_line)
    variants = [v for v in variants if v]
    return title, variants, tags

def _fallback_title(s: str) -> str:
    s = (s or "").strip()
    return (s[:32] + "…") if len(s) > 32 else (s or "Quick take")

def _fallback_variants(s: str):
    base = (s or "").strip() or "Your idea"
    return [
        f"{base} — catchy, tight pacing, one moment that pops.",
        f"{base} reimagined: bold hook, 10–15s beat, keep it ultra-simple.",
        "Short caption: small setup, clear payoff. Let the rhythm do the work."
    ]

def _fallback_tags(s: str):
    words = [w for w in re.findall(r"[A-Za-z]{3,}", s.lower())[:2]]
    base = ["#trend", "#viral", "#shorts", "#content", "#reels"]
    return [f"#{w}" for w in words] + base[: max(0, 5-len(words))]

@app.post("/improve")
async def improve(req: ImproveReq):
    txt = (req.text or "").strip()
    if len(txt) < 3:
        return {"ok": False, "error": "text too short"}
    try:
        out = await _hf_complete(_prompt(txt))
        title, variants, tags = _parse(out)
        if not variants:
            variants = _fallback_variants(txt)
        if not title:
            title = _fallback_title(txt)
        if not tags:
            tags = _fallback_tags(txt)
        return {"ok": True, "title": title, "suggestions": variants, "hashtags": tags}
    except Exception as e:
        return {
            "ok": True,
            "title": _fallback_title(txt),
            "suggestions": _fallback_variants(txt),
            "hashtags": _fallback_tags(txt),
            "detail": str(e)
        }
