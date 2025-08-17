import os, joblib, nltk
import re, httpx, json


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

# ===== AI improver via Google Gemini (English-only) =====
from typing import Optional
from fastapi import HTTPException

# ENV config (להגדיר ב-Render → Environment)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash").strip()
GEMINI_TIMEOUT = int(os.environ.get("GEMINI_TIMEOUT", "45"))

def _prompt_en(text: str) -> str:
    return (
        "Rewrite the following social media post in ENGLISH.\n"
        "Return ONLY compact JSON with EXACTLY this schema (no prose, no backticks):\n"
        '{"title":"<3-6 words>","variants":["<v1>","<v2>","<v3>"],"tags":["#tag1","#tag2","#tag3","#tag4","#tag5"]}\n'
        "Rules: 3 concise variants (<=24 words each), catchy & simple, no emojis, "
        "no hashtags inside variants, do NOT repeat the input verbatim.\n"
        f"Post: {text}\nJSON:"
    )

async def _gemini_complete(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY missing (set it in Render Environment).")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "topP": 0.95, "maxOutputTokens": 256},
    }
    async with httpx.AsyncClient(timeout=GEMINI_TIMEOUT) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        raise RuntimeError(f"Bad Gemini response: {data}")

def _parse_ai_json(text: str):
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip(), flags=re.IGNORECASE | re.DOTALL)
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        s = m.group(0)
    try:
        obj = json.loads(s)
        title = (obj.get("title") or "").strip()
        variants = [v.strip() for v in obj.get("variants", []) if v and v.strip()]
        tags = []
        for t in obj.get("tags", []):
            t = (t or "").strip()
            if not t:
                continue
            if not t.startswith("#"):
                t = "#" + re.sub(r"[^A-Za-z0-9]", "", t)
            tags.append(t[:20])
        return title, variants[:3], tags[:6]
    except Exception:
        return "", [], []

class ImproveReq(BaseModel):
    text: str
    max_words: Optional[int] = 60  # לתאימות ל-frontend (לא חובה למודל)

@app.post("/improve")
async def improve(req: ImproveReq):
    txt = (req.text or "").strip()
    if len(txt) < 3:
        raise HTTPException(400, "text too short")
    out = await _gemini_complete(_prompt_en(txt))
    title, variants, tags = _parse_ai_json(out)
    if not variants:
        return {"ok": False, "error": "Model returned no variants / JSON parse failed", "raw": (out or "")[:400]}
    if not tags:
        tags = [f"#{w}" for w in re.findall(r"[A-Za-z0-9]{3,}", txt.lower())[:5]]
    return {"ok": True, "source": "gemini", "title": title or "Improved post",
            "suggestions": variants, "hashtags": tags}
# ===== end Gemini improver =====










