import os, joblib, nltk
import re, random

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


# ===== AI Improver — English only, no CTA, no verbatim repeat =====
COMMON_EN_STOP = {
    "the","and","for","with","this","that","will","have","has","are","was","were","you","your",
    "from","about","just","like","really","very","today","post","www","http","https","com","net","org"
}
ADJS_EN  = ["catchy","bold","fresh","playful","nostalgic","punchy","clean","uplifting","cheeky"]
FORMATS  = ["mini-trend","quick reel","short loop","remix","throwback","snackable clip","micro edit"]

def _clean(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip()

def _keywords_en(text: str, k: int = 6):
    words = re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", text.lower())
    bag = {}
    for w in words:
        if w in COMMON_EN_STOP or w.startswith(("http","www")):
            continue
        bag[w] = bag.get(w, 0) + 1
    # sort by freq then length
    keys = [w for w, _ in sorted(bag.items(), key=lambda x: (-x[1], -len(x[0])))]
    return keys[:k]

def _titlecase(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s

def _hashtags(keys):
    tags = []
    for w in keys:
        w = re.sub(r"[^A-Za-z0-9]", "", w)
        if not w: 
            continue
        tag = "#" + (w if len(w) <= 15 else w[:15])
        if tag not in tags:
            tags.append(tag)
        if len(tags) >= 6:
            break
    return tags

def _variants_en(text: str):
    keys = _keywords_en(text, 6)
    subject = " ".join(keys[:2]).strip() or "the idea"
    adj1, adj2 = random.sample(ADJS_EN, 2)
    fmt       = random.choice(FORMATS)

    # שלושה נוסחים שונים שלא מחזירים את הטקסט המקורי
    v1 = f"{_titlecase(subject)} {fmt} — {adj1}, tight pacing, one moment that pops."
    v2 = f"{_titlecase(subject)} reimagined: {adj2} hook, 10–15s beat, keep it ultra-simple."
    v3 = "Short caption: small setup, clear payoff. Let the rhythm do the work."

    if not keys:
        v1 = "Fresh take — short, punchy and easy to share."
        v2 = "Think 10–15s with a single bold visual and a clean hook."
        v3 = "One vibe, one beat, one moment. Keep it tiny."

    title = _titlecase(subject) if subject != "the idea" else "Post idea"
    tags  = _hashtags(keys)
    return title, [v1, v2, v3], tags

class ImproveReq(BaseModel):
    text: str

@app.post("/improve")
def improve(req: ImproveReq):
    base = _clean(req.text)
    if len(base) < 3:
        return {"ok": False, "error": "text too short"}

    title, suggestions, hashtags = _variants_en(base)
    return {"ok": True, "title": title, "suggestions": suggestions, "hashtags": hashtags}
