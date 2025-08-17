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


# ===== AI Improver (rewrite — no CTA, no verbatim repeat) =====
import re, random

ADJS_EN = ["playful", "catchy", "nostalgic", "bold", "fresh", "wholesome", "cheeky", "uplifting"]
TONES_EN = ["remix", "mini-trend", "quick reel", "throwback", "short loop", "summer vibe"]

HEB_RE = re.compile(r"[\u0590-\u05FF]")

def _clean(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip()

def _emoji_by_sent(comp: float) -> str:
    if comp >= 0.3:  return "🔥"
    if comp <= -0.3: return "🤔"
    return "✨"

def _keywords_en(text: str, k: int = 6):
    # מילות מפתח פשוטות לאנגלית בלבד (בלי # או @)
    words = re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", text.lower())
    seen, out = set(), []
    for w in words:
        if w.startswith(("http","www")): 
            continue
        if w not in seen:
            seen.add(w); out.append(w)
    return out[:k]

def _hashtags_from_en(keys):
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

def _titlecase(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s

def _make_en_variants(text: str):
    # לא נעתיק את הטקסט המקורי; נלביש סביבו ניסוחים חדשים
    keys = _keywords_en(text, 6)
    subject = " ".join(keys[:2]) if keys else "the idea"
    tone = random.choice(TONES_EN)
    adj1, adj2 = random.sample(ADJS_EN, 2)

    v1 = f"{_titlecase(subject)} {tone} — {adj1} and fun. Tiny hook, quick cut, instant smile."
    v2 = f"{_titlecase(subject)} reimagined: {adj2} beat, 10–15s loop, one clear moment that pops."
    v3 = f"Short caption: Little fins, big energy. Make the rhythm do the work."

    # אם אין מילים משמעותיות, ניפול לנוסחים כלליים טובים
    if not keys:
        v1 = "Fresh take — short, punchy, and easy to share."
        v2 = "Reimagine it as a 10–15s reel with one bold line."
        v3 = "Keep it tiny: one vibe, one visual, one beat."

    # כותרת קצרה מתוך מילת המפתח/נושא
    title = (_titlecase(subject) or "Post idea")
    tags  = _hashtags_from_en(keys)
    return title, [v1, v2, v3], tags

from pydantic import BaseModel
class ImproveReq(BaseModel):
    text: str

@app.post("/improve")
def improve(req: ImproveReq):
    base = _clean(req.text)
    if len(base) < 3:
        return {"ok": False, "error": "text too short"}

    sent = sia.polarity_scores(base)["compound"]
    emo  = _emoji_by_sent(sent)

    if HEB_RE.search(base):
        # ניסוחים כלליים בעברית (ללא CTA וללא חזרה מילה במילה)
        title = "רעיון לפוסט"
        v1 = f"{emo} גרסה קצרה ושובבה — פתיח חד, ויז׳ואל חזק, וקצב מהיר."
        v2 = "מחדש את הרעיון: 10–15 שניות, משפט אחד בולט ותנועה שמושכת עין."
        v3 = "כיתוב אפשרי: קטן אבל בועט. רמיזה בטעם טוב והמשך בוידאו."
        suggestions = [v1, v2, v3]
        hashtags = []
    else:
        title, suggestions, hashtags = _make_en_variants(base)
        # מוסיפים אמוג'י קטן לפתיחה של כל נוסח
        suggestions = [f"{emo} {s}" for s in suggestions]

    return {"ok": True, "title": title, "suggestions": suggestions, "hashtags": hashtags}



