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


# ===== AI Improver (minimal, no external API) =====

# stopwords בסיסיות כדי לא להפוך אותן ל-hashtag
COMMON_EN_STOP = {
    "the","and","for","with","this","that","will","have","has","are","was","were",
    "you","your","from","today","post","about","just","like","really","very",
    "https","http","www","com","net","org"
}

def _top_keywords(text: str, k: int = 6):
    words = re.findall(r"[A-Za-z#@][\w'-]{2,}", text.lower())
    bag = {}
    for w in words:
        if w in COMMON_EN_STOP or w.startswith(("http","www")):
            continue
        bag[w] = bag.get(w, 0) + 1
    # מיון לפי שכיחות ואז לפי אורך מילה (קצת “איכות”)
    return [w for w, _ in sorted(bag.items(), key=lambda x: (-x[1], -len(x[0])) )[:k]]

def _make_hashtags(keys):
    tags = []
    for w in keys:
        w = re.sub(r"[^A-Za-z0-9]", "", w)
        if w:
            tags.append("#" + (w if len(w) <= 15 else w[:15]))
    # ייחודיים, עד 6 תגים
    out = []
    for t in tags:
        if t not in out:
            out.append(t)
        if len(out) >= 6:
            break
    return out

def _emoji_by_sent(sent_compound: float) -> str:
    if sent_compound >= 0.3:  return "🔥"
    if sent_compound <= -0.3: return "🤔"
    return "✨"

class ImproveReq(BaseModel):
    text: str

@app.post("/improve")
def improve(req: ImproveReq):
    t = (req.text or "").strip()
    if len(t) < 3:
        return {"ok": False, "error": "text too short"}

    # סנטימנט קטן בשביל נופך
    sent = sia.polarity_scores(t)["compound"]
    emo  = _emoji_by_sent(sent)

    # הוק = משפט ראשון/60–120 תווים
    first = re.split(r"[.!?\n]", t, maxsplit=1)[0].strip() or t
    hook  = (first[:120]).strip()

    # 3 וריאציות קצרות
    suggestions = [
        f"Hot take: {hook} {emo}\n\n{t}\n\nWhat do you think?",
        f"Looking to boost engagement? {emo}\n\n{t}\n\nShare your take below.",
        f"Real talk → {hook} {emo}\n\n{t}\n\nAgree or disagree?",
    ]

    # hashtags פשוטים
    hashtags = _make_hashtags(_top_keywords(t, 6))

    title = (hook[:60] + ("…" if len(hook) > 60 else "")) or "Post idea"

    return {
        "ok": True,
        "title": title,
        "suggestions": suggestions,
        "hashtags": hashtags
    }


