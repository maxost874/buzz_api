import os, joblib, nltk
import re, random, httpx, json


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
HF_TIMEOUT = int(os.getenv("HF_TIMEOUT", "45"))           # שניות



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
    # מבקש החזרה כ-JSON "יבש" בלבד – הכי קל לפרסר
    return (
        "Rewrite the input for social media in English.\n"
        "Return ONLY compact JSON with this exact schema (no prose, no extra text):\n"
        '{"title":"<3-6 words>","variants":["<v1>","<v2>","<v3>"],"tags":["#tag1","#tag2","#tag3","#tag4","#tag5"]}\n'
        "Rules: 3 concise variants (<=20 words), no emojis, no hashtags inside variants, "
        "do not repeat the input verbatim, catchy & simple.\n"
        f"Input: {text}\nJSON:"
    )


async def _hf_complete(prompt: str) -> str:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN missing")
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 220,
            "temperature": 0.8,
            "top_p": 0.95,
            "return_full_text": False
        },
        "options": {"wait_for_model": True}
    }
    async with httpx.AsyncClient(timeout=HF_TIMEOUT) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()

    # רוב מודלי text2text מחזירים [{"generated_text": "..."}]
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0].get("generated_text", "") or ""

    # לעיתים מוחזר dict עם error
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(data["error"])

    # גיבוי: המרה לטקסט
    return str(data) if data is not None else ""


def _parse(text: str):
    text = (text or "").strip()

    # 1) ניסיון ראשון: JSON מלא
    try:
        obj = json.loads(text)
        title = (obj.get("title") or "").strip()
        variants = [v.strip() for v in obj.get("variants", []) if v and v.strip()]
        tags = []
        for t in obj.get("tags", []):
            t = (t or "").strip()
            if not t:
                continue
            if not t.startswith("#"):
                t = "#" + re.sub(r"[^A-Za-z0-9]", "", t)
            tags.append(t[:16])
        return title, variants[:3], tags[:6]
    except Exception:
        pass

    # 2) לפעמים המודל מחזיר כמה שורות, כשה-JSON באחרונה
    for line in text.splitlines()[::-1]:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            title = (obj.get("title") or "").strip()
            variants = [v.strip() for v in obj.get("variants", []) if v and v.strip()]
            tags = []
            for t in obj.get("tags", []):
                t = (t or "").strip()
                if not t:
                    continue
                if not t.startswith("#"):
                    t = "#" + re.sub(r"[^A-Za-z0-9]", "", t)
                tags.append(t[:16])
            return title, variants[:3], tags[:6]
        except Exception:
            continue

    # 3) גיבוי: אם אין JSON בכלל – מחזירים ריקים כדי שה-fallback יעבוד
    return "", [], []


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
            src = "fallback"
        else:
            src = "hf"

        if not title:
            title = _fallback_title(txt)
        if not tags:
            tags = _fallback_tags(txt)

        return {"ok": True, "source": src, "title": title, "suggestions": variants, "hashtags": tags}

    except Exception as e:
        return {
            "ok": True,
            "source": "fallback",
            "title": _fallback_title(txt),
            "suggestions": _fallback_variants(txt),
            "hashtags": _fallback_tags(txt),
            "detail": str(e)
        }

@app.get("/debug/hf")
def debug_hf():
    return {"has_token": bool(HF_TOKEN), "model": HF_MODEL}


