import os, joblib, nltk
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
