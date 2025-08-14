import os, joblib, nltk
from fastapi import FastAPI
from pydantic import BaseModel
from nltk.sentiment import SentimentIntensityAnalyzer
from fastapi.middleware.cors import CORSMiddleware

# הורדת VADER אם חסר
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

MODEL_PATH = os.getenv("MODEL_PATH", "pipeline_v1.joblib")
pipe = joblib.load(MODEL_PATH)
sia = SentimentIntensityAnalyzer()

app = FastAPI(title="Buzz Predictor API")

# CORS – לאפשר חיבור מהאתר שלך
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

class PostData(BaseModel):
    text: str

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
