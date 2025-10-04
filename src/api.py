# src/api.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import re
import scipy.sparse as sp

# -----------------------------
# FastAPI Lifespan for startup
# -----------------------------
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model and vectorizer once at startup
    global vectorizer, classifier, scaler
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
    VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
    MODEL_PATH = os.path.join(MODEL_DIR, "classifier.joblib")
    SCALER_PATH = os.path.join(MODEL_DIR, "meta_scaler.joblib")

    vectorizer = joblib.load(VECTORIZER_PATH)
    classifier = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    yield

app = FastAPI(lifespan=lifespan)

# -----------------------------
# Request model
# -----------------------------
class EmailInput(BaseModel):
    subject: str = ""
    body: str = ""
    n_urls: int = 0
    has_ip_in_url: int = 0
    suspicious_tld: int = 0

# -----------------------------
# Utility for combining text + meta
# -----------------------------
def prepare_features(email: EmailInput):
    combined_text = f"{email.subject or ''} {email.body or ''}".strip().lower()
    X_text = vectorizer.transform([combined_text])
    # meta features as array
    X_meta = [[email.n_urls, email.has_ip_in_url, email.suspicious_tld]]
    X_meta_scaled = scaler.transform(X_meta)
    # combine sparse text features + scaled meta features
    X = sp.hstack([X_text, X_meta_scaled])
    return X

# -----------------------------
# Predict endpoint
# -----------------------------
@app.post("/predict")
def predict(email: EmailInput):
    X = prepare_features(email)
    y_pred = classifier.predict(X)[0]
    proba = classifier.predict_proba(X)[0][y_pred]

    label = "flagged" if y_pred == 1 else "safe"  # <-- convert numeric to text

    return {"prediction": label, "confidence": float(proba)}

