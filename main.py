"""
ASX Layer 7 ML Microservice v2
FastAPI service — retrained with correct Wilder's RSI and adjusted_close prices
Feature order matches retrain_ml.py exactly
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import os

app = FastAPI(title="ASX Layer 7 ML v2", version="2.0")

model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        import joblib
        # Try v2 model first, fall back to v1
        for name in ['asx_layer7_model_v2.joblib', 'asx_layer7_model.joblib']:
            path = os.path.join(os.path.dirname(__file__), name)
            if os.path.exists(path):
                model = joblib.load(path)
                print(f"Model loaded: {name} ({type(model).__name__})")
                return
        print("No model file found")
    except Exception as e:
        print(f"Model load failed: {e}")

# Feature order must match retrain_ml.py exactly
FEATURE_COLS = [
    'rsi14', 'sma20', 'sma50', 'sma200', 'bb_pos',
    'vol_ratio', 'roc5', 'roc20',
    'pct_from_sma20', 'pct_from_sma200',
    'above_sma20', 'above_sma200', 'golden_cross',
    'hammer', 'bull_candle', 'lower_shadow', 'upper_shadow',
    'body', 'range'
]

class StockFeatures(BaseModel):
    rsi14:            float = 50.0
    sma20:            float = 0.0
    sma50:            float = 0.0
    sma200:           float = 0.0
    bb_pos:           float = 0.5
    vol_ratio:        float = 1.0
    roc5:             float = 0.0
    roc20:            float = 0.0
    pct_from_sma20:   float = 0.0
    pct_from_sma200:  float = 0.0
    above_sma20:      float = 1.0
    above_sma200:     float = 1.0
    golden_cross:     float = 0.0
    hammer:           float = 0.0
    bull_candle:      float = 0.0
    lower_shadow:     float = 0.0
    upper_shadow:     float = 0.0
    body:             float = 0.0
    range:            float = 0.0

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "version": "2.0"}

@app.post("/predict/{ticker}")
def predict(ticker: str, features: StockFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    X = np.array([[getattr(features, f) for f in FEATURE_COLS]])
    try:
        proba = float(model.predict_proba(X)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "ticker":      ticker.upper(),
        "probability": round(proba, 4),
        "signal":      "BUY" if proba >= 0.55 else "WATCH" if proba >= 0.45 else "PASS",
        "confidence":  "HIGH" if proba >= 0.65 else "MEDIUM" if proba >= 0.55 else "LOW"
    }

@app.post("/predict-batch")
def predict_batch(items: list[dict]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    X = np.array([[item.get(f, 0.0) for f in FEATURE_COLS] for item in items])
    try:
        probas = model.predict_proba(X)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return [{"ticker": item.get('ticker','').upper(), "probability": round(float(p),4),
             "signal": "BUY" if p>=0.55 else "WATCH" if p>=0.45 else "PASS",
             "confidence": "HIGH" if p>=0.65 else "MEDIUM" if p>=0.55 else "LOW"}
            for item, p in zip(items, probas)]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
