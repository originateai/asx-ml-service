"""
ASX Layer 7 ML Microservice
FastAPI service that loads the XGBoost model and returns buy probability
Deploy on Railway: https://railway.app
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import os

app = FastAPI(title="ASX Layer 7 ML", version="1.0")

# Load model on startup
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        import joblib
        model_path = os.path.join(os.path.dirname(__file__), 'asx_layer7_model.joblib')
        model = joblib.load(model_path)
        print(f"Model loaded: {type(model).__name__}")
    except Exception as e:
        print(f"Model load failed: {e}")

class StockFeatures(BaseModel):
    # Top features from backtest (in order model was trained)
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

class PredictionResponse(BaseModel):
    ticker:      str
    probability: float
    signal:      str
    confidence:  str

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict/{ticker}", response_model=PredictionResponse)
def predict(ticker: str, features: StockFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Feature order must match training order
    feature_cols = [
        'rsi14', 'sma20', 'sma50', 'sma200', 'bb_pos',
        'vol_ratio', 'roc5', 'roc20',
        'pct_from_sma20', 'pct_from_sma200',
        'above_sma20', 'above_sma200', 'golden_cross',
        'hammer', 'bull_candle', 'lower_shadow', 'upper_shadow',
        'body', 'range'
    ]

    X = np.array([[getattr(features, f) for f in feature_cols]])

    try:
        proba = float(model.predict_proba(X)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    signal = "BUY" if proba >= 0.65 else "WATCH" if proba >= 0.50 else "PASS"
    confidence = "HIGH" if proba >= 0.65 else "MEDIUM" if proba >= 0.55 else "LOW"

    return PredictionResponse(
        ticker=ticker.upper(),
        probability=round(proba, 4),
        signal=signal,
        confidence=confidence
    )

@app.post("/predict-batch")
def predict_batch(items: list[dict]):
    """Predict for multiple stocks at once"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    feature_cols = [
        'rsi14', 'sma20', 'sma50', 'sma200', 'bb_pos',
        'vol_ratio', 'roc5', 'roc20',
        'pct_from_sma20', 'pct_from_sma200',
        'above_sma20', 'above_sma200', 'golden_cross',
        'hammer', 'bull_candle', 'lower_shadow', 'upper_shadow',
        'body', 'range'
    ]

    results = []
    tickers = [item.get('ticker', 'UNKNOWN') for item in items]
    X = np.array([[item.get(f, 0.0) for f in feature_cols] for item in items])

    try:
        probas = model.predict_proba(X)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {e}")

    for ticker, proba in zip(tickers, probas):
        proba = float(proba)
        results.append({
            "ticker":      ticker.upper(),
            "probability": round(proba, 4),
            "signal":      "BUY" if proba >= 0.65 else "WATCH" if proba >= 0.50 else "PASS",
            "confidence":  "HIGH" if proba >= 0.65 else "MEDIUM" if proba >= 0.55 else "LOW"
        })

    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
