# ASX Layer 7 ML Service

FastAPI microservice for ASX stock buy probability predictions.

## Deploy on Railway

1. Create account at railway.app
2. New Project → Deploy from folder
3. Upload this folder INCLUDING `asx_layer7_model.joblib` from your Downloads
4. Railway auto-detects Python and deploys
5. Copy the generated URL

## Test

GET  /health
POST /predict/BHP  (with feature JSON body)
POST /predict-batch (array of feature objects)

## Environment Variables (optional)
None required — model loads from file
