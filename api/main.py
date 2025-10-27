# api/main.py
from fastapi import FastAPI, HTTPException
from api.schemas import PWaveFeatures
from api.inference import load_artifacts, predict_from_features_dict

app = FastAPI(title="PGA-from-P-wave API")

try:
    pipeline, model = load_artifacts()
except Exception as e:
    pipeline, model = None, None
    startup_err = str(e)
else:
    startup_err = None

@app.get("/")
def root():
    return {"status": "ok", "note": "POST 17 P-wave features as JSON to /predict"}

@app.post("/predict")
def predict(payload: PWaveFeatures):
    if startup_err:
        raise HTTPException(status_code=500, detail=f"Model artifacts not loaded: {startup_err}")
    features = payload.__root__
    expected = [
        'pkev12','pkev23','durP','tauPd','tauPt',
        'PDd','PVd','PAd','PDt','PVt','PAt',
        'ddt_PDd','ddt_PVd','ddt_PAd','ddt_PDt','ddt_PVt','ddt_PAt'
    ]
    missing = [f for f in expected if f not in features]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")
    plog, praw = predict_from_features_dict(features, pipeline, model)
    return {"pred_log": plog, "pred_raw": praw}
