# api/inference.py
"""
Load artifacts and run inference (XGBoost).
"""
import os
import joblib
import numpy as np
import pandas as pd

DEFAULT_PIPELINE = os.getenv("PIPELINE_PATH", "models/pipeline.joblib")
DEFAULT_MODEL = os.getenv("MODEL_PATH", "models/xgb_model.joblib")

EXPECTED_FEATURES = [
    'pkev12','pkev23','durP','tauPd','tauPt',
    'PDd','PVd','PAd','PDt','PVt','PAt',
    'ddt_PDd','ddt_PVd','ddt_PAd','ddt_PDt','ddt_PVt','ddt_PAt'
]

def load_artifacts(pipeline_path: str = DEFAULT_PIPELINE, model_path: str = DEFAULT_MODEL):
    if not os.path.exists(pipeline_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Pipeline or model missing. Run training and save into 'models/'.")
    pipeline = joblib.load(pipeline_path)
    model = joblib.load(model_path)
    return pipeline, model

def predict_from_features_dict(features: dict, pipeline, model):
    df = pd.DataFrame([{k: float(features[k]) for k in EXPECTED_FEATURES}])
    X_log = np.log1p(df)
    X_prep = pipeline.transform(X_log)
    preds_log = model.predict(X_prep)
    preds_raw = np.expm1(preds_log)
    return float(preds_log[0]), float(preds_raw[0])
