# src/models/evaluate_model.py
import joblib, os, numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = [
    'pkev12','pkev23','durP','tauPd','tauPt',
    'PDd','PVd','PAd','PDt','PVt','PAt',
    'ddt_PDd','ddt_PVd','ddt_PAd','ddt_PDt','ddt_PVt','ddt_PAt'
]

def evaluate(csv_path, pipeline_path="models/pipeline.joblib", model_path="models/xgb_model.joblib"):
    if not os.path.exists(pipeline_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Artifacts not found.")
    pipeline = joblib.load(pipeline_path)
    model = joblib.load(model_path)
    df = pd.read_csv(csv_path).dropna()
    X = np.log1p(df[FEATURES])
    y_log = np.log1p(df['PGA'])
    X_p = pipeline.transform(X)
    preds_log = model.predict(X_p)
    preds_raw = np.expm1(preds_log)
    y_raw = df['PGA'].values
    print("Log-space: R2", r2_score(y_log, preds_log), "MAE", mean_absolute_error(y_log, preds_log))
    print("Raw-space: R2", r2_score(y_raw, preds_raw), "MAE", mean_absolute_error(y_raw, preds_raw))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    evaluate(args.data)
