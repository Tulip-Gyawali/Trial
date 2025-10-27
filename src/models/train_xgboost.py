# src/models/train_xgboost.py
import argparse, os, joblib, json
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

FEATURES = [
    'pkev12','pkev23','durP','tauPd','tauPt',
    'PDd','PVd','PAd','PDt','PVt','PAt',
    'ddt_PDd','ddt_PVd','ddt_PAd','ddt_PDt','ddt_PVt','ddt_PAt'
]

DEFAULT_PARAMS = {
    'n_estimators': 776,
    'learning_rate': 0.010590433420511285,
    'max_depth': 6,
    'subsample': 0.666852461341688,
    'colsample_bytree': 0.8724127328229327,
    'objective': 'reg:squarederror',
    'random_state': 42
}

def build_pipeline():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', RobustScaler()),
        ('selector', SelectKBest(score_func=f_regression, k='all'))
    ])

def train_and_save(csv_path, out_dir="models", params=None):
    params = params or DEFAULT_PARAMS
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path).dropna()
    X = np.log1p(df[FEATURES])
    y = np.log1p(df['PGA'])
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = build_pipeline()
    pipeline.fit(Xtr, ytr)
    Xtr_p = pipeline.transform(Xtr)
    Xte_p = pipeline.transform(Xte)
    model = xgb.XGBRegressor(**params)
    model.fit(Xtr_p, ytr)
    preds = model.predict(Xte_p)
    print("Validation MAE (log-space):", mean_absolute_error(yte, preds))
    joblib.dump(pipeline, os.path.join(out_dir, "pipeline.joblib"))
    joblib.dump(model, os.path.join(out_dir, "xgb_model.joblib"))
    with open(os.path.join(out_dir, "best_params.json"), "w") as f:
        json.dump(params, f, indent=2)
    print("Saved XGB artifacts to", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="models")
    args = parser.parse_args()
    train_and_save(args.data, args.out)
