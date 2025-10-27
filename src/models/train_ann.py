# src/models/train_ann.py
"""
Train an ANN (PyTorch) using the optimal hyperparameters supplied (defaults to your best).
Saves:
 - models/pipeline_ann.joblib (preprocessing)
 - models/ann_model.pt (state_dict)
 - models/ann_best_params.json
"""
import argparse, os, json, joblib
import numpy as np, pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from copy import deepcopy

FEATURES = [
    'pkev12','pkev23','durP','tauPd','tauPt',
    'PDd','PVd','PAd','PDt','PVt','PAt',
    'ddt_PDd','ddt_PVd','ddt_PAd','ddt_PDt','ddt_PVt','ddt_PAt'
]

# Your optimal hyperparameters (from notebook)
DEFAULT_ANN_PARAMS = {
    "dropout": 0.2835776221997114,
    "epochs": 829,
    "hidden_sizes": [428, 442, 220],
    "input_size": 17,
    "lr": 0.0011676487575205433,
    "weight_decay": 6.370451204388144e-05
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout):
        super().__init__()
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def build_pipeline():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', RobustScaler()),
        ('selector', SelectKBest(score_func=f_regression, k='all'))
    ])

def train_ann(csv_path, out_dir="models", params=None):
    params = params or DEFAULT_ANN_PARAMS
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path).dropna()
    X = df[FEATURES]
    y_raw = df['PGA']
    X_log = np.log1p(X)
    y_log = np.log1p(y_raw)

    # simple split (80/20)
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X_log, y_log, test_size=0.2, random_state=42)

    pipeline = build_pipeline()
    pipeline.fit(Xtr, ytr)
    Xtr_p = pipeline.transform(Xtr)
    Xte_p = pipeline.transform(Xte)

    # convert to tensors
    Xtr_t = torch.tensor(Xtr_p, dtype=torch.float32).to(device)
    ytr_t = torch.tensor(ytr.values, dtype=torch.float32).view(-1,1).to(device)
    Xte_t = torch.tensor(Xte_p, dtype=torch.float32).to(device)
    yte_t = torch.tensor(yte.values, dtype=torch.float32).view(-1,1).to(device)

    model = NeuralNetwork(params['input_size'], params['hidden_sizes'], params['dropout']).to(device)
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    best_state = None
    best_val_loss = float('inf')
    patience, wait = 50, 0

    epochs = int(params['epochs'])
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(Xtr_t), ytr_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xte_t), yte_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # predictions and metrics
    model.eval()
    with torch.no_grad():
        preds_log = model(Xte_t).cpu().numpy().flatten()
    mae_log = mean_absolute_error(yte, preds_log)
    print(f"ANN Validation MAE (log-space): {mae_log:.6f}")

    # save artifacts
    joblib.dump(pipeline, os.path.join(out_dir, "pipeline_ann.joblib"))
    torch.save(model.state_dict(), os.path.join(out_dir, "ann_model.pt"))
    with open(os.path.join(out_dir, "ann_best_params.json"), "w") as f:
        json.dump(params, f, indent=2)
    print("Saved ANN artifacts to", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="models")
    args = parser.parse_args()
    train_ann(args.data, args.out)
