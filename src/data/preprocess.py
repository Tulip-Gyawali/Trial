# src/data/preprocess.py
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

FEATURES = [
    'pkev12','pkev23','durP','tauPd','tauPt',
    'PDd','PVd','PAd','PDt','PVt','PAt',
    'ddt_PDd','ddt_PVd','ddt_PAd','ddt_PDt','ddt_PVt','ddt_PAt'
]

def load_and_clean(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for c in df.columns:
        if c not in ['station','network','starttime','sampling_rate']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna()
    df = df[(df[FEATURES] > 0).all(axis=1)]
    return df

def stratified_split(df, target_col='PGA', test_size=0.2, random_state=42):
    y_log = np.log1p(df[target_col])
    y_bins = pd.qcut(y_log, q=10, labels=False, duplicates='drop')
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tr_idx, te_idx = next(sss.split(df, y_bins))
    return df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)
