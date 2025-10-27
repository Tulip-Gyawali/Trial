# tests/test_features.py
import numpy as np
from src.utils.helpers import p_wave_features_calc

def test_p_wave_features_calc_simple():
    t = np.linspace(0, 1, 100)
    w = np.sin(2 * np.pi * 5 * t)
    feats = p_wave_features_calc(w, dt=t[1]-t[0])
    assert "pkev12" in feats and isinstance(feats["pkev12"], float)
    assert feats["pkev12"] > 0
