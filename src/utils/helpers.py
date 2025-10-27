# src/utils/helpers.py
import numpy as np

def p_wave_features_calc(window: np.ndarray, dt: float) -> dict:
    if window is None or len(window) == 0:
        return {k: np.nan for k in [
            "pkev12","pkev23","durP","tauPd","tauPt","PDd","PVd","PAd","PDt","PVt","PAt",
            "ddt_PDd","ddt_PVd","ddt_PAd","ddt_PDt","ddt_PVt","ddt_PAt"
        ]}
    durP = len(window) * dt
    PDd = np.max(window) - np.min(window)
    grad = np.gradient(window) / dt if dt != 0 else np.gradient(window)
    PVd = np.max(np.abs(grad)) if len(grad) > 0 else 0.0
    PAd = np.mean(np.abs(window))
    PDt = np.max(window)
    PVt = np.max(grad) if len(grad) > 0 else 0.0
    PAt = np.sqrt(np.mean(window ** 2))
    tauPd = durP / PDd if PDd != 0 else 0.0
    tauPt = durP / PDt if PDt != 0 else 0.0

    def ddt(x):
        x = np.asarray(x)
        if len(x) <= 1:
            return 0.0
        return np.mean(np.abs(np.gradient(x)))

    ddt_PDd = ddt(window)
    ddt_PVd = ddt(grad)
    ddt_PAd = ddt(np.abs(window))
    ddt_PDt = ddt(np.maximum(window, 0))
    ddt_PVt = ddt(grad)
    ddt_PAt = ddt(window ** 2)

    pkev12 = np.sum(window ** 2) / len(window)
    pkev23 = np.sum(np.abs(window)) / len(window)

    return {
        "pkev12": pkev12, "pkev23": pkev23,
        "durP": durP, "tauPd": tauPd, "tauPt": tauPt,
        "PDd": PDd, "PVd": PVd, "PAd": PAd,
        "PDt": PDt, "PVt": PVt, "PAt": PAt,
        "ddt_PDd": ddt_PDd, "ddt_PVd": ddt_PVd,
        "ddt_PAd": ddt_PAd, "ddt_PDt": ddt_PDt,
        "ddt_PVt": ddt_PVt, "ddt_PAt": ddt_PAt
    }
