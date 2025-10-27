# tests/test_api.py
import json
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def make_sample_features():
    return {
      "pkev12": 0.5,"pkev23": 0.4,"durP": 0.03,"tauPd": 0.001,"tauPt": 0.0012,
      "PDd": 0.1,"PVd": 0.2,"PAd": 0.05,"PDt": 0.12,"PVt": 0.22,"PAt": 0.06,
      "ddt_PDd": 0.0001,"ddt_PVd": 0.0002,"ddt_PAd": 0.00005,"ddt_PDt": 0.00012,"ddt_PVt": 0.00022,"ddt_PAt": 0.00006
    }

def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_bad_request():
    r = client.post("/predict", json={"some": 1})
    assert r.status_code in (400, 500)

def test_predict_sample():
    r = client.post("/predict", json=make_sample_features())
    assert r.status_code in (200, 400, 500)
