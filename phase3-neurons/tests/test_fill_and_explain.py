import os
from fastapi.testclient import TestClient
from app.main import app

API_KEY = os.environ.get("NEURONS_API_KEY", "SYNAPSE@2025")
client = TestClient(app)
_headers = {"x-api-key": API_KEY, "content-type": "application/json"}

def test_build_iem():
    r = client.post("/iem/build", headers=_headers, json={"uemPath":"./uem.json","dim":256})
    assert r.status_code == 200 and r.json()["ok"]

def test_fill_basic():
    r = client.post("/synapse/fill", headers=_headers, json={"intent":{"ask":"top_k","metric":{"op":"sum"}}, "topKTargets":6})
    j = r.json()
    assert r.status_code == 200 and j["ok"]
    assert j["intentFilled"]["target"]

def test_explain_basic():
    r = client.post("/synapse/explain", headers=_headers, json={"intent":{"ask":"top_k","metric":{"op":"sum"}}, "topKTargets":6})
    j = r.json()
    assert r.status_code == 200 and j["ok"]
    assert "targetChosen" in j and isinstance(j["explains"], list)

