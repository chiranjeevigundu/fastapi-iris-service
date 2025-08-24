from fastapi.testclient import TestClient
from src.app import app
from src.train import main as train_main

def setup_module(module):
    # ensure model exists for tests
    train_main()

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True

def test_predict():
    payload = {"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert "prediction" in r.json()
