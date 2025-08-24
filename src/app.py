from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "model.joblib"
app = FastAPI(title="Iris Service", version="1.0")

class IrisIn(BaseModel):
    sepal_length: float = Field(..., ge=0)
    sepal_width: float = Field(..., ge=0)
    petal_length: float = Field(..., ge=0)
    petal_width: float = Field(..., ge=0)

@app.get("/health")
def health():
    return {"ok": True}

def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run train.py first.")
    return joblib.load(MODEL_PATH)

@app.post("/predict")
def predict(x: IrisIn):
    model = _load_model()
    vec = np.array([[x.sepal_length, x.sepal_width, x.petal_length, x.petal_width]])
    proba = model.predict_proba(vec)[0].tolist()
    pred = int(model.predict(vec)[0])
    target = ["setosa","versicolor","virginica"][pred]
    return {"prediction": target, "proba": proba}
