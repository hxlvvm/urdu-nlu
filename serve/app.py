"""FastAPI service: POST /predict {"text": "..."} -> intent, confidence and slots (English or Urdu)."""
import os
import time
from functools import lru_cache

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="urdu-nlu", version="1.0",
              description="Bilingual (English/Urdu) intent detection and slot filling, int8 ONNX on CPU.")


class Query(BaseModel):
    text: str = Field(min_length=1, max_length=500, examples=["مجھے کل صبح سات بجے جگا دیں"])


class Prediction(BaseModel):
    intent: str
    confidence: float
    slots: dict[str, str]
    tokens: list[str]
    tags: list[str]
    latency_ms: float


@lru_cache(maxsize=1)
def get_predictor():
    from nlu.predictor import Predictor
    return Predictor(os.environ.get("NLU_ARTIFACTS", "artifacts"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(q: Query, predictor=Depends(get_predictor)):
    t0 = time.perf_counter()
    r = predictor.predict_text(q.text)
    return {**r, "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}
