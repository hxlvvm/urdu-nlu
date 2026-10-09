from fastapi.testclient import TestClient

from nlu.data import spans
from serve.app import app, get_predictor


class FakePredictor:
    def predict_text(self, text):
        words = text.split()
        tags = ["B-time" if w.isdigit() else "O" for w in words]
        return {"intent": "alarm_set", "confidence": 0.9, "tags": tags, "tokens": words, "slots": spans(words, tags)}


app.dependency_overrides[get_predictor] = FakePredictor
client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_shape():
    r = client.post("/predict", json={"text": "wake me at 7"})
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "alarm_set" and body["slots"] == {"time": "7"}
    assert len(body["tags"]) == len(body["tokens"]) == 4 and body["latency_ms"] >= 0


def test_predict_validates_input():
    assert client.post("/predict", json={"text": ""}).status_code == 422
    assert client.post("/predict", json={"text": "x" * 501}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422
